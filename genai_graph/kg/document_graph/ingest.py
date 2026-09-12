"""Direct ingestion of a Document Graph into a graph backend.

Builds a ``Folder → Document → MarkdownSection`` tree. The section hierarchy is
self-referential (`MarkdownSection -> MarkdownSection`), which doesn't map onto
the generic Pydantic-nesting extraction used elsewhere in genai-graph
(`extract_graph_data`). This module instead builds `NodeDataCollection` /
`RelationshipRecord` objects directly from a `DocumentGraphFactory` and merges
them with the same Arrow/Ladybug primitives (`merge_nodes_batch`,
`merge_relationships_batch`) used by the rest of the ingestion pipeline.

Documents are keyed by content hash, so re-ingesting an unchanged corpus is a
MERGE no-op. Before parsing sections for a document, the DB is checked for an
existing Document with the same ``markdown_hash`` — when present, section
creation is skipped entirely, avoiding costly recomputation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.backend import KgBackend, KuzuBackend
from genai_graph.kg.document_graph.retrieval import (
    RetrievalConfig,
    RetrievalError,
    attach_chunk_embeddings,
    build_sections_chunks,
    ensure_chunk_embedding_column,
    ensure_section_fts_index,
    prepare_chunk_inputs,
    resolve_embedding_dimension,
)
from genai_graph.kg.embeddings_handler import EmbeddingsHandler
from genai_graph.kg.factories.document_graph_factory import DocumentGraphBundle, DocumentGraphFactory
from genai_graph.kg.ingest.extract import RelationshipRecord, create_schema
from genai_graph.kg.ingest.merge import (
    NodeDataCollection,
    NodeTypeRegistry,
    merge_nodes_batch,
    merge_relationships_batch,
)
from genai_graph.kg.nodes.document import (
    CONTAINS_DOC,
    HAS_SUBFOLDER,
    Document,
    DocumentNode,
    FolderNode,
)
from genai_graph.kg.nodes.document_section import (
    HAS_CHUNK,
    HAS_IMAGE,
    HAS_SECTION,
    HAS_SUBSECTION,
    ImageNode,
    SectionChunkNode,
    SectionNode,
)

_FOLDER_TYPE = FolderNode.node_class.__name__
_DOCUMENT_TYPE = DocumentNode.node_class.__name__
_SECTION_TYPE = SectionNode.node_class.__name__
_CHUNK_TYPE = SectionChunkNode.node_class.__name__
_IMAGE_TYPE = ImageNode.node_class.__name__


class DocumentGraphIngestResult(BaseModel):
    """Outcome of a Document Graph ingestion run."""

    documents_processed: int = 0
    documents_failed: int = 0
    documents_skipped: int = 0
    sections_created: int = 0
    sections_summarized: int = 0
    chunks_created: int = 0
    images_created: int = 0
    relationships_created: int = 0
    embeddings_model: str | None = None
    embeddings_dim: int | None = None
    fts_index: str | None = None
    warnings: list[str] = Field(default_factory=list)


def _document_exists(backend: KgBackend, markdown_hash: str) -> bool:
    """Return True if a Document with this markdown_hash is already in the graph."""
    try:
        df = backend.execute_get_as_df(
            f"MATCH (d:{_DOCUMENT_TYPE} {{markdown_hash: $h}}) RETURN d.markdown_hash AS h LIMIT 1",
            {"h": markdown_hash},
            union=False,
        )
    except Exception as exc:  # noqa: BLE001
        if "does not exist" in str(exc):
            return False
        raise
    return not df.empty


def _sections_described(backend: KgBackend, markdown_hash: str) -> bool:
    """Return True if any section of this document already carries a description.

    Used by the LLM build path to decide whether an already-ingested document's
    sections were built algorithmically (no descriptions) and so should be rebuilt
    to pick up the LLM outline's descriptions/summaries.
    """
    try:
        df = backend.execute_get_as_df(
            f"MATCH (s:{_SECTION_TYPE} {{markdown_hash: $h}}) WHERE s.description IS NOT NULL RETURN count(s) AS c",
            {"h": markdown_hash},
            union=False,
        )
    except Exception as exc:  # noqa: BLE001
        if "does not exist" in str(exc):
            return False
        raise
    if df.empty:
        return False
    return int(df.iloc[0]["c"]) > 0


def ingest_document_graph(
    backend: KgBackend,
    factory: DocumentGraphFactory,
    *,
    force: bool = False,
    retrieval_config: RetrievalConfig | None = None,
    embed_workers: int = 1,
) -> DocumentGraphIngestResult:
    """Ingest a Markdown corpus (via *factory*) into *backend* as a hash-keyed tree.

    Idempotent: unchanged files MERGE in place. A Document already present in the
    graph (matched by ``markdown_hash``) has its sections reused — they are not
    re-parsed into new nodes — unless ``force=True``, which deletes and rebuilds
    them.

    When *retrieval_config* is supplied, each built section is also chunked and
    its ``SectionChunk`` rows carry HNSW-indexed ``chunk_embedding`` vectors, and
    a native FTS/BM25 index is created over the section text (see
    :mod:`genai_graph.kg.document_graph.retrieval`).

    Args:
        backend: Connected `KgBackend` (already `.connect()`ed).
        factory: `DocumentGraphFactory` describing the corpus to ingest.
        force: Rebuild sections for documents already in the graph.
        retrieval_config: Optional embeddings + FTS settings for hybrid retrieval.
        embed_workers: Parallel workers used to chunk + embed documents before
            the (single-writer) merge. Values > 1 overlap per-document embedding
            batches in a thread pool; the DB is only touched afterwards. 1 keeps
            embedding serial.

    Returns:
        `DocumentGraphIngestResult` with counts and any warnings.
    """
    result = DocumentGraphIngestResult()

    schema = factory.build_schema()
    create_schema(backend, schema.nodes, schema.relations)
    registry = NodeTypeRegistry.from_graph_nodes(schema.nodes)

    # Retrieval setup: resolve the embeddings dimension, ensure the
    # SectionChunk.chunk_embedding FLOAT[N] column exists, and initialise the
    # (cached) embeddings handler. A failure here (missing API key, unknown
    # model) disables chunk indexing for this run but leaves FTS intact.
    embeddings_handler: EmbeddingsHandler | None = None
    embeddings_dim: int | None = None
    if retrieval_config is not None and retrieval_config.embeddings_id:
        try:
            embeddings_dim = resolve_embedding_dimension(retrieval_config.embeddings_id)
            ensure_chunk_embedding_column(backend, embeddings_dim)
            embeddings_handler = EmbeddingsHandler(embeddings_id=retrieval_config.embeddings_id)
        except Exception as exc:  # noqa: BLE001
            msg = f"Embeddings disabled for this build: {exc}"
            logger.warning(msg)
            result.warnings.append(msg)
            embeddings_handler = None
            embeddings_dim = None
    chunks_enabled = embeddings_handler is not None

    nodes = NodeDataCollection()
    relationships: list[RelationshipRecord] = []
    seen_folders: set[str] = set()
    seen_folder_edges: set[tuple[str, str]] = set()
    seen_markdown: set[str] = set()

    keys = factory.get_keys()
    total_keys = len(keys)

    # --- Phase 1: parse bundles (cheap once the outline cache is warm) -----
    parsed: list[DocumentGraphBundle] = []
    for key in keys:
        try:
            bundle = factory.get_struct_data_by_key(key)
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to parse {key}: {exc}"
            logger.error(msg)
            result.warnings.append(msg)
            result.documents_failed += 1
            continue

        if bundle is None:
            result.documents_failed += 1
            continue
        parsed.append(bundle)

    # --- Phase 2: structural nodes + skip/rebuild decisions (DB reads) -----
    pending: list[tuple[DocumentGraphBundle, Document, bool]] = []
    for bundle in parsed:
        document = bundle.document
        md_hash = document.markdown_hash or ""

        # Structural nodes (always MERGE; cheap and idempotent)
        for folder in bundle.folders:
            if folder.folder_id not in seen_folders:
                seen_folders.add(folder.folder_id)
                folder_dict = folder.model_dump()
                folder_dict["name"] = folder.name
                nodes.add(_FOLDER_TYPE, folder_dict)

        for parent_folder, child_folder in zip(bundle.folders, bundle.folders[1:], strict=False):
            edge = (parent_folder.folder_id, child_folder.folder_id)
            if edge not in seen_folder_edges:
                seen_folder_edges.add(edge)
                relationships.append(
                    RelationshipRecord(_FOLDER_TYPE, edge[0], _FOLDER_TYPE, edge[1], HAS_SUBFOLDER.name, {})
                )

        doc_dict = document.model_dump()
        doc_dict["name"] = document.filename
        nodes.add(_DOCUMENT_TYPE, doc_dict)

        leaf_folder = bundle.folders[-1]
        relationships.append(
            RelationshipRecord(
                _FOLDER_TYPE, leaf_folder.folder_id, _DOCUMENT_TYPE, document.content_hash, CONTAINS_DOC.name, {}
            )
        )

        # Section reuse: skip if already ingested. In-batch dedup first: the
        # same markdown already queued for merge this run.
        if md_hash in seen_markdown:
            result.documents_skipped += 1
            result.documents_processed += 1
            continue

        in_db = _document_exists(backend, md_hash)
        if force:
            rebuild = True
        elif factory.outline_config is not None:
            # LLM build path: rebuild sections that were built without descriptions
            # (algorithmically, or by a prior degraded run) so a re-run after `--llm`
            # enriches them with the outline's descriptions/summaries.
            rebuild = in_db and not _sections_described(backend, md_hash)
        else:
            rebuild = False

        if in_db and not rebuild:
            result.documents_skipped += 1
            result.documents_processed += 1
            continue
        seen_markdown.add(md_hash)
        pending.append((bundle, document, rebuild))

    # --- Phase 3: chunk + embed documents in parallel (no DB access) -------
    # NOTE: parallelism runs on a single asyncio loop, not threads. The cached
    # embeddings byte-store binds asyncio primitives to the loop that first uses
    # it, so calling the sync cache path from several threads fails with
    # "<Lock object> is bound to a different event loop". CacheBackedEmbeddings'
    # async path (aembed_documents) plus a concurrency semaphore gives the same
    # overlap safely.
    chunk_lists: list[list[tuple[str, dict[str, Any]]]] = []
    if chunks_enabled and retrieval_config is not None and pending:
        assert embeddings_handler is not None  # guaranteed by chunks_enabled

        if embed_workers <= 1:
            chunk_lists = [
                build_sections_chunks(
                    b.sections,
                    handler=embeddings_handler,  # type: ignore[arg-type]
                    chunk_size_tokens=retrieval_config.chunk_size_tokens,
                )
                for b, _, _ in pending
            ]
        else:
            embedder = embeddings_handler.factory

            async def _embed_all() -> list[list[tuple[str, dict[str, Any]]]]:
                semaphore = asyncio.Semaphore(embed_workers)

                async def _one(bundle: DocumentGraphBundle) -> list[tuple[str, dict[str, Any]]]:
                    items = prepare_chunk_inputs(bundle.sections, chunk_size_tokens=retrieval_config.chunk_size_tokens)
                    if not items:
                        return []
                    async with semaphore:
                        embeddings = await embedder.aembed_documents([item[6] for item in items])
                    return attach_chunk_embeddings(items, embeddings)

                return list(await asyncio.gather(*(_one(b) for b, _, _ in pending)))

            try:
                chunk_lists = asyncio.run(_embed_all())
            except Exception as exc:  # noqa: BLE001
                raise RetrievalError(f"Parallel embedding failed for {len(pending)} documents: {exc}") from exc

    # --- Phase 4: accumulate section/chunk rows serially, then merge -------
    for doc_idx, (bundle, document, rebuild) in enumerate(pending, 1):
        md_hash = document.markdown_hash or ""

        if rebuild:
            _delete_document_sections(backend, md_hash)

        for section in bundle.sections:
            section_dict = section.model_dump()
            section_dict["name"] = section.title
            nodes.add(_SECTION_TYPE, section_dict)

            if section.parent_section_id is None:
                relationships.append(
                    RelationshipRecord(
                        _DOCUMENT_TYPE, document.content_hash, _SECTION_TYPE, section.section_id, HAS_SECTION.name, {}
                    )
                )
            else:
                relationships.append(
                    RelationshipRecord(
                        _SECTION_TYPE,
                        section.parent_section_id,
                        _SECTION_TYPE,
                        section.section_id,
                        HAS_SUBSECTION.name,
                        {},
                    )
                )

        result.sections_created += len(bundle.sections)
        result.sections_summarized += sum(1 for s in bundle.sections if s.summary)

        # Ingest extracted images and link to their owning MarkdownSection
        for image in bundle.images:
            image_dict = image.model_dump()
            image_dict["name"] = image.name
            nodes.add(_IMAGE_TYPE, image_dict)
            relationships.append(
                RelationshipRecord(
                    _SECTION_TYPE,
                    image.section_id,
                    _IMAGE_TYPE,
                    image.image_id,
                    HAS_IMAGE.name,
                    {},
                )
            )
        result.images_created += len(bundle.images)

        doc_chunks_count = 0
        if chunks_enabled and retrieval_config is not None:
            section_chunks = chunk_lists[doc_idx - 1]
            for section_id, cd in section_chunks:
                nodes.add(_CHUNK_TYPE, cd)
                relationships.append(
                    RelationshipRecord(
                        _SECTION_TYPE,
                        section_id,
                        _CHUNK_TYPE,
                        cd["chunk_id"],
                        HAS_CHUNK.name,
                        {},
                    )
                )
            doc_chunks_count = len(section_chunks)
            result.chunks_created += doc_chunks_count

        result.documents_processed += 1
        logger.info(
            "Ingested [{}/{}]: {} (sections={}, images={}, chunks={})",
            doc_idx,
            total_keys,
            document.filename,
            len(bundle.sections),
            len(bundle.images),
            doc_chunks_count,
        )

    merge_result = merge_nodes_batch(backend, nodes, registry)
    result.relationships_created = merge_relationships_batch(backend, relationships, registry, merge_result.id_mapping)

    # Post-merge indexing: HNSW over chunk embeddings and a native FTS/BM25
    # index over the section text. Both are best-effort; failures are recorded
    # as warnings rather than aborting an otherwise-successful ingest.
    if chunks_enabled and isinstance(backend, KuzuBackend):
        try:
            backend.create_vector_index(_CHUNK_TYPE, "chunk_embedding", "chunk_embedding_index", metric="cosine")
        except Exception as exc:  # noqa: BLE001
            msg = f"Could not create HNSW index on {_CHUNK_TYPE}.chunk_embedding: {exc}"
            logger.warning(msg)
            result.warnings.append(msg)
    if retrieval_config is not None and retrieval_config.fts:
        try:
            result.fts_index = ensure_section_fts_index(backend)
        except Exception as exc:  # noqa: BLE001
            msg = f"Could not create FTS index: {exc}"
            logger.warning(msg)
            result.warnings.append(msg)
    result.embeddings_model = retrieval_config.embeddings_id if retrieval_config else None
    result.embeddings_dim = embeddings_dim

    logger.info(
        "Document Graph ingest: {} processed ({} skipped), {} failed, {} section(s), {} image(s), {} chunk(s), {} rel(s)",
        result.documents_processed,
        result.documents_skipped,
        result.documents_failed,
        result.sections_created,
        result.images_created,
        result.chunks_created,
        result.relationships_created,
    )
    return result


def _delete_document_sections(backend: KgBackend, markdown_hash: str) -> None:
    """Delete existing sections (and their chunks/images) for a document (used on force)."""
    try:
        backend.execute(f"MATCH (i:{_IMAGE_TYPE} {{markdown_hash: $h}}) DETACH DELETE i", {"h": markdown_hash})
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not clear stale {} for {}: {}", _IMAGE_TYPE, markdown_hash, exc)
    try:
        backend.execute(f"MATCH (c:{_CHUNK_TYPE} {{markdown_hash: $h}}) DETACH DELETE c", {"h": markdown_hash})
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not clear stale {} for {}: {}", _CHUNK_TYPE, markdown_hash, exc)
    try:
        backend.execute(f"MATCH (n:{_SECTION_TYPE} {{markdown_hash: $h}}) DETACH DELETE n", {"h": markdown_hash})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not clear stale {} for {}: {}", _SECTION_TYPE, markdown_hash, exc)


def drop_document_graph(backend: KgBackend, *, drop_documents: bool = False) -> None:
    """Drop the Document Graph structure tables (sections + their relationships).

    By default only drops the Section table and its relationships, leaving
    Folder/Document metadata intact (since they may be shared with other
    factories). This means `list_documents()` will still return entries. Pass
    `drop_documents=True` for a full reset.

    Args:
        backend: Connected `KgBackend`.
        drop_documents: Also drop Folder/Document tables. Leave `False` when those
            are shared with other factories; set `True` for a complete reset.
    """
    for rel in (HAS_IMAGE.name, HAS_CHUNK.name, HAS_SUBSECTION.name, HAS_SECTION.name):
        backend.drop_table(rel)
    backend.drop_table(_IMAGE_TYPE)
    backend.drop_table(_SECTION_TYPE)
    backend.drop_table(_CHUNK_TYPE)
    if drop_documents:
        backend.drop_table(CONTAINS_DOC.name)
        backend.drop_table(_DOCUMENT_TYPE)
        backend.drop_table(HAS_SUBFOLDER.name)
        backend.drop_table(_FOLDER_TYPE)
