"""Parquet staging and bulk ingestion for Document Graph pipelines.

Stages extracted document bundles (Folder, Document, MarkdownSection, SectionChunk,
Image) into columnar Parquet files with an accompanying manifest.json, enabling:
- Fast, reproducible batch ingestion via Parquet files
- Reusable intermediate caches across multi-step ETL / KG pipelines
- Self-contained base64 image persistence and vector embeddings in standard columnar format
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.backend import KgBackend, KuzuBackend
from genai_graph.kg.document_graph.retrieval import (
    RetrievalConfig,
    ensure_chunk_embedding_column,
    ensure_section_fts_index,
    resolve_embedding_dimension,
)
from genai_graph.kg.export.artifacts import ParquetManifest
from genai_graph.kg.factories.document_graph_factory import DocumentGraphBundle
from genai_graph.kg.ingest.extract import create_schema
from genai_graph.kg.ingest.merge import (
    NodeTypeRegistry,
    _prepare_node_arrow_table,
)
from genai_graph.kg.nodes.document import (
    CONTAINS_DOC,
    HAS_SUBFOLDER,
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

_ALL_NODES = [FolderNode, DocumentNode, SectionNode, SectionChunkNode, ImageNode]
_ALL_RELS = [CONTAINS_DOC, HAS_SUBFOLDER, HAS_SECTION, HAS_SUBSECTION, HAS_CHUNK, HAS_IMAGE]


class StagingStats(BaseModel):
    """Statistics for staged parquet files."""

    staging_dir: str
    node_counts: dict[str, int] = Field(default_factory=dict)
    rel_counts: dict[str, int] = Field(default_factory=dict)
    total_files_staged: int = 0
    manifest_path: str = ""


def stage_document_graph_to_parquet(
    bundles: list[DocumentGraphBundle],
    staging_dir: str | Path,
    *,
    chunk_lists: list[list[tuple[str, dict[str, Any]]]] | None = None,
    config_name: str = "document_graph",
    source_files: list[str] | None = None,
) -> StagingStats:
    """Stage DocumentGraphBundles into columnar Parquet files and manifest.json.

    Args:
        bundles: Parsed DocumentGraphBundle instances.
        staging_dir: Directory to output nodes/, rels/, and manifest.json.
        chunk_lists: Optional per-document chunk lists (with embeddings) aligned with bundles.
        config_name: Name identifier for the manifest.
        source_files: List of source file paths for fingerprinting.

    Returns:
        StagingStats with counts and file paths.
    """
    s_dir = Path(staging_dir)
    nodes_dir = s_dir / "nodes"
    rels_dir = s_dir / "rels"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    rels_dir.mkdir(parents=True, exist_ok=True)

    folder_rows: list[dict[str, Any]] = []
    doc_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    contains_doc_rels: list[dict[str, Any]] = []
    has_subfolder_rels: list[dict[str, Any]] = []
    has_section_rels: list[dict[str, Any]] = []
    has_subsection_rels: list[dict[str, Any]] = []
    has_chunk_rels: list[dict[str, Any]] = []
    has_image_rels: list[dict[str, Any]] = []

    seen_folders: set[str] = set()
    seen_folder_edges: set[tuple[str, str]] = set()
    seen_docs: set[str] = set()
    seen_sections: set[str] = set()
    seen_chunks: set[str] = set()
    seen_images: set[str] = set()

    for idx, bundle in enumerate(bundles):
        # 1. Folders
        for folder in bundle.folders:
            if folder.folder_id not in seen_folders:
                seen_folders.add(folder.folder_id)
                f_dict = folder.model_dump()
                f_dict["name"] = folder.name
                folder_rows.append(f_dict)

        for parent_f, child_f in zip(bundle.folders, bundle.folders[1:], strict=False):
            edge = (parent_f.folder_id, child_f.folder_id)
            if edge not in seen_folder_edges:
                seen_folder_edges.add(edge)
                has_subfolder_rels.append({"_from_id": edge[0], "_to_id": edge[1]})

        # 2. Document
        doc = bundle.document
        if doc.content_hash not in seen_docs:
            seen_docs.add(doc.content_hash)
            d_dict = doc.model_dump()
            d_dict["name"] = doc.filename
            doc_rows.append(d_dict)

            leaf_folder = bundle.folders[-1]
            contains_doc_rels.append({"_from_id": leaf_folder.folder_id, "_to_id": doc.content_hash})

        # 3. Sections
        for sec in bundle.sections:
            if sec.section_id not in seen_sections:
                seen_sections.add(sec.section_id)
                s_dict = sec.model_dump()
                s_dict["name"] = sec.title
                section_rows.append(s_dict)

                if sec.parent_section_id is None:
                    has_section_rels.append({"_from_id": doc.content_hash, "_to_id": sec.section_id})
                else:
                    has_subsection_rels.append({"_from_id": sec.parent_section_id, "_to_id": sec.section_id})

        # 4. Images
        for img in bundle.images:
            if img.image_id not in seen_images:
                seen_images.add(img.image_id)
                i_dict = img.model_dump()
                i_dict["name"] = img.filename
                image_rows.append(i_dict)
                has_image_rels.append({"_from_id": img.section_id, "_to_id": img.image_id})

        # 5. Chunks (if present)
        if chunk_lists and idx < len(chunk_lists):
            doc_chunks = chunk_lists[idx]
            for sec_id, cd in doc_chunks:
                c_id = cd["chunk_id"]
                if c_id not in seen_chunks:
                    seen_chunks.add(c_id)
                    chunk_rows.append(dict(cd))
                    has_chunk_rels.append({"_from_id": sec_id, "_to_id": c_id})

    # Convert to schema-typed Arrow Tables and write Parquet
    registry = NodeTypeRegistry.from_graph_nodes(_ALL_NODES)
    node_counts: dict[str, int] = {}
    rel_counts: dict[str, int] = {}

    def _write_node_table(node_type: str, rows: list[dict[str, Any]], out_path: Path) -> int:
        if not rows:
            return 0
        cfg = registry.get(node_type)
        table = _prepare_node_arrow_table(rows, cfg)
        pq.write_table(table, str(out_path))
        return table.num_rows

    def _write_rel_table(df: pd.DataFrame, out_path: Path) -> int:
        if df.empty:
            return 0
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(out_path))
        return len(df)

    if folder_rows:
        node_counts[_FOLDER_TYPE] = _write_node_table(_FOLDER_TYPE, folder_rows, nodes_dir / f"{_FOLDER_TYPE}.parquet")
    if doc_rows:
        node_counts[_DOCUMENT_TYPE] = _write_node_table(_DOCUMENT_TYPE, doc_rows, nodes_dir / f"{_DOCUMENT_TYPE}.parquet")
    if section_rows:
        node_counts[_SECTION_TYPE] = _write_node_table(_SECTION_TYPE, section_rows, nodes_dir / f"{_SECTION_TYPE}.parquet")
    if image_rows:
        node_counts[_IMAGE_TYPE] = _write_node_table(_IMAGE_TYPE, image_rows, nodes_dir / f"{_IMAGE_TYPE}.parquet")
    if chunk_rows:
        node_counts[_CHUNK_TYPE] = _write_node_table(_CHUNK_TYPE, chunk_rows, nodes_dir / f"{_CHUNK_TYPE}.parquet")

    if contains_doc_rels:
        rel_counts[CONTAINS_DOC.name] = _write_rel_table(
            pd.DataFrame(contains_doc_rels), rels_dir / f"{CONTAINS_DOC.name}.parquet"
        )
    if has_subfolder_rels:
        rel_counts[HAS_SUBFOLDER.name] = _write_rel_table(
            pd.DataFrame(has_subfolder_rels), rels_dir / f"{HAS_SUBFOLDER.name}.parquet"
        )
    if has_section_rels:
        rel_counts[HAS_SECTION.name] = _write_rel_table(
            pd.DataFrame(has_section_rels), rels_dir / f"{HAS_SECTION.name}.parquet"
        )
    if has_subsection_rels:
        rel_counts[HAS_SUBSECTION.name] = _write_rel_table(
            pd.DataFrame(has_subsection_rels), rels_dir / f"{HAS_SUBSECTION.name}.parquet"
        )
    if has_chunk_rels:
        rel_counts[HAS_CHUNK.name] = _write_rel_table(
            pd.DataFrame(has_chunk_rels), rels_dir / f"{HAS_CHUNK.name}.parquet"
        )
    if has_image_rels:
        rel_counts[HAS_IMAGE.name] = _write_rel_table(
            pd.DataFrame(has_image_rels), rels_dir / f"{HAS_IMAGE.name}.parquet"
        )

    # Manifest
    manifest = ParquetManifest(
        config_name=config_name,
        exported_at=datetime.now(timezone.utc).isoformat(),
        node_tables=list(node_counts.keys()),
        rel_tables=list(rel_counts.keys()),
        node_count=sum(node_counts.values()),
        rel_count=sum(rel_counts.values()),
        source_files=source_files or [],
    )
    manifest_path = s_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    logger.info(
        "Staged Document Graph Parquet: {} nodes ({}), {} rels ({}) in {}",
        manifest.node_count,
        node_counts,
        manifest.rel_count,
        rel_counts,
        s_dir,
    )

    return StagingStats(
        staging_dir=str(s_dir),
        node_counts=node_counts,
        rel_counts=rel_counts,
        total_files_staged=len(node_counts) + len(rel_counts),
        manifest_path=str(manifest_path),
    )


def ingest_document_graph_from_staging(
    backend: KgBackend,
    staging_dir: str | Path,
    *,
    retrieval_config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    """Import staged Parquet tables into Ladybug backend and build HNSW / BM25 indexes.

    Args:
        backend: Connected KgBackend instance.
        staging_dir: Directory containing nodes/, rels/, manifest.json.
        retrieval_config: Optional embeddings and FTS settings for post-load indexing.

    Returns:
        Summary dict of ingested node and rel counts.
    """
    s_dir = Path(staging_dir)
    manifest_path = s_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found at staging dir: {s_dir}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ParquetManifest.model_validate(manifest_data)

    # Ensure schema is created
    create_schema(backend, _ALL_NODES, _ALL_RELS)

    nodes_dir = s_dir / "nodes"
    rels_dir = s_dir / "rels"

    # Primary key map
    pk_map = {
        _FOLDER_TYPE: "folder_id",
        _DOCUMENT_TYPE: "content_hash",
        _SECTION_TYPE: "section_id",
        _CHUNK_TYPE: "chunk_id",
        _IMAGE_TYPE: "image_id",
    }

    # Rel from/to label map
    rel_type_map = {
        CONTAINS_DOC.name: (_FOLDER_TYPE, _DOCUMENT_TYPE, "folder_id", "content_hash"),
        HAS_SUBFOLDER.name: (_FOLDER_TYPE, _FOLDER_TYPE, "folder_id", "folder_id"),
        HAS_SECTION.name: (_DOCUMENT_TYPE, _SECTION_TYPE, "content_hash", "section_id"),
        HAS_SUBSECTION.name: (_SECTION_TYPE, _SECTION_TYPE, "section_id", "section_id"),
        HAS_CHUNK.name: (_SECTION_TYPE, _CHUNK_TYPE, "section_id", "chunk_id"),
        HAS_IMAGE.name: (_SECTION_TYPE, _IMAGE_TYPE, "section_id", "image_id"),
    }

    total_nodes = 0
    total_rels = 0

    # Ensure chunk_embedding column exists if SectionChunk is staged
    if _CHUNK_TYPE in manifest.node_tables and retrieval_config and retrieval_config.embeddings_id:
        dim = resolve_embedding_dimension(retrieval_config.embeddings_id)
        ensure_chunk_embedding_column(backend, dim)

    conn = backend.conn if hasattr(backend, "conn") else backend

    # Ingest node tables via LOAD FROM raw_table MERGE
    for node_type in manifest.node_tables:
        p_path = nodes_dir / f"{node_type}.parquet"
        if not p_path.exists():
            continue

        raw_table = pq.read_table(str(p_path))
        if raw_table.num_rows == 0:
            continue

        pk_col = pk_map.get(node_type, "id")
        other_cols = [c for c in raw_table.column_names if c != pk_col]
        set_clause = ", ".join([f"n.{c} = {c}" for c in other_cols])
        set_stmt = f" ON CREATE SET {set_clause} ON MATCH SET {set_clause}" if other_cols else ""

        cypher = f"""
            LOAD FROM raw_table
            MERGE (n:{node_type} {{{pk_col}: {pk_col}}})
            {set_stmt}
        """
        conn.execute(cypher)
        total_nodes += raw_table.num_rows
        logger.debug("Ingested {} {} nodes from Parquet", raw_table.num_rows, node_type)

    # Ingest relationship tables
    for rel_name in manifest.rel_tables:
        p_path = rels_dir / f"{rel_name}.parquet"
        if not p_path.exists():
            continue

        raw_rel_table = pq.read_table(str(p_path))
        if raw_rel_table.num_rows == 0:
            continue

        if rel_name not in rel_type_map:
            logger.warning("Unknown relation in staging: {}", rel_name)
            continue

        from_type, to_type, from_pk, to_pk = rel_type_map[rel_name]

        # Use staged WITH to avoid Cartesian cross products in Kuzu
        cypher = f"""
            LOAD FROM raw_rel_table
            MATCH (from:{from_type}) WHERE from.{from_pk} = _from_id
            WITH from, _to_id
            MATCH (to:{to_type}) WHERE to.{to_pk} = _to_id
            MERGE (from)-[:{rel_name}]->(to)
        """
        conn.execute(cypher)
        total_rels += raw_rel_table.num_rows
        logger.debug("Ingested {} {} relationships from Parquet", raw_rel_table.num_rows, rel_name)

    # Post-load indexing
    if _CHUNK_TYPE in manifest.node_tables and retrieval_config and retrieval_config.embeddings_id:
        if isinstance(backend, KuzuBackend):
            try:
                backend.create_vector_index(_CHUNK_TYPE, "chunk_embedding", "chunk_embedding_index", metric="cosine")
                logger.info("Created HNSW vector index on {}.chunk_embedding", _CHUNK_TYPE)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create HNSW index: {}", exc)

    fts_name = None
    if retrieval_config and retrieval_config.fts:
        try:
            fts_name = ensure_section_fts_index(backend)
            logger.info("Created FTS index {} on MarkdownSection", fts_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create FTS index: {}", exc)

    return {
        "status": "success",
        "nodes_ingested": total_nodes,
        "rels_ingested": total_rels,
        "node_tables": manifest.node_tables,
        "rel_tables": manifest.rel_tables,
        "fts_index": fts_name,
    }
