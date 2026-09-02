"""Retrieval augmentation for the Document Graph build.

When a :class:`RetrievalConfig` is supplied to
:func:`genai_graph.kg.document_graph.ingest.ingest_document_graph`, each ingested
``MarkdownSection`` is split into ``SectionChunk`` rows whose ``chunk_embedding``
(computed from a contextualized ``"{title} | {description}\\n\\n{chunk_text}"``
input via :class:`~genai_graph.kg.embeddings_handler.EmbeddingsHandler`) is
HNSW-indexed for cosine search, and a native FTS/BM25 index is built over the
section's own ``title`` / ``text`` / ``description``. The hybrid
``search_sections`` tool fuses the two.

The ``chunk_embedding`` column is a fixed-length ``FLOAT[N]`` (managed here, not
a Pydantic field) so Ladybug can build an HNSW vector index on it; the generic
``index_fields`` mechanism is deliberately not used because the Document Graph
ingest bypasses ``extract_graph_data`` and the embedding input is a constructed
contextualized string rather than a single field value.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.backend import KgBackend, KuzuBackend
from genai_graph.kg.document_graph.chunker import chunk_section_text
from genai_graph.kg.embeddings_handler import EmbeddingsHandler

_CHUNK_TABLE = "SectionChunk"
_EMBEDDING_FIELD = "chunk_embedding"
_DEFAULT_FTS_INDEX = "section_fts"
_DEFAULT_VECTOR_INDEX = "chunk_embedding_index"
_FTS_FIELDS = ("title", "text", "description")


class RetrievalConfig(BaseModel):
    """Settings for the retrieval-augmented Document Graph build."""

    embeddings_id: str | None = Field(
        default=None,
        description="Embeddings model id (name@provider); None disables semantic chunk indexing",
    )
    chunk_size_tokens: int = Field(
        default=1500,
        description="Target chunk size; sections longer than this are split into ~this many tokens",
    )
    fts: bool = Field(
        default=True,
        description="Create a native FTS/BM25 index over MarkdownSection(title, text, description)",
    )


class RetrievalError(RuntimeError):
    """Raised when retrieval build configuration is invalid or conflicts with the DB."""


def resolve_embedding_dimension(embeddings_id: str) -> int:
    """Return the configured dimension for *embeddings_id* (no API key needed)."""
    from genai_tk.core.factories.embeddings_factory import EmbeddingsFactory

    for info in EmbeddingsFactory.known_list():
        if info.id == embeddings_id:
            if info.dimension is None:
                raise RetrievalError(f"Embeddings model '{embeddings_id}' has no configured dimension")
            return info.dimension
    raise RetrievalError(
        f"Unknown embeddings model '{embeddings_id}'; not in EmbeddingsFactory.known_list(). "
        "Check config/providers/embeddings.yaml."
    )


def _column_type(backend: KgBackend, table: str, col: str) -> str | None:
    """Return the Kuzu type string of *col* on *table*, or None when absent."""
    try:
        for row in backend.execute(f"CALL table_info('{table}') RETURN *"):
            if str(row[1]) == col:
                return str(row[2])
    except Exception as exc:  # noqa: BLE001
        logger.debug("table_info('{}') failed: {}", table, exc)
    return None


def _table_columns(backend: KgBackend, table: str) -> set[str]:
    """Return the set of property names on *table* (empty when the table is absent)."""
    try:
        df = backend.execute_get_as_df(f"CALL table_info('{table}') RETURN *", None, union=False)
    except Exception:  # noqa: BLE001
        return set()
    if df is None or df.empty:
        return set()
    name_col = df["name"] if "name" in df.columns else df.iloc[:, 1]
    return {str(v) for v in name_col}


def ensure_chunk_embedding_column(backend: KgBackend, dim: int) -> None:
    """Idempotently ensure ``SectionChunk.chunk_embedding`` exists as ``FLOAT[dim]``."""
    want = f"FLOAT[{dim}]"
    existing = _column_type(backend, _CHUNK_TABLE, _EMBEDDING_FIELD)
    if existing is None:
        try:
            backend.execute(f"ALTER TABLE {_CHUNK_TABLE} ADD {_EMBEDDING_FIELD} {want}")
            logger.info("Added {}.{} {}", _CHUNK_TABLE, _EMBEDDING_FIELD, want)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Could not add {_EMBEDDING_FIELD} {want} to {_CHUNK_TABLE}: {exc}") from exc
        return
    if existing.replace(" ", "") == want.replace(" ", ""):
        return
    raise RetrievalError(
        f"{_CHUNK_TABLE}.{_EMBEDDING_FIELD} is {existing} but the configured model needs {want}. "
        "Rebuild the graph with force=True after changing the embeddings model."
    )


def ensure_section_fts_index(backend: KgBackend, index_name: str = _DEFAULT_FTS_INDEX) -> str | None:
    """Create the native FTS index over the available MarkdownSection text fields.

    Returns the index name, or None when the FTS extension is unavailable or the
    section table has none of the expected text columns.
    """
    if not isinstance(backend, KuzuBackend) or not backend.ensure_fts_extension():
        logger.warning("FTS extension unavailable; skipping BM25 index creation")
        return None
    cols = _table_columns(backend, "MarkdownSection")
    fields = [c for c in _FTS_FIELDS if c in cols]
    if not fields:
        logger.warning("MarkdownSection has none of {}; skipping FTS index", ", ".join(_FTS_FIELDS))
        return None
    fields_literal = "[" + ", ".join(f"'{c}'" for c in fields) + "]"
    try:
        backend.execute(f"CALL CREATE_FTS_INDEX('MarkdownSection', '{index_name}', {fields_literal})")
        logger.info("Created FTS index {} over MarkdownSection({})", index_name, ", ".join(fields))
    except Exception as exc:  # noqa: BLE001
        if "already" in str(exc).lower():
            logger.debug("FTS index {} already exists", index_name)
            return index_name
        raise
    return index_name


def build_sections_chunks(
    sections: list[Any], *, handler: EmbeddingsHandler, chunk_size_tokens: int
) -> list[tuple[str, dict[str, Any]]]:
    """Chunk multiple sections in batch, computing contextualized embeddings in a single batch call.

    Args:
        sections: List of MarkdownSection instances to chunk.
        handler: EmbeddingsHandler instance for computing embeddings.
        chunk_size_tokens: Target chunk size in tokens.

    Returns:
        List of ``(section_id, chunk_dict)`` pairs.
    """
    items: list[tuple[str, str, str, int, str, int, str]] = []
    for section in sections:
        pieces = chunk_section_text(section.text, size_tokens=chunk_size_tokens)
        desc = section.description or ""
        header = section.title + (f" | {desc}" if desc else "")
        for idx, (chunk_text, token_count) in enumerate(pieces):
            embed_input = f"{header}\n\n{chunk_text}"
            chunk_id = f"{section.section_id}::c{idx}"
            items.append(
                (section.section_id, section.markdown_hash, chunk_id, idx, chunk_text, token_count, embed_input)
            )

    if not items:
        return []

    embed_inputs = [item[6] for item in items]
    try:
        embeddings = handler.compute_embeddings_batch(embed_inputs)
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"Batch embedding failed for {len(embed_inputs)} chunks: {exc}") from exc

    results: list[tuple[str, dict[str, Any]]] = []
    for (section_id, md_hash, chunk_id, idx, chunk_text, token_count, _), embedding in zip(
        items, embeddings, strict=True
    ):
        results.append(
            (
                section_id,
                {
                    "chunk_id": chunk_id,
                    "section_id": section_id,
                    "markdown_hash": md_hash,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "token_count": token_count,
                    "chunk_embedding": embedding,
                    "name": chunk_id,
                },
            )
        )
    return results


def build_section_chunks(section: Any, *, handler: EmbeddingsHandler, chunk_size_tokens: int) -> list[dict[str, Any]]:
    """Chunk a section, compute contextualized embeddings, return SectionChunk data dicts.

    Each chunk's embedding input is ``"{title} | {description}\\n\\n{chunk_text}"``
    so the vector encodes the section's identity/routing context alongside the
    chunk body. The returned dicts include a ``chunk_embedding`` key (a list of
    floats) that the merge path picks up as a dynamic embedding column.
    """
    return [cd for _, cd in build_sections_chunks([section], handler=handler, chunk_size_tokens=chunk_size_tokens)]
