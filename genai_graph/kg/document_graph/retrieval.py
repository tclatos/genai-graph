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

from typing import Any, Sequence

from genai_tk.extra.nlp import (
    get_dominant_language,
    get_ladybug_stemmer,
    get_stopwords_union,
    stem_stopwords,
)
from loguru import logger
from pydantic import BaseModel, Field

from genai_graph.kg.backend import KgBackend, KuzuBackend
from genai_graph.kg.document_graph.chunker import chunk_section_text
from genai_graph.kg.embeddings_handler import EmbeddingsHandler

_CHUNK_TABLE = "SectionChunk"
_EMBEDDING_FIELD = "chunk_embedding"
_DEFAULT_FTS_INDEX = "section_fts"
_DEFAULT_VECTOR_INDEX = "chunk_embedding_index"
_STOPWORDS_TABLE = "_FtsStopWords"
_FTS_FIELDS = ("title", "text", "description", "summary")


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


def _ensure_stopwords_table(backend: KgBackend, stopwords: set[str], table_name: str = _STOPWORDS_TABLE) -> bool:
    """Create and populate the stop words node table in Ladybug.

    Returns True if the table was created and populated with stop words, False otherwise.
    """
    clean_words = sorted({w.strip().lower() for w in stopwords if w and w.strip()})
    if not clean_words:
        return False
    try:
        backend.execute(f"CREATE NODE TABLE IF NOT EXISTS {table_name}(word STRING, PRIMARY KEY(word))")
        import pyarrow as pa

        tbl = pa.table({"word": pa.array(clean_words, type=pa.string())})  # noqa: F841
        try:
            backend.execute(f"COPY {table_name} FROM tbl")
        except Exception as copy_exc:
            logger.debug("COPY into {} failed ({}); falling back to LOAD FROM ... MERGE", table_name, copy_exc)
            backend.execute(f"LOAD FROM tbl MERGE (s:{table_name} {{word: word}})")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not populate {} table: {}", table_name, exc)
        return False


def ensure_section_fts_index(
    backend: KgBackend,
    index_name: str = _DEFAULT_FTS_INDEX,
    *,
    languages: Sequence[str] | None = None,
    stemmer: str | None = None,
) -> str | None:
    """Create the native FTS index over the available MarkdownSection text fields.

    Configures Ladybug's Snowball stemmer based on the dominant document language
    and populates a dedicated stop-words node table (_FtsStopWords) with the union
    of stop words across all corpus languages.

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

    # Resolve languages from argument or existing Document nodes
    lang_list: list[str] = [str(lang) for lang in languages if lang] if languages else []
    if not lang_list:
        try:
            rows = backend.execute("MATCH (d:Document) RETURN DISTINCT d.language AS lang")
            lang_list = [r[0] for r in rows if r and r[0]]
        except Exception:  # noqa: BLE001
            lang_list = []
    if not lang_list:
        lang_list = ["en"]

    dominant_lang = get_dominant_language(lang_list, default="en")
    effective_stemmer = stemmer or get_ladybug_stemmer(dominant_lang, default="english")
    # Ladybug stems indexed tokens before matching them against the stop-word
    # list, so the stop words must be provided in their stemmed form (FTS docs).
    stopwords_set = stem_stopwords(get_stopwords_union(lang_list), effective_stemmer)

    has_stopwords = _ensure_stopwords_table(backend, stopwords_set, _STOPWORDS_TABLE)

    if has_stopwords:
        stmt = (
            f"CALL CREATE_FTS_INDEX('MarkdownSection', '{index_name}', {fields_literal}, "
            f"stemmer := '{effective_stemmer}', stopwords := '{_STOPWORDS_TABLE}')"
        )
    else:
        stmt = (
            f"CALL CREATE_FTS_INDEX('MarkdownSection', '{index_name}', {fields_literal}, "
            f"stemmer := '{effective_stemmer}')"
        )

    try:
        backend.execute(stmt)
        logger.info(
            "Created FTS index {} over MarkdownSection({}) (stemmer='{}', stopwords={})",
            index_name,
            ", ".join(fields),
            effective_stemmer,
            _STOPWORDS_TABLE if has_stopwords else "default",
        )
    except Exception as exc:  # noqa: BLE001
        if "already" in str(exc).lower():
            logger.debug("FTS index {} already exists", index_name)
            return index_name
        raise
    return index_name


def prepare_chunk_inputs(
    sections: list[Any], *, chunk_size_tokens: int
) -> list[tuple[str, str, str, int, str, int, str]]:
    """Chunk *sections* and build the embedding inputs for each chunk (pure CPU, no I/O).

    Returns:
        List of ``(section_id, markdown_hash, chunk_id, chunk_index, chunk_text, token_count, embed_input)`` tuples.
    """
    items: list[tuple[str, str, str, int, str, int, str]] = []
    for section in sections:
        pieces = chunk_section_text(section.text, size_tokens=chunk_size_tokens)
        desc = section.description or ""
        kw_list = getattr(section, "keywords", []) or []
        kw_str = ", ".join(kw_list) if kw_list else ""
        header = section.title + (f" | {desc}" if desc else "") + (f" | Keywords: {kw_str}" if kw_str else "")
        for idx, (chunk_text, token_count) in enumerate(pieces):
            embed_input = f"{header}\n\n{chunk_text}"
            chunk_id = f"{section.section_id}::c{idx}"
            items.append(
                (section.section_id, section.markdown_hash, chunk_id, idx, chunk_text, token_count, embed_input)
            )
    return items


def attach_chunk_embeddings(
    items: list[tuple[str, str, str, int, str, int, str]], embeddings: list[list[float]]
) -> list[tuple[str, dict[str, Any]]]:
    """Pair prepared chunk inputs with their computed embeddings as chunk dicts."""
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
    items = prepare_chunk_inputs(sections, chunk_size_tokens=chunk_size_tokens)
    if not items:
        return []

    embed_inputs = [item[6] for item in items]
    try:
        embeddings = handler.compute_embeddings_batch(embed_inputs)
    except Exception as exc:  # noqa: BLE001
        raise RetrievalError(f"Batch embedding failed for {len(embed_inputs)} chunks: {exc}") from exc

    return attach_chunk_embeddings(items, embeddings)


def build_section_chunks(section: Any, *, handler: EmbeddingsHandler, chunk_size_tokens: int) -> list[dict[str, Any]]:
    """Chunk a section, compute contextualized embeddings, return SectionChunk data dicts.

    Each chunk's embedding input is ``"{title} | {description}\\n\\n{chunk_text}"``
    so the vector encodes the section's identity/routing context alongside the
    chunk body. The returned dicts include a ``chunk_embedding`` key (a list of
    floats) that the merge path picks up as a dynamic embedding column.
    """
    return [cd for _, cd in build_sections_chunks([section], handler=handler, chunk_size_tokens=chunk_size_tokens)]
