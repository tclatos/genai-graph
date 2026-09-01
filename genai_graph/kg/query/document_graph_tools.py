"""Navigation tools for the Document Graph.

Exposes read-only Cypher-backed functions an agent (or the `cli docgraph`
command) can call to walk a document's heading hierarchy, fetch individual
sections, and reconstruct a whole document from its sections. Documents are
addressed by the Document content hash (full or prefix), its ``markdown_hash``,
the filename, or the source path.

The tools are **schema-tolerant**: they introspect the actual Ladybug table
columns once (via ``CALL table_info``) and ``CALL show_tables`` for
relationships, then build RETURN clauses and traversals from what is present.
An older database that lacks ``Folder.parent_folder_id`` / ``HAS_SUBFOLDER``
or the ``description`` / ``summary`` columns is handled gracefully (those
fields are omitted, folder navigation falls back to flat) instead of raising a
raw Cypher binder error. Truly-missing structures surface as
:class:`DocumentGraphError` with a plain-English message.
"""

from __future__ import annotations

from typing import Any

import yaml
from genai_tk.utils.ladybug import get_shared_database
from langchain_core.tools import BaseTool, tool
from loguru import logger

from genai_graph.kg.backend import KgBackend, KuzuBackend, LadybugBackend
from genai_graph.kg.embeddings_handler import EmbeddingsHandler
from genai_graph.kg.nodes.document import DocumentNode, FolderNode
from genai_graph.kg.nodes.document_section import SectionChunkNode, SectionNode

_DOCUMENT_LABEL = DocumentNode.node_class.__name__
_SECTION_LABEL = SectionNode.node_class.__name__
_FOLDER_LABEL = FolderNode.node_class.__name__

# Columns a caller can reasonably expect on each row type. Rows are normalized
# so every key is present (``None`` when the column does not exist in the DB),
# which keeps callers and the TUI/CLI from KeyErroing on an older schema.
_FOLDER_KEYS: tuple[str, ...] = ("folder_id", "parent_folder_id", "name", "kind", "uri", "doc_count")
_DOC_KEYS: tuple[str, ...] = (
    "content_hash",
    "markdown_hash",
    "filename",
    "section_count",
    "token_count",
    "description",
    "summary",
    "path",
    "folder_id",
)
_SECTION_TOC_KEYS: tuple[str, ...] = (
    "section_id",
    "parent_section_id",
    "title",
    "level",
    "line_start",
    "sequence",
    "token_count",
    "description",
    "summary",
    "summary_source",
)

# Per-backend introspection caches (keyed by id(backend) so distinct connections
# don't share stale schemas).
_TABLE_COL_CACHE: dict[tuple[int, str], set[str]] = {}
_REL_CACHE: dict[int, set[str]] = {}


class DocumentGraphError(Exception):
    """Raised when the Document Graph cannot answer because required data is missing.

    Carries a plain-English message suitable for surfacing to an agent or a CLI
    user (e.g. "no Document table — ingest first with `cli docgraph build`").
    """


def _table_columns(backend: KgBackend, table: str) -> set[str]:
    """Return the set of property names on a node table, introspected via ``table_info``.

    Returns an empty set when the table does not exist (e.g. a fresh DB that has
    never been ingested) so callers can fall back instead of crashing.
    """
    key = (id(backend), table)
    cached = _TABLE_COL_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        df = backend.execute_get_as_df(f"CALL table_info('{table}') RETURN *", None, union=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("table_info('{}') failed: {}", table, exc)
        _TABLE_COL_CACHE[key] = set()
        return set()
    if df is None or df.empty:
        _TABLE_COL_CACHE[key] = set()
        return set()
    name_col = df["name"] if "name" in df.columns else df.iloc[:, 1]
    cols = {str(v) for v in name_col}
    _TABLE_COL_CACHE[key] = cols
    return cols


def _has_relationship(backend: KgBackend, rel_name: str) -> bool:
    """Return True when a relationship table named *rel_name* exists in the DB."""
    key = id(backend)
    cached = _REL_CACHE.get(key)
    if cached is None:
        try:
            df = backend.execute_get_as_df("CALL show_tables() RETURN *", None, union=False)
            cached = {str(v) for v in (df.values.flatten() if df is not None else [])}
        except Exception as exc:  # noqa: BLE001
            logger.debug("show_tables() failed: {}", exc)
            cached = set()
        _REL_CACHE[key] = cached
    return rel_name in cached


def _has_table(backend: KgBackend, table: str) -> bool:
    """Return True when a node/rel table named *table* exists."""
    return bool(_table_columns(backend, table)) or _has_relationship(backend, table)


def _pick(cols: set[str], candidates: tuple[str, ...]) -> tuple[str, ...]:
    """Return the subset of *candidates* that are present in *cols*, in order."""
    return tuple(c for c in candidates if c in cols)


def _return_fields(backend: KgBackend, table: str, alias: str, candidates: tuple[str, ...]) -> str:
    """Build a ``alias.col AS col, ...`` fragment from the columns that actually exist."""
    avail = _pick(_table_columns(backend, table), candidates)
    if not avail:
        raise DocumentGraphError(
            f"The '{table}' table has none of the expected columns ({', '.join(candidates)}). "
            "The database may not have been ingested as a Document Graph — run `cli docgraph build` first."
        )
    return ", ".join(f"{alias}.{c} AS {c}" for c in avail)


def _normalize_row(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Ensure every key in *keys* is present on *row* (defaulting to None)."""
    return {k: row.get(k) for k in keys}


def _query_rows(
    backend: KgBackend, query: str, parameters: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], str]:
    """Run a Cypher query and return (rows, query_string).

    A fresh (never-ingested) or just-dropped database legitimately has no
    Document/MarkdownSection tables — treat that as "no results".
    """
    try:
        df = backend.execute_get_as_df(query, parameters, union=False)
    except Exception as exc:  # noqa: BLE001
        if "does not exist" in str(exc):
            logger.debug("Document Graph table not found (not yet ingested?): {}", exc)
            return [], query
        raise
    # pandas renders SQL/Cypher NULLs (e.g. a root section's parent_section_id) as
    # float NaN rather than None — normalize so callers can compare with `is None`.
    return df.astype(object).where(df.notna(), None).to_dict(orient="records"), query


def _resolve_markdown_hash(backend: KgBackend, document_id: str) -> str | None:
    """Resolve a document reference to a Document.markdown_hash.

    Accepts a Document content hash (full or prefix), a ``markdown_hash``, a
    filename, or a source Document path.
    """
    query = (
        f"MATCH (d:{_DOCUMENT_LABEL}) "
        "WHERE d.content_hash = $id OR d.content_hash STARTS WITH $id OR d.markdown_hash = $id "
        "OR d.markdown_hash STARTS WITH $id OR d.filename = $id OR d.path = $id "
        "RETURN d.markdown_hash AS h LIMIT 1"
    )
    rows, _ = _query_rows(backend, query, {"id": document_id})
    if rows:
        return rows[0]["h"]
    return None


def resolve_folder_id(backend: KgBackend, folder_ref: str) -> str | None:
    """Resolve a folder reference (hash, hash prefix, or name) to a Folder.folder_id."""
    query = (
        f"MATCH (f:{_FOLDER_LABEL}) "
        "WHERE f.folder_id = $id OR f.folder_id STARTS WITH $id OR f.name = $id "
        "RETURN f.folder_id AS id LIMIT 1"
    )
    rows, _ = _query_rows(backend, query, {"id": folder_ref})
    if rows:
        return rows[0]["id"]
    return None


def get_folder_path(backend: KgBackend, folder_id: str) -> list[dict[str, Any]]:
    """Return the ancestor chain (root-first, ``folder_id``-last) for a folder, for breadcrumb display.

    On a schema without ``Folder.parent_folder_id`` the chain is just the folder
    itself (folders are stored flat).
    """
    cols = _table_columns(backend, _FOLDER_LABEL)
    if "parent_folder_id" not in cols:
        rows, _ = _query_rows(
            backend,
            f"MATCH (f:{_FOLDER_LABEL} {{folder_id: $id}}) "
            "RETURN f.folder_id AS folder_id, f.parent_folder_id AS parent_folder_id, f.name AS name, "
            "f.kind AS kind, f.uri AS uri",
            {"id": folder_id},
        )
        return [_normalize_row(r, _FOLDER_KEYS) for r in rows]

    chain: list[dict[str, Any]] = []
    current: str | None = folder_id
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        rows, _ = _query_rows(
            backend,
            f"MATCH (f:{_FOLDER_LABEL} {{folder_id: $id}}) "
            "RETURN f.folder_id AS folder_id, f.parent_folder_id AS parent_folder_id, f.name AS name, "
            "f.kind AS kind, f.uri AS uri",
            {"id": current},
        )
        if not rows:
            break
        chain.append(_normalize_row(rows[0], _FOLDER_KEYS))
        current = rows[0]["parent_folder_id"]
    chain.reverse()
    return chain


def get_folder_tree(backend: KgBackend, root_folder_id: str | None = None) -> list[dict[str, Any]]:
    """Return the folder hierarchy (subfolders + direct document counts) rooted at *root_folder_id*.

    Each row is ``{folder_id, parent_folder_id, name, kind, uri, doc_count}``.
    When ``root_folder_id`` is None, returns every top-level source folder (those
    with no parent) plus their full descendant subtree.

    On a schema without ``HAS_SUBFOLDER`` the hierarchy is flat: the single root
    (when given) or every folder, each with its direct document count.
    """
    if not _has_table(backend, _FOLDER_LABEL):
        return []
    cols = _table_columns(backend, _FOLDER_LABEL)
    folder_fields = ", ".join(
        f"f.{c} AS {c}" for c in _pick(cols, ("folder_id", "parent_folder_id", "name", "kind", "uri"))
    )
    has_subfolder = _has_relationship(backend, "HAS_SUBFOLDER")

    def _with_counts(folder_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for r in folder_rows:
            r = _normalize_row(r, _FOLDER_KEYS)
            r["doc_count"] = 0
            by_id[r["folder_id"]] = r
        if by_id and _has_table(backend, _DOCUMENT_LABEL):
            ids = list(by_id)
            count_rows, _ = _query_rows(
                backend,
                f"MATCH (f:{_FOLDER_LABEL})-[:CONTAINS]->(d:{_DOCUMENT_LABEL}) "
                "WHERE f.folder_id IN $ids RETURN f.folder_id AS fid, count(d) AS n",
                {"ids": ids},
            )
            for r in count_rows:
                fid = r["fid"]
                if fid in by_id:
                    by_id[fid]["doc_count"] = int(r["n"] or 0)
        return sorted(by_id.values(), key=lambda r: r["name"] or "")

    if not has_subfolder:
        if root_folder_id is not None:
            rows, _ = _query_rows(
                backend,
                f"MATCH (f:{_FOLDER_LABEL} {{folder_id: $id}}) RETURN {folder_fields}",
                {"id": root_folder_id},
            )
        else:
            rows, _ = _query_rows(backend, f"MATCH (f:{_FOLDER_LABEL}) RETURN {folder_fields}")
        return _with_counts(rows)

    if root_folder_id is None:
        root_rows, _ = _query_rows(
            backend, f"MATCH (r:{_FOLDER_LABEL}) WHERE r.parent_folder_id IS NULL RETURN r.folder_id AS folder_id"
        )
        root_ids = [r["folder_id"] for r in root_rows]
    else:
        root_ids = [root_folder_id]

    by_folder_id: dict[str, dict[str, Any]] = {}
    for root_id in root_ids:
        query = f"""
            MATCH (root:{_FOLDER_LABEL} {{folder_id: $root_id}})-[:HAS_SUBFOLDER*0..30]->(f:{_FOLDER_LABEL})
            OPTIONAL MATCH (f)-[:CONTAINS]->(d:{_DOCUMENT_LABEL})
            RETURN {folder_fields}, count(d) AS doc_count
        """
        rows, _ = _query_rows(backend, query, {"root_id": root_id})
        for row in rows:
            row = _normalize_row(row, _FOLDER_KEYS)
            row["doc_count"] = int(row.get("doc_count") or 0)
            by_folder_id[row["folder_id"]] = row

    return sorted(by_folder_id.values(), key=lambda r: r["name"] or "")


def list_documents(backend: KgBackend, folder_id: str | None = None) -> list[dict[str, Any]]:
    """List ingested documents with their section count and hashes.

    Args:
        folder_id: When given, only return documents under this folder's subtree
            (the folder itself or any nested subfolder). On a schema without
            ``HAS_SUBFOLDER``, restricted to the folder's direct documents.

    Returns:
        List of ``{content_hash, markdown_hash, filename, section_count, token_count,
        description, summary, path, folder_id}`` dicts (optional fields are ``None``
        when absent from the DB).
    """
    if not _has_table(backend, _DOCUMENT_LABEL):
        return []
    fields = _return_fields(backend, _DOCUMENT_LABEL, "d", _DOC_KEYS)
    order = "ORDER BY d.filename"

    if folder_id is None:
        query = f"MATCH (d:{_DOCUMENT_LABEL}) RETURN {fields} {order}"
        params: dict[str, Any] = {}
    elif _has_relationship(backend, "HAS_SUBFOLDER"):
        query = f"""
            MATCH (root:{_FOLDER_LABEL} {{folder_id: $folder_id}})-[:HAS_SUBFOLDER*0..30]->(f:{_FOLDER_LABEL})
            MATCH (f)-[:CONTAINS]->(d:{_DOCUMENT_LABEL})
            RETURN {fields} {order}
        """
        params = {"folder_id": folder_id}
    else:
        query = f"""
            MATCH (f:{_FOLDER_LABEL} {{folder_id: $folder_id}})-[:CONTAINS]->(d:{_DOCUMENT_LABEL})
            RETURN {fields} {order}
        """
        params = {"folder_id": folder_id}

    rows, _ = _query_rows(backend, query, params)
    out: list[dict[str, Any]] = []
    for row in rows:
        row = _normalize_row(row, _DOC_KEYS)
        row["section_count"] = int(row.get("section_count") or 0)
        row["token_count"] = int(row.get("token_count") or 0)
        out.append(row)
    return out


def get_document(backend: KgBackend, document_id: str) -> dict[str, Any] | None:
    """Return one Document's full metadata, including ``token_count`` and ``summary``.

    Accepts the same references as `get_document_toc`: a content hash (full or
    prefix), a ``markdown_hash``, a filename, or a source path.
    """
    if not _has_table(backend, _DOCUMENT_LABEL):
        return None
    fields = _return_fields(backend, _DOCUMENT_LABEL, "d", _DOC_KEYS)
    query = (
        f"MATCH (d:{_DOCUMENT_LABEL}) "
        "WHERE d.content_hash = $id OR d.content_hash STARTS WITH $id OR d.markdown_hash = $id "
        "OR d.markdown_hash STARTS WITH $id OR d.filename = $id OR d.path = $id "
        f"RETURN {fields} LIMIT 1"
    )
    rows, _ = _query_rows(backend, query, {"id": document_id})
    if not rows:
        return None
    row = _normalize_row(rows[0], _DOC_KEYS)
    row["section_count"] = int(row.get("section_count") or 0)
    row["token_count"] = int(row.get("token_count") or 0)
    return row


def apply_section_summaries(backend: KgBackend, rows: list[dict[str, Any]]) -> int:
    """Write `description`/`summary`/`summary_source` onto MarkdownSection nodes.

    Args:
        rows: `{section_id, description, summary, summary_source}` dicts, one per section.

    Returns:
        Number of sections updated.
    """
    if not rows:
        return 0
    query = (
        "UNWIND $rows AS row "
        f"MATCH (s:{_SECTION_LABEL} {{section_id: row.section_id}}) "
        "SET s.description = row.description, s.summary = row.summary, s.summary_source = row.summary_source"
    )
    backend.execute(query, {"rows": rows})
    return len(rows)


def apply_document_summary(
    backend: KgBackend, markdown_hash: str, *, description: str | None = None, summary: str | None = None
) -> None:
    """Write the document-level description/abstract onto every Document sharing *markdown_hash*."""
    backend.execute(
        f"MATCH (d:{_DOCUMENT_LABEL} {{markdown_hash: $markdown_hash}}) "
        "SET d.description = $description, d.summary = $summary",
        {"markdown_hash": markdown_hash, "description": description, "summary": summary},
    )


def get_document_toc(
    backend: KgBackend, document_id: str, return_query: bool = False
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    """Return the table of contents (heading tree) for one document.

    Section body text is not returned, but `token_count` and `summary` are — this
    is the map an agent uses to decide which sections to fetch in full with
    `get_section_content`.
    """
    markdown_hash = _resolve_markdown_hash(backend, document_id)
    if not markdown_hash:
        result: list[dict[str, Any]] = []
        query = f"-- No document found matching: {document_id}"
        return (result, query) if return_query else result

    # Document exists but its MarkdownSection table was dropped (e.g. after a partial
    # drop) — degrade to an empty TOC rather than raising a binder error.
    if not _has_table(backend, _SECTION_LABEL):
        result: list[dict[str, Any]] = []
        query = f"-- No sections table for document: {document_id}"
        return (result, query) if return_query else result

    fields = _return_fields(backend, _SECTION_LABEL, "s", _SECTION_TOC_KEYS)
    query = f"""
        MATCH (s:{_SECTION_LABEL} {{markdown_hash: $markdown_hash}})
        RETURN {fields}
        ORDER BY s.sequence
    """
    rows, q = _query_rows(backend, query, {"markdown_hash": markdown_hash})
    rows = [_normalize_row(r, _SECTION_TOC_KEYS) for r in rows]
    return (rows, q) if return_query else rows


def build_toc_tree(
    toc_rows: list[dict[str, Any]], *, include_summaries: bool = False, max_level: int | None = None
) -> list[dict[str, Any]]:
    """Nest flat `get_document_toc` rows into a tree by `parent_section_id`.

    Each node is `{id, title, description?, summary?, sections?}`. Heading level and
    token count are deliberately not emitted: level is redundant with the tree's own
    nesting, and token count adds nothing an agent can act on once `description`
    already answers "is this worth opening?". The synthetic level-0 root section is
    unwrapped — it is a container for preamble text, not a heading an agent would
    navigate to.

    Args:
        include_summaries: Also emit the fuller `summary` where one exists. Off by
            default so the agent sees descriptions first without blowing context.
        max_level: Drop sections deeper than this heading level.
    """
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for row in toc_rows:
        by_parent.setdefault(row.get("parent_section_id"), []).append(row)

    def build(parent_id: str | None) -> list[dict[str, Any]]:
        nodes = []
        for row in sorted(by_parent.get(parent_id, []), key=lambda r: r["sequence"]):
            if max_level is not None and int(row["level"]) > max_level:
                continue
            node: dict[str, Any] = {
                "id": row["section_id"],
                "title": row["title"],
            }
            if row.get("description"):
                node["description"] = row["description"]
            if include_summaries and row.get("summary"):
                node["summary"] = row["summary"]
            children = build(row["section_id"])
            if children:
                node["sections"] = children
            nodes.append(node)
        return nodes

    root_rows = [r for r in toc_rows if r.get("level") == 0]
    if root_rows:
        return build(root_rows[0]["section_id"])
    return build(None)


def render_toc_outline(toc_rows: list[dict[str, Any]]) -> str:
    """Render a document's TOC as compact indented text (`- [id] Title`), for LLM prompts and the CLI."""
    lines = []
    for row in toc_rows:
        if int(row.get("level") or 0) == 0:
            continue  # synthetic root: not a navigable heading
        indent = "  " * max(int(row["level"]) - 1, 0)
        lines.append(f"{indent}- [{row['section_id']}] {row['title']} (line {row['line_start']})")
    return "\n".join(lines)


def document_toc_yaml(
    backend: KgBackend, document_id: str, *, include_summaries: bool = False, max_level: int | None = None
) -> str:
    """Return one document's table of contents as a YAML string.

    Section `description`s are always included (they are the routing signal), and
    the fuller per-section `summary` can be included by passing `include_summaries=True`.
    """
    doc = get_document(backend, document_id)
    toc_rows = get_document_toc(backend, document_id)
    if doc is None or not toc_rows:
        return yaml.safe_dump({"error": f"No document found matching: {document_id}"}, sort_keys=False)
    payload: dict[str, Any] = {
        "document": doc["filename"],
        "id": doc["content_hash"],
    }
    if doc.get("description"):
        payload["description"] = doc["description"]
    if doc.get("summary"):
        payload["summary"] = doc["summary"]
    payload["sections"] = build_toc_tree(
        toc_rows,  # type: ignore[arg-type]
        include_summaries=include_summaries,
        max_level=max_level,
    )
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def folder_toc_yaml(
    backend: KgBackend,
    folder_id: str | None,
    *,
    include_sections: bool = False,
    include_summaries: bool = True,
    max_level: int | None = None,
) -> str:
    """Return the documents under a folder's subtree as one YAML string.

    Sections are omitted by default — this is the *orientation* view an agent reads
    first to pick a document, and inlining every section of every document defeats
    the point (and blows the context window on a large corpus). Call
    `document_toc_yaml` for the chosen document, or pass `include_sections=True`.
    """
    docs = list_documents(backend, folder_id=folder_id)
    if not docs:
        return yaml.safe_dump({"documents": []}, sort_keys=False)
    payload: dict[str, Any] = {"documents": []}
    for doc in docs:
        entry: dict[str, Any] = {
            "id": doc["content_hash"],
            "name": doc["filename"],
            "sections": doc["section_count"],
        }
        if doc.get("description"):
            entry["description"] = doc["description"]
        if include_summaries and doc.get("summary"):
            entry["summary"] = doc["summary"]
        if include_sections:
            toc_rows = get_document_toc(backend, doc["markdown_hash"])
            entry["toc"] = build_toc_tree(
                toc_rows,  # type: ignore[arg-type]
                include_summaries=include_summaries,
                max_level=max_level,
            )
        payload["documents"].append(entry)
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def get_section_content(
    backend: KgBackend, section_ids: list[str], return_query: bool = False
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    """Fetch the raw Markdown text of one or more sections.

    Each entry in *section_ids* may be a full ``section_id`` (``{markdown_hash}::{sequence}``)
    or a prefix of one — e.g. a bare (or truncated) document hash matches every section of
    that document, so `cli docgraph list`'s short hash column can be used directly.
    """
    query = f"""
        UNWIND $section_ids AS sid
        MATCH (s:{_SECTION_LABEL})
        WHERE s.section_id = sid OR s.section_id STARTS WITH sid
        RETURN DISTINCT s.section_id AS section_id, s.markdown_hash AS markdown_hash, s.title AS title,
               s.line_start AS line_start, s.line_end AS line_end, s.sequence AS sequence, s.text AS text
        ORDER BY markdown_hash, sequence
    """
    rows, _ = _query_rows(backend, query, {"section_ids": section_ids})
    return (rows, query) if return_query else rows


def reconstruct_document(
    backend: KgBackend, document_id: str, return_query: bool = False
) -> str | None | tuple[str | None, str]:
    """Rebuild a document's full Markdown text by concatenating its sections.

    Sections partition the document's lines without overlap, so concatenating
    their ``text`` in ``sequence`` order reproduces the original document.
    """
    query = f"MATCH (s:{_SECTION_LABEL} {{markdown_hash: $markdown_hash}}) RETURN s.text AS text ORDER BY s.sequence"
    markdown_hash = _resolve_markdown_hash(backend, document_id)
    if not markdown_hash:
        return (None, query) if return_query else None
    rows, _ = _query_rows(backend, query, {"markdown_hash": markdown_hash})
    text = "\n".join(r["text"] for r in rows)
    return (text, query) if return_query else text


def _collect_subtree_section_ids(toc_rows: list[dict[str, Any]], root_section_id: str) -> list[str]:
    """Return *root_section_id* plus every descendant section_id (any depth)."""
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for row in toc_rows:
        by_parent.setdefault(row["parent_section_id"], []).append(row)

    ids = [root_section_id]

    def collect(parent_id: str) -> None:
        for row in by_parent.get(parent_id, []):
            ids.append(row["section_id"])
            collect(row["section_id"])

    collect(root_section_id)
    return ids


def reconstruct_section(
    backend: KgBackend, section_id: str, return_query: bool = False
) -> str | None | tuple[str | None, str]:
    """Rebuild the Markdown text of one section plus all of its nested subsections.

    Accepts a full ``section_id`` (``{markdown_hash}::{sequence}``) or a prefix of one.
    """
    query = (
        f"MATCH (s:{_SECTION_LABEL}) WHERE s.section_id = $id OR s.section_id STARTS WITH $id "
        "RETURN s.section_id AS section_id, s.markdown_hash AS markdown_hash LIMIT 1"
    )
    rows, _ = _query_rows(backend, query, {"id": section_id})
    if not rows:
        return (None, query) if return_query else None

    resolved_id = rows[0]["section_id"]
    markdown_hash = rows[0]["markdown_hash"]

    toc = get_document_toc(backend, markdown_hash)
    subtree_ids = _collect_subtree_section_ids(toc, resolved_id)  # type: ignore[arg-type]

    content_rows = get_section_content(backend, subtree_ids)
    content_rows.sort(key=lambda r: r["sequence"])  # type: ignore[union-attr]
    text = "\n".join(r["text"] for r in content_rows)  # type: ignore[union-attr]
    return (text, query) if return_query else text


_CHUNK_LABEL = SectionChunkNode.node_class.__name__
# Index names must match genai_graph.kg.document_graph.retrieval defaults.
_CHUNK_VECTOR_INDEX = "chunk_embedding_index"
_SECTION_FTS_INDEX = "section_fts"
_RRF_K = 60
_HNSW_OVERFETCH = 5
_MAX_CHUNK_SNIPPET = 180


def _scope_markdown_hashes(backend: KgBackend, folder_id: str | None) -> set[str] | None:
    """Return the markdown_hash set under *folder_id*'s subtree, or None (no scope).

    Used to filter ranked candidates to one folder without folding the folder
    traversal into the vector/FTS index calls themselves.
    """
    if folder_id is None:
        return None
    if _has_relationship(backend, "HAS_SUBFOLDER"):
        query = (
            f"MATCH (root:{_FOLDER_LABEL} {{folder_id: $fid}})-[:HAS_SUBFOLDER*0..30]->(f:{_FOLDER_LABEL})"
            f"-[:CONTAINS]->(d:{_DOCUMENT_LABEL}) RETURN DISTINCT d.markdown_hash AS mh"
        )
    else:
        query = f"MATCH (f:{_FOLDER_LABEL} {{folder_id: $fid}})-[:CONTAINS]->(d:{_DOCUMENT_LABEL}) RETURN DISTINCT d.markdown_hash AS mh"
    rows, _ = _query_rows(backend, query, {"fid": folder_id})
    return {r["mh"] for r in rows}


def _semantic_section_hits(
    backend: KgBackend, query_vec: list[float], limit: int, allowed: set[str] | None
) -> list[dict[str, Any]]:
    """HNSW search over SectionChunk embeddings, deduped to parent sections (best chunk)."""
    k = max(limit * _HNSW_OVERFETCH, 50)
    query = (
        f"CALL QUERY_VECTOR_INDEX('{_CHUNK_LABEL}','{_CHUNK_VECTOR_INDEX}', $query_vector, {k}) "
        "RETURN node.section_id AS section_id, node.markdown_hash AS markdown_hash, "
        "node.chunk_id AS chunk_id, node.chunk_text AS chunk_text, distance AS distance "
        "ORDER BY distance ASC"
    )
    rows, _ = _query_rows(backend, query, {"query_vector": query_vec})
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        if allowed is not None and r.get("markdown_hash") not in allowed:
            continue
        sid = r["section_id"]
        if sid not in best or r["distance"] < best[sid]["distance"]:
            best[sid] = r
    return sorted(best.values(), key=lambda r: r["distance"])[:limit]


def _contains_section_hits(backend: KgBackend, keyword: str, limit: int, folder_id: str | None) -> list[dict[str, Any]]:
    """Legacy CONTAINS search (fallback when the FTS index is unavailable)."""
    where = "WHERE s.title CONTAINS $keyword OR s.text CONTAINS $keyword"
    ret = (
        "RETURN s.markdown_hash AS markdown_hash, s.section_id AS section_id, s.title AS title, "
        "s.level AS level, s.line_start AS line_start "
        "ORDER BY s.markdown_hash, s.line_start LIMIT $limit"
    )
    if folder_id is None:
        query = f"MATCH (s:{_SECTION_LABEL}) {where} {ret}"
        params: dict[str, Any] = {"keyword": keyword, "limit": limit}
    elif _has_relationship(backend, "HAS_SUBFOLDER"):
        query = (
            f"MATCH (root:{_FOLDER_LABEL} {{folder_id: $folder_id}})-[:HAS_SUBFOLDER*0..30]->(f:{_FOLDER_LABEL}) "
            f"MATCH (f)-[:CONTAINS]->(d:{_DOCUMENT_LABEL}) "
            f"MATCH (s:{_SECTION_LABEL} {{markdown_hash: d.markdown_hash}}) {where} {ret}"
        )
        params = {"keyword": keyword, "limit": limit, "folder_id": folder_id}
    else:
        query = (
            f"MATCH (f:{_FOLDER_LABEL} {{folder_id: $folder_id}})-[:CONTAINS]->(d:{_DOCUMENT_LABEL}) "
            f"MATCH (s:{_SECTION_LABEL} {{markdown_hash: d.markdown_hash}}) {where} {ret}"
        )
        params = {"keyword": keyword, "limit": limit, "folder_id": folder_id}
    rows, _ = _query_rows(backend, query, params)
    return rows


def _keyword_section_hits(
    backend: KgBackend, query: str, limit: int, folder_id: str | None, allowed: set[str] | None
) -> list[dict[str, Any]]:
    """BM25 (FTS) search over MarkdownSection; falls back to CONTAINS."""
    if isinstance(backend, KuzuBackend):
        try:
            backend.ensure_fts_extension()
            fts = (
                f"CALL QUERY_FTS_INDEX('{_SECTION_LABEL}','{_SECTION_FTS_INDEX}', $query) "
                "RETURN node.markdown_hash AS markdown_hash, node.section_id AS section_id, "
                "node.title AS title, node.level AS level, node.line_start AS line_start, "
                "score AS score ORDER BY score DESC LIMIT $limit"
            )
            rows, _ = _query_rows(backend, fts, {"query": query, "limit": limit})
            if allowed is not None:
                rows = [r for r in rows if r.get("markdown_hash") in allowed]
            return rows
        except Exception as exc:  # noqa: BLE001
            logger.debug("FTS search unavailable, falling back to CONTAINS: {}", exc)
    rows = _contains_section_hits(backend, query, limit, folder_id)
    if allowed is not None:
        rows = [r for r in rows if r.get("markdown_hash") in allowed]
    return rows


def _fetch_section_meta(backend: KgBackend, section_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return section_id -> {title, level, line_start, markdown_hash} for the given ids."""
    if not section_ids:
        return {}
    query = (
        f"MATCH (s:{_SECTION_LABEL}) WHERE s.section_id IN $ids "
        "RETURN s.section_id AS section_id, s.title AS title, s.level AS level, "
        "s.line_start AS line_start, s.markdown_hash AS markdown_hash"
    )
    rows, _ = _query_rows(backend, query, {"ids": section_ids})
    return {r["section_id"]: r for r in rows}


def search_sections(
    backend: KgBackend,
    query: str,
    limit: int = 20,
    folder_id: str | None = None,
    *,
    mode: str = "hybrid",
    embeddings_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search sections by hybrid (HNSW + BM25/RRF), semantic, or keyword mode.

    Args:
        query: Natural-language query (embedded for semantic; used as the BM25
            term for keyword).
        limit: Max sections to return.
        folder_id: Restrict to one folder's subtree.
        mode: "hybrid" (default) fuses vector + BM25 via reciprocal rank fusion;
            "semantic" uses only the SectionChunk vector index; "keyword" uses only
            FTS/BM25 (CONTAINS fallback). Hybrid and semantic degrade to keyword
            search when embeddings are unavailable.
        embeddings_id: Model id for query embedding (required for the vector leg).
            None disables semantic/hybrid vector search.

    Returns:
        Ranked section dicts (best first) with ``section_id``, ``markdown_hash``,
        ``title``, ``level``, ``line_start``, ``score`` and, when a chunk matched,
        ``matched_chunk``.
    """
    allowed = _scope_markdown_hashes(backend, folder_id)

    sem: list[dict[str, Any]] = []
    if mode in ("hybrid", "semantic") and embeddings_id and isinstance(backend, KuzuBackend):
        try:
            handler = EmbeddingsHandler(embeddings_id=embeddings_id)
            query_vec = handler.compute_embeddings(query)
            sem = _semantic_section_hits(backend, query_vec, limit, allowed)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Semantic search unavailable, keyword-only: {}", exc)
            sem = []

    kw: list[dict[str, Any]] = []
    if mode in ("hybrid", "keyword") or not sem:
        kw = _keyword_section_hits(backend, query, limit, folder_id, allowed)

    sem_sorted = sorted(sem, key=lambda r: r["distance"])
    sem_rank = {r["section_id"]: i for i, r in enumerate(sem_sorted)}
    sem_by_sid = {r["section_id"]: r for r in sem_sorted}

    kw_sorted = sorted(kw, key=lambda r: (r.get("score") is None, -(r.get("score") or 0)))
    kw_rank = {r["section_id"]: i for i, r in enumerate(kw_sorted)}
    kw_by_sid = {r["section_id"]: r for r in kw}

    sem_available = bool(sem)
    all_sids = set(sem_rank) | set(kw_rank)
    meta = _fetch_section_meta(backend, list(all_sids))

    results: list[dict[str, Any]] = []
    for sid in all_sids:
        m = meta.get(sid, {})
        srow = sem_by_sid.get(sid, {})
        krow = kw_by_sid.get(sid, {})
        if mode == "hybrid" and sem_available:
            sc = 0.0
            if sid in sem_rank:
                sc += 1.0 / (_RRF_K + sem_rank[sid] + 1)
            if sid in kw_rank:
                sc += 1.0 / (_RRF_K + kw_rank[sid] + 1)
        elif mode == "semantic" and sem_available:
            d = srow.get("distance")
            sc = (1.0 - (d / 2.0)) if d is not None else 0.0
        else:  # keyword, or hybrid/semantic degraded to keyword-only
            sc = krow.get("score") or 0.0
        chunk = srow.get("chunk_text")
        if chunk and len(chunk) > _MAX_CHUNK_SNIPPET:
            chunk = chunk[:_MAX_CHUNK_SNIPPET] + "…"
        results.append(
            {
                "section_id": sid,
                "markdown_hash": m.get("markdown_hash") or srow.get("markdown_hash") or krow.get("markdown_hash"),
                "title": m.get("title") or krow.get("title"),
                "level": m.get("level") or krow.get("level"),
                "line_start": m.get("line_start") or krow.get("line_start"),
                "score": round(sc, 6),
                "matched_chunk": chunk,
                "distance": srow.get("distance"),
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]


def _connect(db_path: str) -> KgBackend:
    """Connect to the Document Graph reusing the process-shared ladybug.Database handle."""
    backend = LadybugBackend()
    shared_db = get_shared_database(db_path)
    backend.attach(shared_db)
    return backend


def _tool_error(exc: Exception) -> str:
    """Turn an exception into a concise, agent-friendly tool result string."""
    if isinstance(exc, DocumentGraphError):
        return f"Error: {exc}"
    return f"Error: {type(exc).__name__}: {exc}"


def create_document_graph_tools(db_path: str, *, embeddings_id: str | None = None) -> list[BaseTool]:
    """Build the LangChain tools an agent uses to navigate a Document Graph.

    Args:
        db_path: Path to the Ladybug database holding the ingested graph.
        embeddings_id: Optional embeddings model id enabling the hybrid (vector +
            BM25) ``search_sections`` mode. None keeps keyword search only.

    Returns:
        ``[list_documents, get_document_toc, get_folder_toc, get_section_content, search_sections]`` tools.
    """

    @tool("list_documents")
    def _list_documents() -> str:
        """List every ingested document with its section count and one-line description."""
        try:
            rows = list_documents(_connect(db_path))
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)
        if not rows:
            return "No documents ingested yet. Build the graph first with `cli docgraph build`."
        lines = []
        for r in rows:
            line = f"- [{r['content_hash']}] {r['filename']} ({r['section_count']} sections, {r['token_count']} tokens)"
            if r.get("description"):
                line += f"\n  {r['description']}"
            lines.append(line)
        return "\n".join(lines)

    @tool("get_folder_toc")
    def _get_folder_toc(folder_id: str | None = None) -> str:
        """Start here. List the documents in a folder, each with an id and a one-line description.

        Sections are NOT included — pick a document from this list, then call
        `get_document_toc` with its id to see that document's sections.
        Omit `folder_id` to cover every ingested document.
        """
        try:
            backend = _connect(db_path)
            resolved = resolve_folder_id(backend, folder_id) if folder_id else None
            if folder_id and resolved is None:
                return f"No folder found matching {folder_id!r}. Omit folder_id to list every ingested document."
            return folder_toc_yaml(backend, resolved)
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)

    @tool("get_document_toc")
    def _get_document_toc(document_id: str, include_summaries: bool = True, max_level: int | None = None) -> str:
        """Get one document's section tree as YAML: each section's id, title, size and description.

        Use the section ids from here with `get_section_content` to read the actual text.
        `document_id` is a content hash (full or prefix), a filename, or a source path.
        Per-section `summary` is included by default where one was generated; pass
        `include_summaries=False` to omit them. Use `max_level` to show only the
        top-level sections of a very long document.
        """
        try:
            return document_toc_yaml(
                _connect(db_path), document_id, include_summaries=include_summaries, max_level=max_level
            )
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)

    @tool("get_section_content")
    def _get_section_content(section_ids: str) -> str:
        """Fetch the raw Markdown text of one or more sections. Comma-separated section_ids."""
        ids = [s.strip() for s in section_ids.split(",") if s.strip()]
        try:
            rows = get_section_content(_connect(db_path), ids)
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)
        if not rows:
            return f"No sections found for ids: {section_ids}"
        return "\n\n---\n\n".join(f"### [{r['section_id']}] {r['title']}\n\n{r['text']}" for r in rows)

    @tool("search_sections")
    def _search_sections(query: str, limit: int = 20, folder_id: str | None = None, mode: str = "hybrid") -> str:
        """Semantic + keyword search over section titles, text, and chunks, ranked by relevance.

        Hybrid (default) fuses SectionChunk vector search with BM25 keyword search
        (FTS over section title/text) via reciprocal rank fusion. Use ``mode="semantic"``
        or ``mode="keyword"`` for one side only. Pass ``folder_id`` to restrict to one
        folder's subtree. Results are ranked best-first with a relevance score and,
        when a chunk matched, a short snippet of the matching text.
        """
        try:
            backend = _connect(db_path)
            resolved = resolve_folder_id(backend, folder_id) if folder_id else None
            rows = search_sections(
                backend, query, limit=limit, folder_id=resolved, mode=mode, embeddings_id=embeddings_id
            )
        except Exception as exc:  # noqa: BLE001
            return _tool_error(exc)
        if not rows:
            return f"No sections matched: {query!r}"
        lines = []
        for r in rows:
            line = f"- [{r['section_id']}] {r['title']} (line {r['line_start']}) — score {r['score']}"
            if r.get("matched_chunk"):
                line += f"\n    matched: {r['matched_chunk']!r}"
            lines.append(line)
        return "\n".join(lines)

    return [_get_folder_toc, _get_document_toc, _get_section_content, _search_sections, _list_documents]
