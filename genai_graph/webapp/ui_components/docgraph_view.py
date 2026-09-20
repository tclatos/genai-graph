"""UI helpers, query adapters, and tree structures for the Document Graph Streamlit browser."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path, PurePosixPath
from typing import Any

import streamlit as st
from loguru import logger

from genai_graph.kg.backend import KgBackend, KuzuBackend
from genai_graph.kg.query.document_graph_tools import (
    _DOCUMENT_LABEL,
    _FOLDER_LABEL,
    _SECTION_LABEL,
    _has_table,
    _query_rows,
    _return_fields,
    get_folder_tree,
    list_documents,
    search_images,
    search_sections,
    search_tables,
)

# Regex patterns for fallback detection in raw markdown
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_HTML_TABLE_RE = re.compile(r"<\s*table(?:\s+[^>]*)?>", re.IGNORECASE)
_MD_IMAGE_RE = re.compile(r"!\[(?P<alt>.*?)\]\((?P<url>[^\s\)\"\']+)(?:\s+[\"'](?P<title>.*?)[\"'])?\)")
_MISTRAL_IMG_RE = re.compile(
    r"<!--\s*Image:\s*(?P<filename>[^\s\(\)]+)\s*\(hash:\s*(?P<hash>[a-fA-F0-9]+)\)\s*-->",
    re.IGNORECASE,
)
_HTML_IMG_RE = re.compile(r"<\s*img\s+[^>]*src=[\"'](?P<url>[^\"']+)[\"']", re.IGNORECASE)
_ORIGIN_COMMENT_RE = re.compile(r"<!--\s*source:\s*(.+?)\s*-->")


def read_origin_path(md_path: str | None) -> str | None:
    """Return the original source document path recorded in a converted Markdown file, if any."""
    if not md_path:
        return None
    try:
        with open(md_path, encoding="utf-8") as f:
            first_line = f.readline()
    except OSError:
        return None
    match = _ORIGIN_COMMENT_RE.search(first_line)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------------------------


def get_docgraph_backend(db_path: str) -> KgBackend:
    """Connect to a Ladybug/Kuzu database and return the backend instance."""
    backend = KuzuBackend()
    backend.connect(db_path)
    return backend


# ---------------------------------------------------------------------------
# Async query adapters (leveraging Streamlit's native async/await support)
# ---------------------------------------------------------------------------


async def fetch_database_stats(backend: KgBackend) -> dict[str, int]:
    """Asynchronously collect high-level counts and statistics of the Document Graph."""

    def _query() -> dict[str, int]:
        stats: dict[str, int] = {
            "folders": 0,
            "documents": 0,
            "sections": 0,
            "images": 0,
            "tables": 0,
            "chunks": 0,
            "total_tokens": 0,
        }
        if _has_table(backend, _FOLDER_LABEL):
            rows, _ = _query_rows(backend, f"MATCH (f:{_FOLDER_LABEL}) RETURN count(f) AS cnt")
            stats["folders"] = int(rows[0]["cnt"]) if rows else 0

        if _has_table(backend, _DOCUMENT_LABEL):
            rows, _ = _query_rows(
                backend,
                f"MATCH (d:{_DOCUMENT_LABEL}) RETURN count(d) AS cnt, coalesce(sum(d.token_count), 0) AS tokens",
            )
            if rows:
                stats["documents"] = int(rows[0]["cnt"])
                stats["total_tokens"] = int(rows[0].get("tokens") or 0)

        if _has_table(backend, _SECTION_LABEL):
            rows, _ = _query_rows(backend, f"MATCH (s:{_SECTION_LABEL}) RETURN count(s) AS cnt")
            stats["sections"] = int(rows[0]["cnt"]) if rows else 0

        if _has_table(backend, "Image"):
            rows, _ = _query_rows(backend, "MATCH (i:Image) RETURN count(i) AS cnt")
            stats["images"] = int(rows[0]["cnt"]) if rows else 0

        if _has_table(backend, "MarkdownTable"):
            rows, _ = _query_rows(backend, "MATCH (t:MarkdownTable) RETURN count(t) AS cnt")
            stats["tables"] = int(rows[0]["cnt"]) if rows else 0

        if _has_table(backend, "SectionChunk"):
            rows, _ = _query_rows(backend, "MATCH (c:SectionChunk) RETURN count(c) AS cnt")
            stats["chunks"] = int(rows[0]["cnt"]) if rows else 0

        return stats

    return await asyncio.to_thread(_query)


async def fetch_folders_and_documents(
    backend: KgBackend, folder_id: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Asynchronously fetch the list of folders and documents."""

    def _query() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        folders = get_folder_tree(backend, root_folder_id=folder_id)
        docs = list_documents(backend, folder_id=folder_id)
        return folders, docs

    return await asyncio.to_thread(_query)


async def fetch_document_sections_full(backend: KgBackend, markdown_hash: str) -> list[dict[str, Any]]:
    """Asynchronously fetch all sections for a document including body text, tables, and images."""

    def _query() -> list[dict[str, Any]]:
        if not _has_table(backend, _SECTION_LABEL):
            return []

        fields = _return_fields(
            backend,
            _SECTION_LABEL,
            "s",
            (
                "section_id",
                "markdown_hash",
                "parent_section_id",
                "title",
                "level",
                "line_start",
                "line_end",
                "token_count",
                "sequence",
                "description",
                "summary",
                "summary_source",
                "keywords",
                "text",
            ),
        )
        cypher = f"""
            MATCH (s:{_SECTION_LABEL} {{markdown_hash: $markdown_hash}})
            RETURN {fields}
            ORDER BY s.sequence
        """
        rows, _ = _query_rows(backend, cypher, {"markdown_hash": markdown_hash})

        # Fetch table counts for this document
        tables_by_sec: dict[str, list[dict[str, Any]]] = {}
        if _has_table(backend, "MarkdownTable"):
            t_rows, _ = _query_rows(
                backend,
                """
                MATCH (s:MarkdownSection)-[:HAS_TABLE]->(t:MarkdownTable)
                WHERE s.markdown_hash = $markdown_hash
                RETURN s.section_id AS section_id, t.table_id AS table_id, t.name AS name,
                       t.table_format AS table_format, t.caption AS caption,
                       t.token_count AS token_count, t.content AS content
                """,
                {"markdown_hash": markdown_hash},
            )
            for tr in t_rows:
                tables_by_sec.setdefault(tr["section_id"], []).append(tr)

        # Fetch image counts for this document
        images_by_sec: dict[str, list[dict[str, Any]]] = {}
        if _has_table(backend, "Image"):
            i_rows, _ = _query_rows(
                backend,
                """
                MATCH (s:MarkdownSection)-[:HAS_IMAGE]->(i:Image)
                WHERE s.markdown_hash = $markdown_hash
                RETURN s.section_id AS section_id, i.image_id AS image_id, i.name AS name,
                       i.filename AS filename, i.path AS path, i.description AS description,
                       i.size AS size
                """,
                {"markdown_hash": markdown_hash},
            )
            for ir in i_rows:
                images_by_sec.setdefault(ir["section_id"], []).append(ir)

        # Fetch graph entity links for this document's sections
        graph_links_by_sec: dict[str, list[dict[str, Any]]] = {}
        try:
            # Query non-structural relationships connected to MarkdownSection
            rel_rows, _ = _query_rows(
                backend,
                """
                MATCH (s:MarkdownSection)-[r]-(e)
                WHERE s.markdown_hash = $markdown_hash
                  AND type(r) NOT IN ['HAS_SECTION', 'HAS_SUBSECTION', 'HAS_CHUNK', 'HAS_TABLE', 'HAS_IMAGE']
                RETURN s.section_id AS section_id, type(r) AS rel_type,
                       labels(e) AS entity_labels,
                       coalesce(e.name, e.title, e.id, 'Entity') AS entity_name
                """,
                {"markdown_hash": markdown_hash},
            )
            for gr in rel_rows:
                graph_links_by_sec.setdefault(gr["section_id"], []).append(gr)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Graph links query skipped: {}", exc)

        # Merge and enrich sections
        enriched_sections: list[dict[str, Any]] = []
        for r in rows:
            sid = r["section_id"]
            sec_text = r.get("text") or ""

            # Check for tables
            sec_tables = tables_by_sec.get(sid, [])
            has_table = (
                bool(sec_tables) or bool(_MD_TABLE_ROW_RE.search(sec_text)) or bool(_HTML_TABLE_RE.search(sec_text))
            )

            # Check for images
            sec_images = images_by_sec.get(sid, [])
            has_image = (
                bool(sec_images)
                or bool(_MD_IMAGE_RE.search(sec_text))
                or bool(_MISTRAL_IMG_RE.search(sec_text))
                or bool(_HTML_IMG_RE.search(sec_text))
            )

            # Check for graph links
            sec_graph_links = graph_links_by_sec.get(sid, [])
            has_graph = bool(sec_graph_links)

            r_copy = dict(r)
            r_copy["tables"] = sec_tables
            r_copy["images"] = sec_images
            r_copy["graph_links"] = sec_graph_links
            r_copy["has_table"] = has_table
            r_copy["has_image"] = has_image
            r_copy["has_graph"] = has_graph
            enriched_sections.append(r_copy)

        return enriched_sections

    return await asyncio.to_thread(_query)


async def fetch_all_images_async(
    backend: KgBackend,
    query: str | None = None,
    document_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Asynchronously search and retrieve image nodes."""
    return await asyncio.to_thread(search_images, backend, query=query, document_id=document_id, limit=limit)


async def fetch_all_tables_async(
    backend: KgBackend,
    query: str | None = None,
    document_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Asynchronously search and retrieve table nodes."""
    return await asyncio.to_thread(search_tables, backend, query=query, document_id=document_id, limit=limit)


async def search_sections_async(
    backend: KgBackend,
    query: str,
    *,
    mode: str = "hybrid",
    folder_id: str | None = None,
    document_id: str | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Asynchronously execute section search."""
    return await asyncio.to_thread(
        search_sections,
        backend,
        query,
        mode=mode,
        folder_id=folder_id,
        document_id=document_id,
        limit=limit,
    )


async def fetch_section_markdown_async(backend: KgBackend, section_id: str) -> str | None:
    """Asynchronously reconstruct the full markdown text for a section."""
    from genai_graph.kg.query.document_graph_tools import reconstruct_section

    return await asyncio.to_thread(reconstruct_section, backend, section_id)


# ---------------------------------------------------------------------------
# Document deduplication & Tree Builder for `streamlit-tree-select2`
# ---------------------------------------------------------------------------


def dedupe_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one Document per source path/filename — the richest (most sections)."""
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("path") or row.get("filename") or row.get("markdown_hash") or ""
        current = best.get(key)
        if current is None or (row.get("section_count", 0) > current.get("section_count", 0)):
            best[key] = row
    return sorted(best.values(), key=lambda r: str(r.get("filename") or r.get("path") or ""))


def _folder_of(doc: dict[str, Any]) -> str:
    """Parent directory or folder identifier of a document row, '.' for root."""
    path = doc.get("path") or ""
    if path:
        parent = str(PurePosixPath(path).parent)
        if parent and parent != ".":
            return parent
    folder_id = doc.get("folder_id")
    if folder_id:
        return folder_id
    return "."


def build_tree_select_nodes(
    folders: list[dict[str, Any]] | None,
    documents: list[dict[str, Any]],
    sections_by_doc: dict[str, list[dict[str, Any]]] | None = None,
    include_sections: bool = True,
    max_section_level: int | None = None,
) -> list[dict[str, Any]]:
    """Convert documents and their sections into a clean hierarchical tree for `streamlit-tree-select2`.

    Tree structure:
        📁 Folder (doc_count)
           └── 📄 Document (section_count, token_count)
                └── 📑 [H1] Section Title
                     └── 📑 [H2] Subsection Title
    """
    sections_by_doc = sections_by_doc or {}
    deduped_docs = dedupe_documents(documents)
    folder_map = {f["folder_id"]: (f.get("name") or f["folder_id"]) for f in (folders or []) if f.get("folder_id")}

    # Group documents by their parent folder path / id
    docs_by_folder: dict[str, list[dict[str, Any]]] = {}
    for doc in deduped_docs:
        f_path = _folder_of(doc)
        docs_by_folder.setdefault(f_path, []).append(doc)

    def _build_section_nodes(toc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Recursively build nested section nodes from TOC rows."""
        by_parent: dict[str | None, list[dict[str, Any]]] = {}
        for r in toc_rows:
            by_parent.setdefault(r.get("parent_section_id"), []).append(r)

        def _build_subtree(parent_id: str | None) -> list[dict[str, Any]]:
            children = by_parent.get(parent_id, [])
            nodes: list[dict[str, Any]] = []
            for sec in sorted(children, key=lambda s: s.get("sequence", 0)):
                lvl = sec.get("level", 1)
                if max_section_level is not None and lvl > max_section_level:
                    continue

                sid = sec["section_id"]
                title = sec.get("title") or "Untitled"
                tok = sec.get("token_count", 0)

                # Icon indicators in tree label
                indicators = []
                if sec.get("has_table"):
                    indicators.append("📊")
                if sec.get("has_image"):
                    indicators.append("🖼️")
                if sec.get("has_graph"):
                    indicators.append("🕸️")
                ind_str = (" " + "".join(indicators)) if indicators else ""

                lvl_tag = f"H{lvl}" if lvl > 0 else "Doc"
                label = f"📑 [{lvl_tag}] {title}{ind_str} ({tok:,} tok)"

                node: dict[str, Any] = {
                    "label": label,
                    "value": f"sec:{sid}",
                }
                sub_children = _build_subtree(sid)
                if sub_children:
                    node["children"] = sub_children
                nodes.append(node)
            return nodes

        roots = by_parent.get(None, [])
        if len(roots) == 1 and roots[0].get("level") == 0 and by_parent.get(roots[0].get("section_id")):
            return _build_subtree(roots[0].get("section_id"))
        return _build_subtree(None)

    def _build_doc_node(doc: dict[str, Any]) -> dict[str, Any]:
        """Build a tree node for a single document."""
        md_hash = doc.get("markdown_hash") or doc.get("content_hash") or ""
        fname = doc.get("filename") or "Untitled Document"
        sec_count = doc.get("section_count", 0)
        tok_count = doc.get("token_count", 0)

        label = f"📄 {fname} ({sec_count} sec, {tok_count:,} tok)"
        doc_node: dict[str, Any] = {
            "label": label,
            "value": f"doc:{md_hash}",
        }

        if include_sections and md_hash in sections_by_doc:
            sec_nodes = _build_section_nodes(sections_by_doc[md_hash])
            if sec_nodes:
                doc_node["children"] = sec_nodes

        return doc_node

    # If all documents are in root ('.')
    if len(docs_by_folder) == 1 and "." in docs_by_folder:
        return [_build_doc_node(doc) for doc in docs_by_folder["."]]

    # If all documents are in one single folder
    if len(docs_by_folder) == 1:
        f_path, f_docs = next(iter(docs_by_folder.items()))
        f_name = folder_map.get(f_path) or PurePosixPath(f_path).name or f_path
        return [
            {
                "label": f"📁 {f_name} ({len(f_docs)} docs)",
                "value": f"folder:{f_path}",
                "children": [_build_doc_node(doc) for doc in f_docs],
            }
        ]

    # Multiple distinct folders
    tree_nodes: list[dict[str, Any]] = []
    for f_path in sorted(docs_by_folder.keys()):
        f_docs = docs_by_folder[f_path]
        f_name = folder_map.get(f_path) or (PurePosixPath(f_path).name if f_path != "." else "(root)")
        tree_nodes.append(
            {
                "label": f"📁 {f_name} ({len(f_docs)} docs)",
                "value": f"folder:{f_path}",
                "children": [_build_doc_node(doc) for doc in f_docs],
            }
        )
    return tree_nodes


# ---------------------------------------------------------------------------
# Section Expander Label Formatter
# ---------------------------------------------------------------------------


def format_section_expander_label(section: dict[str, Any]) -> str:
    """Format a compact, descriptive label for a section's `st.expander`.

    Example:
        `📑 [H2] Setup Guide — Step-by-step instructions · 📊 🖼️ · 420 tok (L15-L52) · 🔑 hash::2`
    """
    level = section.get("level", 1)
    lvl_tag = f"[H{level}]" if level > 0 else "[Doc]"
    title = (section.get("title") or "Untitled Section").strip()
    sid = section.get("section_id") or ""

    summary = (section.get("summary") or section.get("description") or "").strip()
    if summary:
        summary_clean = re.sub(r"\s+", " ", summary)
        if len(summary_clean) > 80:
            summary_preview = f" — {summary_clean[:77]}..."
        else:
            summary_preview = f" — {summary_clean}"
    else:
        summary_preview = ""

    icons: list[str] = []
    if section.get("has_table") or section.get("tables"):
        icons.append("📊")
    if section.get("has_image") or section.get("images"):
        icons.append("🖼️")
    if section.get("has_graph") or section.get("graph_links"):
        icons.append("🕸️")

    icon_str = f" {' '.join(icons)}" if icons else ""

    tokens = section.get("token_count", 0)
    l_start = section.get("line_start")
    l_end = section.get("line_end")

    if l_start and l_end:
        line_info = f"L{l_start}-L{l_end}"
    elif l_start:
        line_info = f"L{l_start}"
    else:
        line_info = None

    token_str = f"{tokens:,} tok"
    length_str = f"{token_str} ({line_info})" if line_info else token_str
    id_str = f" · 🔑 {sid}" if sid else ""

    return f"📑 {lvl_tag} {title}{summary_preview}{icon_str} · {length_str}{id_str}"

    # Length & Line numbers
    tokens = section.get("token_count", 0)
    l_start = section.get("line_start")
    l_end = section.get("line_end")

    if l_start and l_end:
        line_info = f"L{l_start}-L{l_end}"
    elif l_start:
        line_info = f"L{l_start}"
    else:
        line_info = None

    token_str = f"{tokens:,} tokens"
    length_str = f"{token_str} ({line_info})" if line_info else token_str

    return f"📑 {lvl_tag} {title}{summary_preview}{icon_str} · {length_str}"


# ---------------------------------------------------------------------------
# Image Resolution and Extraction
# ---------------------------------------------------------------------------


def resolve_image_path(
    path: str | None,
    filename: str | None,
    doc_path: str | None = None,
    db_path: str | None = None,
) -> str | None:
    """Resolve an image reference (path or filename) to a usable file path or URL.

    Checks:
    1. HTTP/HTTPS/data URLs
    2. Absolute existing paths
    3. Relative to document path / document directory
    4. Relative to markdown output folder `<db_path>_markdown/`
    5. Relative to CWD / project root / data root
    """
    if path and path.startswith(("http://", "https://", "data:")):
        return path

    candidates: list[Path] = []

    # 1. Direct path
    if path:
        p = Path(path).expanduser()
        if p.is_absolute() and p.is_file():
            return str(p)
        candidates.append(p)
        candidates.append(Path.cwd() / p)

    # 2. Filename
    if filename:
        fn = Path(filename).name
        candidates.append(Path(fn))
        candidates.append(Path.cwd() / fn)

        # Relative to doc_path
        if doc_path:
            doc_dir = Path(doc_path).parent
            candidates.append(doc_dir / fn)
            candidates.append(doc_dir / "images" / fn)
            candidates.append(doc_dir / "media" / fn)

        # Relative to db_path markdown folder
        if db_path:
            db_dir = Path(db_path).parent
            db_stem = Path(db_path).stem
            md_dir = db_dir / f"{db_stem}_markdown"
            candidates.append(md_dir / fn)
            candidates.append(md_dir / "images" / fn)

            # Look in subdirectories of md_dir
            if md_dir.is_dir():
                for sub in md_dir.glob("**/images"):
                    candidates.append(sub / fn)

        # Check standard data roots
        home_data = Path.home() / "data"
        if home_data.is_dir():
            candidates.append(home_data / fn)

    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand.resolve())
        except (OSError, PermissionError):
            continue

    return path if (path and Path(path).exists()) else None


def extract_markdown_images(text: str) -> list[dict[str, str]]:
    """Extract markdown image links and HTML img tags from text."""
    results: list[dict[str, str]] = []

    # Markdown images: ![alt](url "title")
    for m in _MD_IMAGE_RE.finditer(text):
        results.append(
            {
                "alt": m.group("alt") or "",
                "url": m.group("url") or "",
                "title": m.group("title") or "",
            }
        )

    # Mistral comments: <!-- Image: filename.png (hash: 1234) -->
    for m in _MISTRAL_IMG_RE.finditer(text):
        results.append(
            {
                "alt": m.group("filename") or "",
                "url": m.group("filename") or "",
                "title": f"Hash: {m.group('hash')}",
            }
        )

    # HTML img tags: <img src="..." />
    for m in _HTML_IMG_RE.finditer(text):
        results.append(
            {
                "alt": "",
                "url": m.group("url") or "",
                "title": "",
            }
        )

    return results


# ---------------------------------------------------------------------------
# Section Content & Banner Renderers
# ---------------------------------------------------------------------------


def render_section_banner(
    section: dict[str, Any],
    doc_name: str | None = None,
    score: float | None = None,
    show_summary: bool = True,
) -> None:
    """Render a compact, informative banner for a section with Section ID prominently displayed."""
    level = section.get("level", 1)
    lvl_tag = f"H{level}" if level > 0 else "Doc"
    title = (section.get("title") or "Untitled Section").strip()
    sid = section.get("section_id") or "-"
    seq = section.get("sequence", 0)
    tok = section.get("token_count", 0)
    l_start = section.get("line_start")
    l_end = section.get("line_end")
    line_str = f"L{l_start}-L{l_end}" if (l_start and l_end) else (f"L{l_start}" if l_start else "-")
    summary = section.get("summary") or section.get("description")
    summary_source = section.get("summary_source")

    score_badge = f" · 🎯 Score: <strong>{score:.3f}</strong>" if score is not None else ""
    doc_badge = f"📄 Doc: <strong>{doc_name}</strong> · " if doc_name else ""

    st.markdown(
        f"""
        <div style="background: #f0f4f8; border-left: 5px solid #00005B; padding: 10px 14px; border-radius: 4px; margin-bottom: 10px;">
            <div style="font-size: 1.15em; font-weight: bold; color: #00005B;">
                📑 [{lvl_tag}] #{seq}: {title}
            </div>
            <div style="font-size: 0.88em; color: #444; margin-top: 4px;">
                {doc_badge}🔑 <strong>Section ID:</strong> <code style="color: #0073E6; font-weight: bold;">{sid}</code>
                · 🔤 <strong>{tok:,}</strong> tokens · 📏 <strong>{line_str}</strong>{score_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_summary and summary:
        source_badge = f" *(source: `{summary_source}`)*" if summary_source else ""
        st.markdown(
            f"""
            <div style="background-color: #f7fbff; border-left: 4px solid #0073E6; padding: 8px 12px; border-radius: 4px; margin-bottom: 10px;">
                <strong style="color: #00005B;">💡 Summary{source_badge}:</strong>
                <span style="color: #222;">{summary}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_document_banner(doc_data: dict[str, Any]) -> None:
    """Render a compact header card for a document."""
    fname = doc_data.get("filename") or "Document"
    m_hash = doc_data.get("markdown_hash") or doc_data.get("content_hash") or "-"
    p = doc_data.get("path") or ""
    origin_src = read_origin_path(p) if p else None
    sec_count = doc_data.get("section_count", 0)
    tok_count = doc_data.get("token_count", 0)
    lang = str(doc_data.get("language") or "en").upper()
    doc_abstract = doc_data.get("summary") or doc_data.get("description")

    st.markdown(
        f"""
        <div style="background: #eef3f8; border-left: 5px solid #0073E6; padding: 12px 16px; border-radius: 4px; margin-bottom: 10px;">
            <div style="font-size: 1.25em; font-weight: bold; color: #00005B;">
                📄 {fname}
            </div>
            <div style="font-size: 0.88em; color: #444; margin-top: 4px;">
                🔑 <strong>Hash:</strong> <code>{m_hash[:12]}...</code> · 📑 <strong>{sec_count}</strong> sections · 🔤 <strong>{tok_count:,}</strong> tokens · 🌐 <strong>{lang}</strong>
                {(" · 📁 <code>" + p + "</code>") if p else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if origin_src:
        st.caption(f"**Original Source Document:** `{origin_src}`")

    if doc_abstract:
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border-left: 4px solid #00005B; padding: 8px 12px; border-radius: 4px; margin-bottom: 12px;">
                <strong style="color: #00005B;">📋 Abstract:</strong>
                <span style="color: #333;">{doc_abstract}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_folder_banner(folder_data: dict[str, Any]) -> None:
    """Render a compact header card for a folder."""
    name = folder_data.get("name") or folder_data.get("folder_id") or "Folder"
    fid = folder_data.get("folder_id") or "-"
    doc_count = folder_data.get("doc_count", 0)

    st.markdown(
        f"""
        <div style="background: #eef3f8; border-left: 5px solid #3d5a80; padding: 12px 16px; border-radius: 4px; margin-bottom: 10px;">
            <div style="font-size: 1.25em; font-weight: bold; color: #1b3a4b;">
                📁 {name}
            </div>
            <div style="font-size: 0.88em; color: #444; margin-top: 4px;">
                🔑 <strong>Path / ID:</strong> <code>{fid}</code> · 📄 <strong>{doc_count}</strong> document{"s" if doc_count != 1 else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_content_view(
    section: dict[str, Any],
    doc_path: str | None = None,
    doc_name: str | None = None,
    db_path: str | None = None,
    show_banner: bool = True,
) -> None:
    """Render the full contents of a section inside an open `st.expander` or standalone view."""
    summary = section.get("summary")
    description = section.get("description")
    summary_source = section.get("summary_source")
    keywords = section.get("keywords") or []
    sec_text = section.get("text") or ""
    tables = section.get("tables") or []
    images = section.get("images") or []
    graph_links = section.get("graph_links") or []

    # 1. Summary Box / Banner
    if show_banner:
        render_section_banner(section, doc_name=doc_name, show_summary=True)
    else:
        if summary or description:
            with st.container():
                if summary:
                    source_badge = f" *(source: `{summary_source}`)*" if summary_source else ""
                    st.markdown(
                        f"""
                        <div style="background-color: #f0f7ff; border-left: 4px solid #0073E6; padding: 8px 12px; border-radius: 4px; margin-bottom: 10px;">
                            <strong style="color: #00005B;">💡 Summary{source_badge}:</strong>
                            <span style="color: #1a1a1a;">{summary}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif description:
                    st.markdown(
                        f"""
                        <div style="background-color: #f6f8fa; border-left: 4px solid #6c757d; padding: 6px 10px; border-radius: 4px; margin-bottom: 10px;">
                            <strong>📌 Overview:</strong> {description}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # 2. Metadata line
        sid = section.get("section_id", "-")
        seq = section.get("sequence", 0)
        lvl = section.get("level", 1)
        tok = section.get("token_count", 0)
        kw_str = f" · Keywords: {', '.join(f'`{k}`' for k in keywords[:4])}" if keywords else ""
        st.caption(f"🔑 ID: `{sid}` · #{seq} (H{lvl}) · {tok:,} tok{kw_str}")
        st.divider()

    # 3. Main Section Markdown Text
    if sec_text.strip():
        st.markdown(sec_text, unsafe_allow_html=True)
    else:
        st.info("*(Section body is empty or contains only sub-headings)*")

    # 4. Images Display (explicit node images + parsed markdown images)
    all_img_refs = list(images)
    if not all_img_refs:
        parsed_imgs = extract_markdown_images(sec_text)
        for pi in parsed_imgs:
            all_img_refs.append(
                {
                    "name": pi["alt"] or pi["url"],
                    "filename": pi["url"],
                    "path": pi["url"],
                    "description": pi["title"] or pi["alt"],
                }
            )

    if all_img_refs:
        st.markdown("##### 🖼️ Images in this Section")
        img_cols = st.columns(min(len(all_img_refs), 2))
        for idx, img in enumerate(all_img_refs):
            col = img_cols[idx % len(img_cols)]
            with col:
                img_path = img.get("path") or img.get("filename")
                img_name = img.get("name") or img.get("filename") or f"Image {idx + 1}"
                img_desc = img.get("description") or ""

                resolved = resolve_image_path(img_path, img.get("filename"), doc_path=doc_path, db_path=db_path)
                if resolved:
                    try:
                        st.image(
                            resolved,
                            caption=f"{img_name}: {img_desc}" if img_desc else img_name,
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.warning(f"Could not render image `{img_path}`: {exc}")
                else:
                    st.info(f"🖼️ **{img_name}**\n\n*Caption:* {img_desc or '(No caption)'}\n\n*Reference:* `{img_path}`")

    # 5. Tables Display
    if tables:
        st.markdown("##### 📊 Structured Tables in this Section")
        for idx, tbl in enumerate(tables):
            tbl_name = tbl.get("name") or f"Table {idx + 1}"
            tbl_caption = tbl.get("caption") or ""
            tbl_format = tbl.get("table_format") or "markdown"
            tbl_content = tbl.get("content") or ""

            with st.expander(
                f"📊 {tbl_name} ({tbl_format.upper()}) {('— ' + tbl_caption) if tbl_caption else ''}", expanded=True
            ):
                if tbl_caption:
                    st.caption(f"**Caption:** {tbl_caption}")
                if tbl_format == "html":
                    st.markdown(tbl_content, unsafe_allow_html=True)
                else:
                    st.markdown(tbl_content)

                with st.expander("Show raw table markup", expanded=False):
                    st.code(tbl_content, language="html" if tbl_format == "html" else "markdown")

    # 6. Graph Links / Knowledge Graph Entity Mentions
    if graph_links:
        st.markdown("##### 🕸️ Connected Knowledge Graph Entities")
        for gl in graph_links:
            rel = gl.get("rel_type") or "RELATED_TO"
            labels = gl.get("entity_labels") or []
            label_str = f":{':'.join(labels)}" if labels else ""
            ename = gl.get("entity_name") or "Entity"
            st.markdown(f"- **`-{rel}->`** `{ename}` `({label_str})`")

    # 7. Raw Markdown Source Expander
    if sec_text.strip():
        with st.expander("📄 View Raw Markdown Source", expanded=False):
            st.code(sec_text, language="markdown")
