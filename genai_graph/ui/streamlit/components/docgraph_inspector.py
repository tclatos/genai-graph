"""Reusable Document Graph and Section Inspector component for Streamlit.

Provides a structured, interactive browser for long technical documents:
- Document selection with metadata and statistics
- Section selection (H1, H2, ... with Section ID, line range, and summary)
- Focused single-section inspection (renders Markdown, resolved figures, and tables)
- Full document view toggle
- Side-by-side split view or tabbed comparison with the original PDF via native ``st.pdf``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from genai_graph.kg.backend import KgBackend
from genai_graph.kg.query.document_graph_tools import (
    get_document_toc,
    list_documents,
    reconstruct_document,
)
from genai_graph.ui.streamlit.components.docgraph_view import (
    dedupe_documents,
    get_document_sections_full,
    read_origin_path,
    render_document_banner,
    render_section_content_view,
)

# ---------------------------------------------------------------------------
# PDF Resolution
# ---------------------------------------------------------------------------


def find_document_pdf(
    doc_filename: str,
    doc_path: str | None = None,
    additional_dirs: list[Path | str] | None = None,
) -> Path | None:
    """Resolve the original source PDF for a document in the graph.

    Args:
        doc_filename: Document filename stored in the graph (e.g. ``TO_00-25-172CL-1_pdf.md``).
        doc_path: Full file path recorded in the graph, if available.
        additional_dirs: Extra directories to search for matching PDFs.

    Returns:
        Resolved absolute Path to the PDF file, or None if not found.
    """
    # 1. Check embedded origin comment in the Markdown file: <!-- source: ... -->
    if doc_path and Path(doc_path).is_file():
        origin = read_origin_path(doc_path)
        if origin:
            p_orig = Path(origin).expanduser()
            if p_orig.is_file() and p_orig.suffix.lower() == ".pdf":
                return p_orig.resolve()

    # 2. Candidate PDF filenames
    stem = Path(doc_filename).stem
    candidate_names = [
        f"{stem.removesuffix('_pdf')}.pdf",
        f"{stem}.pdf",
    ]
    # Also handle .pdf if filename was already .pdf
    if doc_filename.lower().endswith(".pdf"):
        candidate_names.insert(0, Path(doc_filename).name)

    # 3. Candidate search directories
    search_dirs: list[Path] = []
    if doc_path:
        dp = Path(doc_path).resolve()
        search_dirs.extend(
            [
                dp.parent,
                dp.parent / "pdfs",
                dp.parent.parent / "pdfs",
                dp.parent.parent / "data" / "pdfs",
            ]
        )

    search_dirs.extend(
        [
            Path.cwd() / "data" / "pdfs",
            Path.cwd() / "pdfs",
            Path.cwd() / "data",
        ]
    )

    if additional_dirs:
        for ad in additional_dirs:
            search_dirs.append(Path(ad).resolve())

    # 4. Search candidate paths
    seen_dirs: set[Path] = set()
    for sdir in search_dirs:
        try:
            resolved_dir = sdir.resolve()
            if resolved_dir in seen_dirs or not resolved_dir.is_dir():
                continue
            seen_dirs.add(resolved_dir)

            for cname in candidate_names:
                target = resolved_dir / cname
                if target.is_file() and target.suffix.lower() == ".pdf":
                    return target.resolve()
        except (OSError, PermissionError):
            continue

    return None


def render_pdf_view(pdf_path: Path, height: int = 850) -> None:
    """Render a PDF file in Streamlit using st.pdf or base64 iframe fallback.

    Args:
        pdf_path: Path to the local PDF file.
        height: Display height in pixels.
    """
    try:
        st.pdf(str(pdf_path), height=height)
    except Exception:
        # Fallback for environments where st.pdf has component registration issues
        import base64

        try:
            pdf_bytes = pdf_path.read_bytes()
            encoded = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_html = (
                f'<iframe src="data:application/pdf;base64,{encoded}" '
                f'width="100%" height="{height}px" type="application/pdf"></iframe>'
            )
            st.markdown(pdf_html, unsafe_allow_html=True)
            st.download_button(
                f"⬇️ Download {pdf_path.name}",
                data=pdf_bytes,
                file_name=pdf_path.name,
                mime="application/pdf",
                key=f"dl_{pdf_path.stem}",
            )
        except Exception as exc:
            st.error(f"Could not render PDF: {exc}")


# ---------------------------------------------------------------------------
# Document & Section Inspector Component
# ---------------------------------------------------------------------------


def render_docgraph_inspector(
    backend: KgBackend,
    db_path: str,
    *,
    initial_doc: str | None = None,
    pdf_dirs: list[Path | str] | None = None,
    default_layout: str = "tabbed",  # "tabbed" | "split"
) -> None:
    """Render an interactive document and section explorer with PDF comparison.

    Args:
        backend: Connected KgBackend instance.
        db_path: Path to the active database.
        initial_doc: Optional filename or content hash to pre-select.
        pdf_dirs: Optional directories containing source PDF files.
        default_layout: Default layout style ("tabbed" or "split").
    """
    docs = dedupe_documents(list_documents(backend))
    if not docs:
        st.warning("No documents found in the Document Graph database.")
        return

    # Document labels
    doc_map: dict[str, dict[str, Any]] = {}
    for d in docs:
        label = f"📄 {d['filename']} ({d.get('section_count', 0)} sections · {d.get('token_count', 0):,} tok)"
        doc_map[label] = d

    labels = list(doc_map.keys())

    # Pre-selection index
    default_idx = 0
    if initial_doc:
        for i, d in enumerate(docs):
            if d.get("filename") == initial_doc or d.get("content_hash") == initial_doc:
                default_idx = i
                break

    # Top Control Bar
    col_doc, col_layout, col_view = st.columns([4, 2, 2])
    with col_doc:
        selected_label = st.selectbox(
            "Select Document",
            options=labels,
            index=default_idx,
            help="Choose a manual / document from the Document Graph",
        )
    selected_doc = doc_map[selected_label]
    doc_hash = selected_doc["content_hash"]
    doc_filename = selected_doc["filename"]
    doc_path = selected_doc.get("path")

    # Resolve PDF
    pdf_path = find_document_pdf(doc_filename, doc_path, additional_dirs=pdf_dirs)

    with col_layout:
        layout_mode = st.radio(
            "Comparison View",
            options=["Tabbed", "Split Screen"],
            index=0 if default_layout == "tabbed" else 1,
            horizontal=True,
            help="Switch between tabbed layout and side-by-side Markdown/PDF split screen",
        )

    with col_view:
        view_scope = st.radio(
            "Reading Scope",
            options=["Single Section", "Full Document"],
            index=0,
            horizontal=True,
            help="Single Section is fast and focused for long manuals; Full Document renders the entire text",
        )

    # Document summary banner
    render_document_banner(selected_doc)

    # Fetch document sections
    toc = get_document_toc(backend, doc_hash)
    if not toc:
        st.info("This document does not have any recorded sections.")
        return

    # Section selection for Single Section mode
    selected_section_row: dict[str, Any] | None = None
    if view_scope == "Single Section":
        # Format section options
        sec_options: list[str] = []
        sec_lookup: dict[str, dict[str, Any]] = {}
        for s in toc:
            lvl = s.get("level", 1)
            lvl_tag = f"H{lvl}" if lvl > 0 else "Doc"
            seq = s.get("sequence", 0)
            title = s.get("title") or "Untitled"
            tok = s.get("token_count", 0)
            sid = s.get("section_id", "")
            lbl = f"#{seq:03d} [{lvl_tag}] {title} ({tok:,} tok) — {sid}"
            sec_options.append(lbl)
            sec_lookup[lbl] = s

        col_sec, col_nav = st.columns([6, 2])
        with col_sec:
            sec_idx = st.selectbox(
                "Select Section to View",
                options=range(len(sec_options)),
                format_func=lambda i: sec_options[i],
                help="Navigate directly to any section or checklist",
            )
            selected_section_row = sec_lookup[sec_options[sec_idx]]

        with col_nav:
            st.metric("Total Sections", len(toc))

    # Fetch full section details if viewing a single section
    full_section: dict[str, Any] | None = None
    if selected_section_row:
        all_sections = get_document_sections_full(backend, doc_hash)
        target_sid = selected_section_row.get("section_id")
        full_section = next((s for s in all_sections if s.get("section_id") == target_sid), selected_section_row)

    # Render Content based on layout
    if layout_mode == "Split Screen":
        _render_split_screen(
            full_section=full_section,
            all_sections_func=lambda: get_document_sections_full(backend, doc_hash),
            doc_path=doc_path,
            doc_name=doc_filename,
            db_path=db_path,
            pdf_path=pdf_path,
            view_scope=view_scope,
            backend=backend,
            doc_hash=doc_hash,
        )
    else:
        _render_tabbed_view(
            full_section=full_section,
            all_sections_func=lambda: get_document_sections_full(backend, doc_hash),
            doc_path=doc_path,
            doc_name=doc_filename,
            db_path=db_path,
            pdf_path=pdf_path,
            view_scope=view_scope,
            toc=toc,
            backend=backend,
            doc_hash=doc_hash,
            doc_data=selected_doc,
        )


def _render_split_screen(
    full_section: dict[str, Any] | None,
    all_sections_func: Any,
    doc_path: str | None,
    doc_name: str,
    db_path: str,
    pdf_path: Path | None,
    view_scope: str,
    backend: KgBackend,
    doc_hash: str,
) -> None:
    """Render side-by-side Markdown content and source PDF."""
    col_md, col_pdf = st.columns([1, 1], gap="medium")

    with col_md:
        st.subheader("📑 Markdown & Figures (Graph)")
        if view_scope == "Single Section" and full_section:
            render_section_content_view(
                full_section,
                doc_path=doc_path,
                doc_name=doc_name,
                db_path=db_path,
                show_banner=True,
            )
        else:
            all_secs = all_sections_func()
            for sec in all_secs:
                render_section_content_view(
                    sec,
                    doc_path=doc_path,
                    doc_name=doc_name,
                    db_path=db_path,
                    show_banner=True,
                )
                st.divider()

    with col_pdf:
        st.subheader("📕 Original Source PDF")
        if pdf_path and pdf_path.is_file():
            st.caption(f"File: `{pdf_path.name}` · Size: {pdf_path.stat().st_size / (1024 * 1024):.1f} MB")
            render_pdf_view(pdf_path, height=850)
        else:
            st.info("Original PDF file was not found on local disk.")
            if doc_path:
                st.caption(f"Expected near: `{doc_path}`")


def _render_tabbed_view(
    full_section: dict[str, Any] | None,
    all_sections_func: Any,
    doc_path: str | None,
    doc_name: str,
    db_path: str,
    pdf_path: Path | None,
    view_scope: str,
    toc: list[dict[str, Any]],
    backend: KgBackend,
    doc_hash: str,
    doc_data: dict[str, Any],
) -> None:
    """Render tabbed layout for document details, PDF, TOC, and metadata."""
    tab_md, tab_pdf, tab_toc, tab_raw = st.tabs(
        [
            "📑 Markdown & Figures",
            "📕 Source PDF",
            "🗂️ Sections Outline",
            "📄 Full Markdown Reconstruct",
        ]
    )

    with tab_md:
        if view_scope == "Single Section" and full_section:
            render_section_content_view(
                full_section,
                doc_path=doc_path,
                doc_name=doc_name,
                db_path=db_path,
                show_banner=True,
            )
        else:
            all_secs = all_sections_func()
            st.caption(f"Rendering all {len(all_secs)} sections sequentially:")
            for sec in all_secs:
                with st.expander(
                    f"#{sec.get('sequence', 0)} [{sec.get('level', 1)}] {sec.get('title', 'Untitled')} ({sec.get('section_id', '')})",
                    expanded=False,
                ):
                    render_section_content_view(
                        sec,
                        doc_path=doc_path,
                        doc_name=doc_name,
                        db_path=db_path,
                        show_banner=False,
                    )

    with tab_pdf:
        if pdf_path and pdf_path.is_file():
            st.caption(f"📄 **PDF:** `{pdf_path.resolve()}` ({pdf_path.stat().st_size / (1024 * 1024):.1f} MB)")
            render_pdf_view(pdf_path, height=900)
        else:
            st.warning("⚠️ Original PDF file could not be located on disk.")
            st.write("Checked paths:")
            st.code(f"doc_filename: {doc_name}\ndoc_path: {doc_path}\nexpected in: data/pdfs/")

    with tab_toc:
        st.subheader("Document Table of Contents")
        toc_rows: list[dict[str, Any]] = []
        for s in toc:
            lvl = s.get("level", 1)
            indent = "    " * max(0, lvl - 1)
            toc_rows.append(
                {
                    "Seq": s.get("sequence", 0),
                    "Level": f"H{lvl}",
                    "Title": f"{indent}{s.get('title', '')}",
                    "Section ID": s.get("section_id", ""),
                    "Tokens": s.get("token_count", 0),
                    "Lines": f"L{s.get('line_start')}-L{s.get('line_end')}" if s.get("line_start") else "-",
                    "Summary": s.get("summary") or s.get("description") or "",
                }
            )
        st.dataframe(toc_rows, use_container_width=True, hide_index=True)

    with tab_raw:
        st.subheader("Reconstruct Full Document Markdown")
        if st.button("Generate Complete Reconstructed Markdown", key="btn_recon"):
            with st.spinner("Reconstructing markdown from graph sections..."):
                full_md = reconstruct_document(backend, doc_hash)
            st.download_button(
                "⬇️ Download Markdown",
                data=full_md,
                file_name=f"{Path(doc_name).stem}.md",
                mime="text/markdown",
            )
            st.text_area("Markdown text", value=full_md[:20000] + ("..." if len(full_md) > 20000 else ""), height=500)
