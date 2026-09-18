"""Streamlit page for interactive Document Graph exploration.

Provides a rich visual interface to browse, inspect, search, and navigate
Ladybug Document Graphs:
- Hierarchical tree navigation with `streamlit-tree-select2` (Folders → Docs → Sections)
- Section expanders featuring summaries, token lengths, and Table/Image/Graph indicators
- Rendered inline images with intelligent local/remote path resolution
- Rendered tables and connected Knowledge Graph entity links
- Cross-document Hybrid / Vector / BM25 / Cypher search
- Corpus-wide Image Gallery and Tables Inspector
- Full document Markdown reconstruction and download

Usage:
    cli docgraph web
    cli docbench web
    uv run streamlit run genai_graph/webapp/pages/demos/docgraph_browser.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger
from streamlit_tree_select import tree_select

from genai_graph.agent.docgraph_agent import find_docgraph_db_path
from genai_graph.kg.backend import KgBackend
from genai_graph.kg.query.document_graph_tools import reconstruct_document
from genai_graph.kg.query.document_graph_tui import _read_origin_path
from genai_graph.webapp.ui_components.docgraph_view import (
    build_tree_select_nodes,
    fetch_all_images_async,
    fetch_all_tables_async,
    fetch_database_stats,
    fetch_document_sections_full,
    fetch_folders_and_documents,
    format_section_expander_label,
    get_docgraph_backend,
    render_section_content_view,
    resolve_image_path,
    search_sections_async,
)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialize page session state variables."""
    if "docgraph_profile" not in st.session_state:
        st.session_state.docgraph_profile = os.getenv("DOCGRAPH_PROFILE", "default")
    if "docgraph_db_path" not in st.session_state:
        st.session_state.docgraph_db_path = os.getenv("DOCGRAPH_DB_PATH", "")
    if "selected_doc_hash" not in st.session_state:
        st.session_state.selected_doc_hash = None
    if "selected_section_id" not in st.session_state:
        st.session_state.selected_section_id = None
    if "expand_all_sections" not in st.session_state:
        st.session_state.expand_all_sections = False


def _get_configured_profiles() -> list[str]:
    """Discover available DocGraph profiles from global configuration."""
    profiles = ["default"]
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        cfg = global_config()
        p_dict = cfg.get("docgraph_profiles", None)
        if isinstance(p_dict, dict):
            profiles = list(p_dict.keys())
        elif hasattr(p_dict, "keys"):
            profiles = list(p_dict.keys())
    except Exception as exc:
        logger.debug("Could not load docgraph_profiles from config: {}", exc)
    if "default" not in profiles:
        profiles.insert(0, "default")
    return sorted(set(profiles))


# ---------------------------------------------------------------------------
# Main Application Component (Async Streamlit)
# ---------------------------------------------------------------------------


async def render_docgraph_explorer() -> None:
    """Main entry point for the Document Graph Explorer."""
    _init_session_state()

    st.set_page_config(
        page_title="Document Graph Explorer",
        page_icon="📑",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📑 Document Graph Explorer")
    st.caption("Interactive browser, search, and hierarchy navigator for Ladybug Document Graphs")

    # -----------------------------------------------------------------------
    # Sidebar: Profile / Database Selector & Tree Navigation
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("🗄️ Database Connection")

        profiles = _get_configured_profiles()
        current_prof = st.session_state.docgraph_profile
        prof_index = profiles.index(current_prof) if current_prof in profiles else 0

        selected_profile = st.selectbox(
            "DocGraph Profile",
            options=profiles,
            index=prof_index,
            help="Select a profile from config/docgraph.yaml",
        )

        # Resolve DB path for chosen profile if not manually set
        resolved_db = find_docgraph_db_path(selected_profile) or st.session_state.docgraph_db_path

        db_path_input = st.text_input(
            "Database Path (.lbdb / .db)",
            value=st.session_state.docgraph_db_path or resolved_db or "",
            placeholder="/path/to/docgraph.db",
            help="Path to the Ladybug / Kuzu database directory",
        )

        st.session_state.docgraph_profile = selected_profile
        st.session_state.docgraph_db_path = db_path_input.strip()

        col_ref, col_stat = st.columns([1, 1])
        with col_ref:
            if st.button("🔄 Reconnect", use_container_width=True):
                st.session_state.selected_doc_hash = None
                st.session_state.selected_section_id = None
                st.rerun()

        active_db_path = st.session_state.docgraph_db_path
        if not active_db_path:
            st.warning("⚠️ Please provide or configure a valid database path.")
            st.stop()

        # Connect to Ladybug DB
        backend: KgBackend | None = None
        try:
            backend = get_docgraph_backend(active_db_path)
            stats = await fetch_database_stats(backend)
            with col_stat:
                st.success("🟢 Connected")
        except Exception as exc:
            st.error(f"❌ Connection failed: {exc}")
            st.info("Ensure the database exists or build one with `cli docgraph build <sources>`.")
            st.stop()

        st.divider()

        # Sidebar Search & Tree Navigation
        st.header("📂 Document Hierarchy")
        st.caption("Navigate folders, documents, and sections via `streamlit-tree-select2`")

        folders, documents = await fetch_folders_and_documents(backend)

        if not documents:
            st.warning("No documents found in this database.")
            st.stop()

        # Build tree nodes
        # Pre-fetch TOC for all documents to build deep tree if reasonable count
        sections_by_doc: dict[str, list[dict[str, Any]]] = {}
        if len(documents) <= 50:
            for d in documents:
                m_hash = d.get("markdown_hash")
                if m_hash:
                    from genai_graph.kg.query.document_graph_tools import get_document_toc

                    sections_by_doc[m_hash] = get_document_toc(backend, m_hash)  # type: ignore[assignment]

        tree_nodes = build_tree_select_nodes(
            folders=folders,
            documents=documents,
            sections_by_doc=sections_by_doc,
            include_sections=True,
        )

        # Tree Select widget
        tree_result = tree_select(
            nodes=tree_nodes,
            checked=None,
            expanded=None,
            expand_on_click=True,
            show_expand_all=True,
            key="docgraph_tree_select",
        )

        # Process tree selection
        if tree_result and tree_result.get("checked"):
            checked_items = tree_result["checked"]
            # Prioritize the most specific checked item (section > doc > folder)
            for val in reversed(checked_items):
                if val.startswith("sec:"):
                    sid = val.split("sec:", 1)[1]
                    st.session_state.selected_section_id = sid
                    # Extract document hash from section_id (format: {hash}::{seq})
                    doc_hash = sid.split("::", 1)[0]
                    st.session_state.selected_doc_hash = doc_hash
                    break
                elif val.startswith("doc:"):
                    doc_hash = val.split("doc:", 1)[1]
                    st.session_state.selected_doc_hash = doc_hash
                    st.session_state.selected_section_id = None
                    break

        st.divider()
        st.markdown("### 🎛️ Section Display Filters")
        filter_text = st.text_input("Filter section title/text", "", placeholder="e.g. revenue, setup...")
        filter_level = st.selectbox("Max Heading Level", ["All Levels", "H1 Only", "Up to H2", "Up to H3", "Up to H4"])
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            only_tables = st.checkbox("Only Tables 📊", value=False)
            only_summaries = st.checkbox("Only Summarized 📝", value=False)
        with col_f2:
            only_images = st.checkbox("Only Images 🖼️", value=False)
            only_graph = st.checkbox("Only Graph Links 🕸️", value=False)

    # -----------------------------------------------------------------------
    # Main Dashboard Header: Metrics
    # -----------------------------------------------------------------------
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("📁 Folders", f"{stats['folders']:,}")
    m_col2.metric("📄 Documents", f"{stats['documents']:,}")
    m_col3.metric("📑 Sections", f"{stats['sections']:,}")
    m_col4.metric("🖼️ Images", f"{stats['images']:,}")
    m_col5.metric("📊 Tables", f"{stats['tables']:,}")
    m_col6.metric("🔤 Total Tokens", f"{stats['total_tokens']:,}")

    st.divider()

    # -----------------------------------------------------------------------
    # Navigation Tabs
    # -----------------------------------------------------------------------
    tab_doc, tab_search, tab_images, tab_tables, tab_graph, tab_raw = st.tabs(
        [
            "📑 Document Sections",
            "🔍 Cross-Document Search",
            "🖼️ Image Gallery",
            "📊 Tables Inspector",
            "🕸️ Graph & Schema",
            "📄 Full Markdown",
        ]
    )

    # -----------------------------------------------------------------------
    # Tab 1: Document Sections Explorer (Core Feature)
    # -----------------------------------------------------------------------
    with tab_doc:
        doc_options: dict[str, dict[str, Any]] = {}
        for d in documents:
            m_hash = d.get("markdown_hash") or d.get("content_hash") or ""
            fname = d.get("filename") or m_hash
            p = d.get("path") or ""
            parent_dir = str(Path(p).parent) if p else "."
            label = f"{fname} ({parent_dir}) — {d.get('section_count', 0)} sections"
            doc_options[m_hash] = {"label": label, "data": d}

        # Select active document
        doc_hashes = list(doc_options.keys())
        current_doc_hash = st.session_state.selected_doc_hash
        default_idx = doc_hashes.index(current_doc_hash) if current_doc_hash in doc_hashes else 0

        selected_hash = st.selectbox(
            "Select Document to Explore",
            options=doc_hashes,
            index=default_idx,
            format_func=lambda h: doc_options[h]["label"],
            key="doc_select_box",
        )

        st.session_state.selected_doc_hash = selected_hash
        active_doc_data = doc_options[selected_hash]["data"]

        # Fetch sections for selected document
        doc_sections = await fetch_document_sections_full(backend, selected_hash)

        # Document Header Card
        with st.container():
            dh_cols = st.columns([3, 1, 1, 1])
            with dh_cols[0]:
                st.subheader(f"📄 {active_doc_data.get('filename')}")
                origin_src = _read_origin_path(active_doc_data.get("path"))
                if origin_src:
                    st.caption(f"**Original Source:** `{origin_src}`")
                elif active_doc_data.get("path"):
                    st.caption(f"**Path:** `{active_doc_data.get('path')}`")
            with dh_cols[1]:
                st.metric("Sections", f"{len(doc_sections)}")
            with dh_cols[2]:
                st.metric("Tokens", f"{active_doc_data.get('token_count', 0):,}")
            with dh_cols[3]:
                st.metric("Language", str(active_doc_data.get("language") or "en").upper())

            # Document Abstract / Summary Callout
            doc_abstract = active_doc_data.get("summary") or active_doc_data.get("description")
            if doc_abstract:
                st.markdown(
                    f"""
                    <div style="background-color: #f4f6f8; border-left: 4px solid #00005B; padding: 10px 14px; border-radius: 4px; margin-bottom: 16px;">
                        <strong style="color: #00005B;">📋 Document Abstract:</strong><br/>
                        <span style="color: #2b2b2b;">{doc_abstract}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # Section Expanders Section
        st.markdown(f"### Sections ({len(doc_sections)})")

        # Expand / Collapse toggle
        exp_col1, exp_col2, _ = st.columns([1, 1, 4])
        with exp_col1:
            if st.button("Expand All", use_container_width=True):
                st.session_state.expand_all_sections = True
                st.rerun()
        with exp_col2:
            if st.button("Collapse All", use_container_width=True):
                st.session_state.expand_all_sections = False
                st.rerun()

        # Filter sections according to sidebar criteria
        filtered_sections = []
        max_lvl_map = {"All Levels": 10, "H1 Only": 1, "Up to H2": 2, "Up to H3": 3, "Up to H4": 4}
        max_allowed_lvl = max_lvl_map.get(filter_level, 10)

        for sec in doc_sections:
            lvl = sec.get("level", 1)
            if lvl > max_allowed_lvl and lvl > 0:
                continue
            if only_tables and not sec.get("has_table"):
                continue
            if only_images and not sec.get("has_image"):
                continue
            if only_graph and not sec.get("has_graph"):
                continue
            if only_summaries and not sec.get("summary"):
                continue
            if filter_text:
                q_low = filter_text.lower()
                sec_title = (sec.get("title") or "").lower()
                sec_body = (sec.get("text") or "").lower()
                sec_sum = (sec.get("summary") or "").lower()
                if q_low not in sec_title and q_low not in sec_body and q_low not in sec_sum:
                    continue
            filtered_sections.append(sec)

        if not filtered_sections:
            st.info("No sections match the current filter criteria.")
        else:
            if len(filtered_sections) < len(doc_sections):
                st.caption(f"Showing {len(filtered_sections)} of {len(doc_sections)} sections (filtered)")

            for sec in filtered_sections:
                sid = sec["section_id"]
                label = format_section_expander_label(sec)

                # Determine whether this expander should be open
                is_selected_sec = st.session_state.selected_section_id == sid
                is_expanded = st.session_state.expand_all_sections or is_selected_sec

                with st.expander(label, expanded=is_expanded):
                    render_section_content_view(
                        sec,
                        doc_path=active_doc_data.get("path"),
                        db_path=active_db_path,
                    )

    # -----------------------------------------------------------------------
    # Tab 2: Cross-Document Search
    # -----------------------------------------------------------------------
    with tab_search:
        st.subheader("🔍 Search Sections Across Documents")
        s_col1, s_col2, s_col3 = st.columns([3, 1, 1])
        with s_col1:
            search_query = st.text_input(
                "Search query / keywords", placeholder="e.g. cloud migration strategy", key="search_query_input"
            )
        with s_col2:
            search_mode = st.selectbox("Search Mode", ["hybrid", "vector", "bm25", "cypher"], index=0)
        with s_col3:
            search_limit = st.number_input("Max Results", min_value=1, max_value=100, value=20)

        if search_query:
            with st.spinner("Searching Document Graph..."):
                results = await search_sections_async(
                    backend,
                    search_query,
                    mode=search_mode,
                    limit=int(search_limit),
                )

            if not results:
                st.warning(f"No sections matched query: '{search_query}'")
            else:
                st.success(f"Found {len(results)} matching section(s)")
                for r in results:
                    score_val = r.get("score")
                    score_badge = f" `(Score: {score_val:.3f})`" if score_val else ""
                    sec_title = r.get("title") or "Untitled"
                    sid = r.get("section_id") or ""
                    m_hash = r.get("markdown_hash") or ""
                    desc = r.get("description") or ""

                    with st.expander(f"📑 {sec_title}{score_badge} — Section `{sid}`", expanded=False):
                        st.caption(f"**Document Hash:** `{m_hash}` | **Line:** {r.get('line_start', '-')}")
                        if desc:
                            st.markdown(f"**Description:** {desc}")
                        if r.get("matched_chunk"):
                            st.markdown("**Matched Excerpt:**")
                            st.code(r["matched_chunk"], language="markdown")

    # -----------------------------------------------------------------------
    # Tab 3: Image Gallery
    # -----------------------------------------------------------------------
    with tab_images:
        st.subheader("🖼️ Extracted Images Gallery")
        img_q_col, img_doc_col = st.columns([2, 2])
        with img_q_col:
            img_query = st.text_input(
                "Search image captions/filenames", "", placeholder="e.g. architecture, revenue chart..."
            )
        with img_doc_col:
            doc_filter_choices = ["All Documents"] + [d.get("filename") or d["markdown_hash"] for d in documents]
            chosen_doc = st.selectbox("Filter by Document", doc_filter_choices)

        doc_filter_hash = None
        if chosen_doc != "All Documents":
            for d in documents:
                if (d.get("filename") == chosen_doc) or (d.get("markdown_hash") == chosen_doc):
                    doc_filter_hash = d.get("markdown_hash")
                    break

        all_imgs = await fetch_all_images_async(backend, query=img_query or None, document_id=doc_filter_hash, limit=60)

        if not all_imgs:
            st.info("No images found in the database matching criteria.")
        else:
            st.caption(f"Showing {len(all_imgs)} image(s)")
            grid_cols = st.columns(3)
            for idx, img in enumerate(all_imgs):
                col = grid_cols[idx % 3]
                with col:
                    img_path = img.get("path") or img.get("filename")
                    img_name = img.get("name") or img.get("filename") or f"Image {idx + 1}"
                    img_desc = img.get("description") or "(No caption)"
                    sec_title = img.get("section_title") or img.get("section_id") or ""
                    doc_name = img.get("document_name") or ""

                    resolved = resolve_image_path(img_path, img.get("filename"), db_path=active_db_path)
                    if resolved:
                        try:
                            st.image(resolved, caption=img_name, use_container_width=True)
                        except Exception:
                            st.warning(f"Could not load image `{img_path}`")
                    else:
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f0f0; border: 1px dashed #aaa; padding: 20px; text-align: center; border-radius: 4px;">
                                🖼️<br/><strong>{img_name}</strong>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**Caption:** {img_desc}")
                    st.caption(f"**Document:** {doc_name} | **Section:** {sec_title}")
                    st.divider()

    # -----------------------------------------------------------------------
    # Tab 4: Tables Inspector
    # -----------------------------------------------------------------------
    with tab_tables:
        st.subheader("📊 Structured Tables Inspector")
        tbl_q_col, tbl_doc_col = st.columns([2, 2])
        with tbl_q_col:
            tbl_query = st.text_input("Search table captions/content", "", placeholder="e.g. balance sheet, metrics...")
        with tbl_doc_col:
            chosen_tbl_doc = st.selectbox("Filter Tables by Document", doc_filter_choices, key="tbl_doc_filter")

        tbl_doc_hash = None
        if chosen_tbl_doc != "All Documents":
            for d in documents:
                if (d.get("filename") == chosen_tbl_doc) or (d.get("markdown_hash") == chosen_tbl_doc):
                    tbl_doc_hash = d.get("markdown_hash")
                    break

        all_tbls = await fetch_all_tables_async(backend, query=tbl_query or None, document_id=tbl_doc_hash, limit=60)

        if not all_tbls:
            st.info("No tables found in the database matching criteria.")
        else:
            st.caption(f"Showing {len(all_tbls)} table(s)")
            for tbl in all_tbls:
                t_name = tbl.get("name") or "Table"
                t_caption = tbl.get("caption") or ""
                t_fmt = tbl.get("table_format") or "markdown"
                t_content = tbl.get("content") or ""
                t_doc = tbl.get("document_name") or ""
                t_sec = tbl.get("section_title") or ""

                with st.expander(f"📊 {t_name} — {t_caption or '(No caption)'} ({t_fmt.upper()})", expanded=False):
                    st.caption(
                        f"**Document:** {t_doc} | **Section:** {t_sec} | **Tokens:** {tbl.get('token_count', 0):,}"
                    )
                    if t_fmt == "html":
                        st.markdown(t_content, unsafe_allow_html=True)
                    else:
                        st.markdown(t_content)
                    with st.expander("Raw Table Source", expanded=False):
                        st.code(t_content, language="html" if t_fmt == "html" else "markdown")

    # -----------------------------------------------------------------------
    # Tab 5: Graph & Schema Inspector
    # -----------------------------------------------------------------------
    with tab_graph:
        st.subheader("🕸️ Document Graph Schema & Cypher Runner")
        st.caption("Inspect tables, relationships, and execute custom Cypher queries")

        try:
            tbl_df = backend.execute_get_as_df("CALL show_tables() RETURN *", union=False)
            st.dataframe(tbl_df, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not fetch table info: {exc}")

        st.markdown("#### Run Cypher Query")
        sample_query = (
            "MATCH (d:Document)-[:HAS_SECTION]->(s:MarkdownSection) RETURN d.filename, s.title, s.token_count LIMIT 10"
        )
        user_cypher = st.text_area("Cypher Query", value=sample_query, height=100)
        if st.button("Execute Cypher", use_container_width=False):
            try:
                res_df = backend.execute_get_as_df(user_cypher, union=False)
                if res_df is not None and not res_df.empty:
                    st.dataframe(res_df, use_container_width=True)
                else:
                    st.info("Query returned 0 rows.")
            except Exception as exc:
                st.error(f"Cypher Error: {exc}")

    # -----------------------------------------------------------------------
    # Tab 6: Full Document Markdown
    # -----------------------------------------------------------------------
    with tab_raw:
        st.subheader(f"📄 Full Reconstructed Markdown: {active_doc_data.get('filename')}")
        full_md_text = reconstruct_document(backend, selected_hash)
        if full_md_text:
            st.download_button(
                "⬇️ Download Markdown (.md)",
                data=full_md_text,
                file_name=f"{Path(active_doc_data.get('filename', 'doc')).stem}.md",
                mime="text/markdown",
            )
            st.code(full_md_text, language="markdown")
        else:
            st.warning("Could not reconstruct Markdown for this document.")


# ---------------------------------------------------------------------------
# Direct / Standalone Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__" or "streamlit" in __name__:
    # Execute async application
    import asyncio

    asyncio.run(render_docgraph_explorer())
