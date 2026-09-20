"""Streamlit page for interactive Document Graph exploration.

Provides a rich visual interface to browse, inspect, search, and navigate
Ladybug Document Graphs:
- Hierarchical tree navigation in the main window (Folders → Docs → Sections → Subsections)
- Clicking on a tree node displays full content of documents, sections, or sub-sections
- Section banner with Section ID prominently displayed
- Cross-document Hybrid / Vector / BM25 / Cypher search with rendered Markdown content
- Corpus-wide Image Gallery and Tables Inspector
- Full document Markdown reconstruction and download

Usage:
    cli docgraph web
    cli docbench web
    uv run streamlit run genai_graph/webapp/pages/demos/docgraph_browser.py
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

import streamlit as st
from loguru import logger
from streamlit_tree_select import tree_select

from genai_graph.agent.docgraph_agent import find_docgraph_db_path
from genai_graph.kg.backend import KgBackend
from genai_graph.kg.query.document_graph_tools import reconstruct_document
from genai_graph.webapp.ui_components.docgraph_view import (
    _folder_of,
    build_tree_select_nodes,
    dedupe_documents,
    fetch_all_images_async,
    fetch_all_tables_async,
    fetch_database_stats,
    fetch_document_sections_full,
    fetch_folders_and_documents,
    fetch_section_markdown_async,
    format_section_expander_label,
    get_docgraph_backend,
    render_document_banner,
    render_folder_banner,
    render_section_banner,
    render_section_content_view,
    resolve_image_path,
    search_sections_async,
)

# ---------------------------------------------------------------------------
# Session State & Profile Discovery
# ---------------------------------------------------------------------------


def _get_configured_profiles() -> list[str]:
    """Discover available DocGraph profiles from docgraph.yaml and global configuration."""
    profiles: list[str] = []

    # 1. Read config/docgraph.yaml
    try:
        from genai_graph.bench.config import list_docgraph_profiles

        profs = list_docgraph_profiles()
        if profs:
            profiles.extend(profs.keys())
    except Exception as exc:
        logger.debug("Could not read docgraph_profiles from docgraph.yaml: {}", exc)

    # 2. Read global_config()
    try:
        from genai_tk.config_mgmt.config_mngr import global_config

        cfg = global_config()
        p_dict = cfg.get("docgraph_profiles", None)
        if isinstance(p_dict, dict):
            profiles.extend(p_dict.keys())
        elif hasattr(p_dict, "keys"):
            profiles.extend(p_dict.keys())
    except Exception as exc:
        logger.debug("Could not load docgraph_profiles from global config: {}", exc)

    if not profiles:
        profiles = ["default"]
    elif "default" not in profiles:
        profiles.insert(0, "default")
    return sorted(set(profiles))


def _init_session_state() -> None:
    """Initialize page session state variables."""
    if "docgraph_profile" not in st.session_state:
        st.session_state.docgraph_profile = os.getenv("DOCGRAPH_PROFILE", "default")
    if "docgraph_db_path" not in st.session_state:
        initial_db = os.getenv("DOCGRAPH_DB_PATH") or find_docgraph_db_path(st.session_state.docgraph_profile) or ""
        st.session_state.docgraph_db_path = initial_db
    if "selected_tree_val" not in st.session_state:
        st.session_state.selected_tree_val = None
    if "selected_node_type" not in st.session_state:
        st.session_state.selected_node_type = None
    if "selected_doc_hash" not in st.session_state:
        st.session_state.selected_doc_hash = None
    if "selected_section_id" not in st.session_state:
        st.session_state.selected_section_id = None
    if "selected_folder_id" not in st.session_state:
        st.session_state.selected_folder_id = None
    if "expand_all_sections" not in st.session_state:
        st.session_state.expand_all_sections = False


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
    # Sidebar: Database Connection & Global Filters
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

        # When profile is changed, automatically update the resolved DB path
        if selected_profile != st.session_state.docgraph_profile:
            st.session_state.docgraph_profile = selected_profile
            resolved = find_docgraph_db_path(selected_profile)
            if resolved:
                st.session_state.docgraph_db_path = resolved
                st.session_state.selected_doc_hash = None
                st.session_state.selected_section_id = None
                st.session_state.selected_folder_id = None
                st.session_state.selected_node_type = None

        if not st.session_state.docgraph_db_path:
            resolved = find_docgraph_db_path(selected_profile)
            if resolved:
                st.session_state.docgraph_db_path = resolved

        db_path_input = st.text_input(
            "Database Path (.lbdb / .db)",
            value=st.session_state.docgraph_db_path,
            placeholder="/path/to/docgraph.db",
            help="Path to the Ladybug / Kuzu database directory",
        )
        st.session_state.docgraph_db_path = db_path_input.strip()

        col_ref, col_stat = st.columns([1, 1])
        with col_ref:
            if st.button("🔄 Reconnect", use_container_width=True):
                st.session_state.selected_doc_hash = None
                st.session_state.selected_section_id = None
                st.session_state.selected_folder_id = None
                st.session_state.selected_node_type = None
                st.rerun()

        active_db_path = st.session_state.docgraph_db_path
        if not active_db_path:
            st.warning("⚠️ Please provide or configure a valid database path.")
            st.info("Ensure the database exists or build one with `cli docgraph build <sources>`.")
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
    # Fetch Folders & Documents
    # -----------------------------------------------------------------------
    folders, documents = await fetch_folders_and_documents(backend)

    if not documents:
        st.warning("No documents found in this database.")
        st.stop()

    # Deduplicate documents across potential duplicate versions
    documents = dedupe_documents(documents)

    # Map documents by hash
    doc_map: dict[str, dict[str, Any]] = {}
    for d in documents:
        m_hash = d.get("markdown_hash") or d.get("content_hash") or ""
        doc_map[m_hash] = d

    # Pre-fetch TOC for documents to build deep hierarchy tree
    sections_by_doc: dict[str, list[dict[str, Any]]] = {}
    if len(documents) <= 150:
        for d in documents:
            m_hash = d.get("markdown_hash")
            if m_hash:
                from genai_graph.kg.query.document_graph_tools import get_document_toc

                sections_by_doc[m_hash] = get_document_toc(backend, m_hash)  # type: ignore[assignment]
    elif st.session_state.selected_doc_hash:
        from genai_graph.kg.query.document_graph_tools import get_document_toc

        sections_by_doc[st.session_state.selected_doc_hash] = get_document_toc(  # type: ignore[assignment]
            backend, st.session_state.selected_doc_hash
        )

    # Default selection if none set
    if not st.session_state.selected_doc_hash and documents:
        first_hash = documents[0].get("markdown_hash") or documents[0].get("content_hash")
        st.session_state.selected_doc_hash = first_hash
        st.session_state.selected_tree_val = f"doc:{first_hash}"
        st.session_state.selected_node_type = "document"

    # -----------------------------------------------------------------------
    # Navigation Tabs
    # -----------------------------------------------------------------------
    tab_doc, tab_search, tab_images, tab_tables, tab_graph = st.tabs(
        [
            "📑 Document & Section Explorer",
            "🔍 Cross-Document Search",
            "🖼️ Image Gallery",
            "📊 Tables Inspector",
            "🕸️ Graph & Schema",
        ]
    )

    # -----------------------------------------------------------------------
    # Tab 1: Document & Section Explorer (Main View: Tree on Left, Content on Right)
    # -----------------------------------------------------------------------
    with tab_doc:
        col_tree, col_content = st.columns([5, 8], gap="large")

        # -------------------------------------------------------------------
        # Left Panel: Tree Navigation
        # -------------------------------------------------------------------
        with col_tree:
            st.markdown("#### 📂 Document Hierarchy")
            st.caption("Browse Folders 📁, Documents 📄, Sections & Subsections 📑")

            # Quick document select dropdown as a shortcut
            doc_options_list = list(doc_map.keys())
            curr_doc_hash = st.session_state.selected_doc_hash
            default_doc_idx = doc_options_list.index(curr_doc_hash) if curr_doc_hash in doc_options_list else 0

            quick_doc = st.selectbox(
                "Quick Document Jump",
                options=doc_options_list,
                index=default_doc_idx,
                format_func=lambda h: f"📄 {doc_map[h].get('filename', h)} ({doc_map[h].get('section_count', 0)} sec)",
                key="quick_doc_jump",
            )
            if quick_doc != st.session_state.selected_doc_hash:
                st.session_state.selected_doc_hash = quick_doc
                st.session_state.selected_tree_val = f"doc:{quick_doc}"
                st.session_state.selected_section_id = None
                st.session_state.selected_node_type = "document"

            # Build tree select nodes
            tree_nodes = build_tree_select_nodes(
                folders=folders,
                documents=documents,
                sections_by_doc=sections_by_doc,
                include_sections=True,
            )

            current_checked_val = st.session_state.get("selected_tree_val")

            # Tree select widget in main area with no_cascade=True for single selection
            tree_result = tree_select(
                nodes=tree_nodes,
                checked=[current_checked_val] if current_checked_val else None,
                no_cascade=True,
                expand_on_click=True,
                show_expand_all=True,
                key="main_docgraph_tree",
            )

            # Handle user clicks on tree nodes - strict single selection
            if tree_result and "checked" in tree_result:
                checked_items = tree_result.get("checked") or []
                if checked_items:
                    new_picks = [v for v in checked_items if v != current_checked_val]
                    chosen_val = new_picks[-1] if new_picks else checked_items[-1]
                else:
                    chosen_val = None

                if chosen_val != current_checked_val:
                    st.session_state.selected_tree_val = chosen_val
                    if chosen_val:
                        if chosen_val.startswith("sec:"):
                            sid = chosen_val.split("sec:", 1)[1]
                            st.session_state.selected_section_id = sid
                            doc_hash = sid.split("::", 1)[0]
                            st.session_state.selected_doc_hash = doc_hash
                            st.session_state.selected_node_type = "section"
                        elif chosen_val.startswith("doc:"):
                            doc_hash = chosen_val.split("doc:", 1)[1]
                            st.session_state.selected_doc_hash = doc_hash
                            st.session_state.selected_section_id = None
                            st.session_state.selected_node_type = "document"
                        elif chosen_val.startswith("folder:"):
                            fid = chosen_val.split("folder:", 1)[1]
                            st.session_state.selected_folder_id = fid
                            st.session_state.selected_section_id = None
                            st.session_state.selected_node_type = "folder"
                    else:
                        st.session_state.selected_section_id = None
                        st.session_state.selected_node_type = "document"
                    st.rerun()

        # -------------------------------------------------------------------
        # Right Panel: Selected Node Content Viewer (Compact & Clean)
        # -------------------------------------------------------------------
        with col_content:
            active_hash = st.session_state.selected_doc_hash
            active_doc = doc_map.get(active_hash, documents[0] if documents else {})
            active_sid = st.session_state.selected_section_id
            node_type = st.session_state.selected_node_type or ("section" if active_sid else "document")

            # A. SECTION SELECTED
            if node_type == "section" and active_sid:
                doc_sections = await fetch_document_sections_full(backend, active_hash)
                target_sec = next((s for s in doc_sections if s["section_id"] == active_sid), None)

                if target_sec:
                    render_section_banner(target_sec, doc_name=active_doc.get("filename"), show_summary=True)

                    btn_c1, btn_c2 = st.columns([1, 1])
                    with btn_c1:
                        if st.button(f"📑 View All Sections in {active_doc.get('filename')}", use_container_width=True):
                            st.session_state.selected_section_id = None
                            st.session_state.selected_tree_val = f"doc:{active_hash}"
                            st.session_state.selected_node_type = "document"
                            st.rerun()
                    with btn_c2:
                        sec_md_raw = target_sec.get("text") or ""
                        st.download_button(
                            "⬇️ Download Section (.md)",
                            data=sec_md_raw,
                            file_name=f"section_{target_sec.get('sequence', 0)}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

                    st.divider()

                    render_section_content_view(
                        target_sec,
                        doc_path=active_doc.get("path"),
                        doc_name=active_doc.get("filename"),
                        db_path=active_db_path,
                        show_banner=False,
                    )
                else:
                    st.warning(f"Section `{active_sid}` not found.")
                    sec_md = await fetch_section_markdown_async(backend, active_sid)
                    if sec_md:
                        st.markdown(f"**Section ID:** `{active_sid}`")
                        st.markdown(sec_md, unsafe_allow_html=True)

            # B. DOCUMENT SELECTED
            elif node_type == "document" and active_doc:
                render_document_banner(active_doc)

                doc_sections = await fetch_document_sections_full(backend, active_hash)

                doc_sub_tab1, doc_sub_tab2 = st.tabs(["📑 Document Sections", "📄 Full Document Markdown"])

                with doc_sub_tab1:
                    exp_col1, exp_col2, _ = st.columns([1, 1, 3])
                    with exp_col1:
                        if st.button("➕ Expand All", use_container_width=True):
                            st.session_state.expand_all_sections = True
                            st.rerun()
                    with exp_col2:
                        if st.button("➖ Collapse All", use_container_width=True):
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
                            label = format_section_expander_label(sec)
                            is_expanded = st.session_state.expand_all_sections or (
                                st.session_state.selected_section_id == sec.get("section_id")
                            )

                            with st.expander(label, expanded=is_expanded):
                                render_section_content_view(
                                    sec,
                                    doc_path=active_doc.get("path"),
                                    doc_name=active_doc.get("filename"),
                                    db_path=active_db_path,
                                    show_banner=False,
                                )

                with doc_sub_tab2:
                    full_md_text = reconstruct_document(backend, active_hash)
                    if full_md_text:
                        st.download_button(
                            "⬇️ Download Markdown (.md)",
                            data=full_md_text,
                            file_name=f"{Path(active_doc.get('filename', 'doc')).stem}.md",
                            mime="text/markdown",
                            use_container_width=False,
                        )
                        st.markdown(full_md_text, unsafe_allow_html=True)
                        with st.expander("Show raw Markdown source", expanded=False):
                            st.code(full_md_text, language="markdown")
                    else:
                        st.warning("Could not reconstruct Markdown for this document.")

            # C. FOLDER SELECTED
            elif node_type == "folder" and st.session_state.selected_folder_id:
                fid = st.session_state.selected_folder_id
                folder_name = PurePosixPath(fid).name if fid != "." else "(root)"
                folder_obj = {"folder_id": fid, "name": folder_name, "doc_count": 0}

                # Find contained documents
                folder_docs = [d for d in documents if _folder_of(d) == fid or d.get("folder_id") == fid]
                folder_obj["doc_count"] = len(folder_docs)
                render_folder_banner(folder_obj)

                st.markdown(f"#### Documents in this Folder ({len(folder_docs)})")
                if not folder_docs:
                    st.info("No documents directly in this folder.")
                else:
                    for d in folder_docs:
                        d_hash = d.get("markdown_hash") or d.get("content_hash") or ""
                        with st.container():
                            st.markdown(
                                f"📄 **{d.get('filename')}** — {d.get('section_count', 0)} sections, {d.get('token_count', 0):,} tokens"
                            )
                            if st.button(f"Open Document: {d.get('filename')}", key=f"btn_doc_{d_hash}"):
                                st.session_state.selected_doc_hash = d_hash
                                st.session_state.selected_tree_val = f"doc:{d_hash}"
                                st.session_state.selected_section_id = None
                                st.session_state.selected_node_type = "document"
                                st.rerun()
                            st.divider()

    # -----------------------------------------------------------------------
    # Tab 2: Cross-Document Search (with Markdown Content Display)
    # -----------------------------------------------------------------------
    with tab_search:
        st.subheader("🔍 Search Sections Across Documents")
        st.caption("Search across all documents and sections with hybrid, semantic vector, BM25, or Cypher modes")

        s_col1, s_col2, s_col3 = st.columns([3, 1, 1])
        with s_col1:
            search_query = st.text_input(
                "Search query / keywords",
                placeholder="e.g. cloud migration strategy, revenue metrics...",
                key="search_query_input",
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
                for idx, r in enumerate(results):
                    score_val = r.get("score")
                    score_badge = f" `(Score: {score_val:.3f})`" if score_val else ""
                    sec_title = r.get("title") or "Untitled Section"
                    sid = r.get("section_id") or ""
                    m_hash = r.get("markdown_hash") or ""
                    lvl = r.get("level", 1)
                    lvl_tag = f"H{lvl}" if lvl > 0 else "Doc"

                    parent_doc = doc_map.get(m_hash, {})
                    doc_fname = parent_doc.get("filename") or m_hash

                    # Reconstruct section markdown content
                    sec_md = (
                        await fetch_section_markdown_async(backend, sid)
                        or r.get("text")
                        or r.get("matched_chunk")
                        or ""
                    )

                    with st.expander(
                        f"📑 [{lvl_tag}] {sec_title}{score_badge} — 🔑 ID: `{sid}` — 📄 {doc_fname}",
                        expanded=(idx == 0),
                    ):
                        # Section Banner with Section ID displayed
                        render_section_banner(r, doc_name=doc_fname, score=score_val, show_summary=True)

                        # Matched Chunk callout if available
                        if r.get("matched_chunk"):
                            st.markdown(
                                f"""
                                <div style="background-color: #f6f8fa; border-left: 4px solid #43C7F4; padding: 8px 12px; border-radius: 4px; margin-bottom: 12px;">
                                    <strong style="color: #00005B;">🎯 Matched Passage:</strong><br/>
                                    <span style="color: #333;">{r["matched_chunk"]}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Rendered Markdown Content
                        st.markdown("#### 📄 Section Markdown Content")
                        if sec_md.strip():
                            st.markdown(sec_md, unsafe_allow_html=True)
                            with st.expander("Show raw markdown source", expanded=False):
                                st.code(sec_md, language="markdown")
                        else:
                            st.info("*(Section body is empty or contains only sub-headings)*")

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


# ---------------------------------------------------------------------------
# Direct / Standalone Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__" or "streamlit" in __name__:
    # Execute async application
    import asyncio

    asyncio.run(render_docgraph_explorer())
