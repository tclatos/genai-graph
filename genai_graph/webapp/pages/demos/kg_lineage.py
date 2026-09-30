"""Streamlit page for Knowledge Graph data source lineage.

This page lets users:
- Select a KG configuration/profile.
- Discover BAML-generated JSON files that contributed to the graph.
- Trace each JSON file back to the originating Markdown and source
  document (PDF or similar) using nearby manifest.json files.
- View Markdown, source PDF, and JSON content side by side.
- View how the Markdown file is chunked for RAG processing.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from genai_tk.config_mgmt.config_mngr import global_config
from genai_tk.config_mgmt.file_patterns import resolve_config_path
from genai_tk.core.factories.chunker_factory import ChunkerFactory
from loguru import logger
from streamlit import session_state as sss

from genai_graph.kg.manager import get_kg_manager
from genai_graph.webapp.ui_components.kg_config_selector import (
    init_kg_config_session_state,
    render_kg_config_selector,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from genai_graph.kg.ingest.lineage import JsonArtifact, MarkdownLineage


def _get_data_roots() -> list[Path]:
    """Get resolved data root paths for relative path display.

    Includes paths.ekg_data (common parent for all data) plus any
    explicit data_root values from subgraph configurations.
    """
    roots = []
    try:
        cfg = global_config()
        # Primary root: paths.ekg_data is the common parent for md/, json/, pdf/
        ekg_data = cfg.get_dir_path("paths.ekg_data")
        if ekg_data:
            roots.append(Path(ekg_data))

        # Also include explicit data_root/md_root/json_cache_root values from graphs
        manager = get_kg_manager()
        profile_cfg = manager.get_profile_dict()
        graphs_cfg = profile_cfg.get("graphs", []) or []

        for graph_cfg in graphs_cfg:
            if not isinstance(graph_cfg, dict):
                continue
            for key in ("data_root", "md_root", "json_cache_root"):
                if key in graph_cfg:
                    resolved = resolve_config_path(graph_cfg[key])
                    roots.append(Path(resolved))
    except Exception as exc:
        logger.warning("Could not extract data_roots: {}", exc)

    return roots


def _make_relative_path(full_path: Path | str, data_roots: list[Path]) -> str:
    """Convert a full path to a relative path based on data_root.

    Tries each data_root and returns the relative path for the first match.
    Falls back to the full path if no data_root matches.
    """
    full_path = Path(full_path)
    for root in data_roots:
        try:
            if full_path.is_relative_to(root):
                return str(full_path.relative_to(root))
        except (ValueError, TypeError):
            continue
    return str(full_path)


def _initialize_session_state() -> None:
    """Initialize session state variables for this page."""
    init_kg_config_session_state()
    if "lineage_selected_dir" not in sss:
        sss.lineage_selected_dir = None
    if "lineage_selected_markdown" not in sss:
        sss.lineage_selected_markdown = None
    if "lineage_selected_json_index" not in sss:
        sss.lineage_selected_json_index = 0
    if "lineage_chunker" not in sss:
        sss.lineage_chunker = "auto"


def _select_configuration() -> None:
    """Render KG configuration selector in the sidebar and apply changes."""
    with st.sidebar:
        render_kg_config_selector(help="Select the Knowledge Graph configuration whose lineage you want to inspect.")

        st.divider()
        st.subheader("Chunker Settings")

        # Get available chunkers from config
        try:
            cfg = global_config()
            chunker_names = list(cfg.get_dict("chunkers", {}).keys())
            chunker_options = ["auto", *chunker_names]
        except Exception:
            chunker_options = ["auto", "markdown", "recursive"]

        sss.lineage_chunker = st.selectbox(
            "Chunker for Markdown Rendering",
            options=chunker_options,
            index=0 if sss.lineage_chunker == "auto" else max(0, chunker_options.index(sss.lineage_chunker)),
            help='Choose "auto" to auto-detect based on file extension, or select a specific chunker.',
        )


def _group_lineage_by_directory(
    lineage: list[MarkdownLineage],
    data_roots: list[Path],
) -> dict[str, list[MarkdownLineage]]:
    """Group lineage entries by their markdown parent directory (relative to data_root)."""

    grouped: dict[str, list[MarkdownLineage]] = defaultdict(list)
    for entry in lineage:
        relative_dir = _make_relative_path(entry.markdown_path.parent, data_roots)
        grouped[relative_dir].append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda e: e.markdown_path.name.lower())
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _select_markdown_entry(
    grouped: dict[str, list[MarkdownLineage]],
) -> MarkdownLineage | None:
    """Render directory + markdown selectors and return the chosen entry."""

    if not grouped:
        return None

    directories = list(grouped.keys())

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📁 Markdown Files")

        # Directory selector to approximate a tree view
        selected_dir = st.selectbox(
            "Directory",
            options=directories,
            index=directories.index(sss.lineage_selected_dir) if sss.lineage_selected_dir in directories else 0,
        )
        sss.lineage_selected_dir = selected_dir

        entries = grouped[selected_dir]
        labels = [entry.markdown_path.name for entry in entries]

        # Keep previous selection when possible
        default_index = 0
        if sss.lineage_selected_markdown in labels:
            default_index = labels.index(sss.lineage_selected_markdown)

        selected_label = st.selectbox(
            "Markdown file",
            options=labels,
            index=default_index,
        )
        sss.lineage_selected_markdown = selected_label

    for entry in entries:
        if entry.markdown_path.name == selected_label:
            return entry

    return entries[0]


def _render_markdown_tab(entry: MarkdownLineage, data_roots: list[Path]) -> None:
    """Render the Markdown content tab."""

    st.subheader("Markdown Document")
    relative_path = _make_relative_path(entry.markdown_path, data_roots)
    st.caption(relative_path)

    try:
        content = entry.markdown_path.read_text(encoding="utf-8")
        st.markdown(content)
    except FileNotFoundError:
        st.error(f"Markdown file not found: {relative_path}")
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to read markdown file: {exc}")


def _render_source_tab(entry: MarkdownLineage, data_roots: list[Path]) -> None:
    """Render the source (PDF or other) tab."""

    st.subheader("Source Document (PDF or original)")

    if not entry.source_path:
        st.info("No source document could be resolved from manifest files for this markdown.")
        return

    relative_path = _make_relative_path(entry.source_path, data_roots)
    st.caption(relative_path)

    if entry.source_path.suffix.lower() == ".pdf":
        # Use the new Streamlit PDF widget when available
        st.pdf(str(entry.source_path), height=800)
    else:
        st.warning(
            "Source document is not a PDF. It cannot be embedded directly, "
            "but you can access it on disk using the path above.",
        )


def _render_json_tab(entry: MarkdownLineage, data_roots: list[Path]) -> None:
    """Render the JSON content tab with optional per-file selector."""

    st.subheader("BAML Generated Files")

    if not entry.json_files:
        st.info("No JSON files recorded for this markdown.")
        return

    json_files: list[JsonArtifact] = entry.json_files

    if len(json_files) == 1:
        selected_index = 0
    else:
        labels = [f"{art.path.name} ({art.subgraph})" for art in json_files]
        selected_index = st.selectbox(
            "JSON file",
            options=list(range(len(json_files))),
            format_func=lambda i: labels[i],
            index=min(sss.lineage_selected_json_index, len(json_files) - 1),
        )
        sss.lineage_selected_json_index = selected_index

    artifact = json_files[selected_index]

    relative_path = _make_relative_path(artifact.path, data_roots)
    st.caption(f"Path: {relative_path}\nSubgraph: {artifact.subgraph}")

    try:
        raw = artifact.path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.error(f"JSON file not found: {relative_path}")
        return
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to read JSON file: {exc}")
        return

    try:
        data = st.session_state.get("_kg_lineage_last_json_data")
        # Always parse fresh to avoid showing stale content if files change
        data = __import__("json").loads(raw)
        st.session_state["_kg_lineage_last_json_data"] = data
        st.json(data)
    except Exception:
        # Fall back to raw text if JSON is not well-formed
        st.code(raw, language="json")


def _get_schema_directory() -> Path:
    """Get the path to the Python schema directory."""
    root = global_config().get_dir_path("paths.src")
    return root / "ekg" / "schema"


def _get_baml_schema_directory() -> Path:
    """Get the path to the BAML schema directory."""
    return Path(__file__).parent.parent.parent.parent / "ekg" / "baml_src" / "schema"


def _render_schema_tab() -> None:
    """Render the Schema code tab with Python and BAML sub-tabs."""

    st.subheader("Schema Definitions")

    root = global_config().get_dir_path("paths.src")
    schema_dir = root / "ekg" / "schema"
    baml_dir = root / "ekg" / "baml_src" / "schema"
    baml_tab, python_tab = st.tabs(["📐 BAML Schema", "🐍 Python Schema"])

    with python_tab:
        _render_python_schema(schema_dir)

    with baml_tab:
        _render_baml_schema(baml_dir)


def _render_python_schema(schema_dir: Path) -> None:
    """Render Python schema files from the schema directory."""

    if not schema_dir.exists():
        st.warning(f"Schema directory not found: {schema_dir}")
        return

    # Find all Python files (excluding __pycache__ and __init__.py)
    python_files = sorted(
        [f for f in schema_dir.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_")]
    )

    if not python_files:
        st.info("No Python schema files found.")
        return

    # File selector
    selected_file = st.selectbox(
        "Select Python schema file",
        options=python_files,
        format_func=lambda f: f.stem,
        key="python_schema_selector",
    )

    if selected_file:
        st.caption(f"📁 {selected_file.name}")
        try:
            content = selected_file.read_text(encoding="utf-8")
            st.code(content, language="python", line_numbers=True)
        except Exception as exc:
            st.error(f"Failed to read file: {exc}")


def _render_baml_schema(baml_dir: Path) -> None:
    """Render BAML schema files from the baml_src/schema directory."""

    if not baml_dir.exists():
        st.warning(f"BAML schema directory not found: {baml_dir}")
        return

    # Find all BAML files
    baml_files = sorted(baml_dir.glob("*.baml"))

    if not baml_files:
        st.info("No BAML schema files found.")
        return

    # File selector
    selected_file = st.selectbox(
        "Select BAML schema file",
        options=baml_files,
        format_func=lambda f: f.stem,
        key="baml_schema_selector",
    )

    if selected_file:
        st.caption(f"📁 {selected_file.name}")
        try:
            content = selected_file.read_text(encoding="utf-8")
            st.code(content, language="typescript", line_numbers=True)
        except Exception as exc:
            st.error(f"Failed to read file: {exc}")


def _increase_markdown_header_levels(content: str) -> str:
    """Increase markdown header levels by 1 (# -> ##, ## -> ###, etc).

    This makes top-level headers appear smaller in the UI.
    """
    import re

    def replace_header(match: re.Match[str]) -> str:
        hashes = match.group(1)
        # Add one more hash to increase header level
        return "#" + hashes + match.group(2)

    # Match markdown headers (# at start of line with optional whitespace)
    return re.sub(r"^(#+)(\s+.*)$", replace_header, content, flags=re.MULTILINE)


def _render_chunks_tab(entry: MarkdownLineage, data_roots: list[Path]) -> None:
    """Render the Markdown chunks tab showing how the file is chunked for RAG."""

    st.subheader("Markdown Chunks")
    relative_path = _make_relative_path(entry.markdown_path, data_roots)
    st.caption(f"Chunks for: {relative_path}")

    try:
        path = Path(entry.markdown_path)
        if not path.exists():
            st.error(f"Markdown file not found: {relative_path}")
            return

        # Auto-detect chunker based on file extension
        splitter = ChunkerFactory.create_for_file(path, "auto")
        content = path.read_text(encoding="utf-8")
        docs = splitter.create_documents([content], metadatas=[{"source": str(relative_path)}])
    except Exception as exc:
        st.error(f"Failed to chunk markdown file: {exc}")
        return

    if not docs:
        st.info("No chunks generated from this file.")
        return

    # Summary statistics
    total_tokens = sum(doc.metadata.get("token_count", 0) for doc in docs)
    type_counts: dict[str, int] = {}
    for doc in docs:
        chunk_type = doc.metadata.get("chunk_type", "text")
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", len(docs))
    with col2:
        st.metric("Total Tokens", f"{total_tokens:,}")
    with col3:
        types_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
        st.metric("Chunk Types", types_str)

    st.markdown("---")

    # Display chunks as a table with full content in markdown
    for idx, doc in enumerate(docs):
        with st.container(border=True):
            col_meta, col_content = st.columns([1, 6])
            chunk_type = doc.metadata.get("chunk_type", "text")
            token_count = doc.metadata.get("token_count", 0)
            start_index = doc.metadata.get("start_index", 0)
            with col_meta:
                st.markdown(f"**#{idx + 1}**  \n`{chunk_type}`  \n{token_count} tokens  \npos: {start_index}+")
            with col_content:
                display_content = _increase_markdown_header_levels(doc.page_content)
                st.markdown(display_content)


def main() -> None:
    """Main Streamlit app for KG data source lineage exploration."""

    st.set_page_config(
        page_title="Knowledge Graph Data Lineage",
        page_icon="🧬",
        layout="wide",
    )

    _initialize_session_state()

    st.title("🧬 Knowledge Graph Lineage")
    st.markdown(
        """Inspect how your Knowledge Graph was built from BAML JSON
        files, intermediate Markdown, and original source documents.""",
    )

    _select_configuration()

    # Get data_roots for relative path display
    data_roots = _get_data_roots()

    # Load lineage information for the active configuration
    try:
        manager = get_kg_manager()
        lineage_entries, import_errors = manager.get_data_lineage()
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to collect data lineage: {exc}")
        return

    if import_errors:
        with st.expander(f"⚠️ {len(import_errors)} subgraph(s) could not be loaded", expanded=not lineage_entries):
            for err in import_errors:
                if err.hint:
                    st.warning(f"**{err.factory_path}**  \n💡 {err.hint}")
                else:
                    st.warning(f"**{err.factory_path}**  \n```\n{err.error}\n```")

    if not lineage_entries:
        st.info(
            "No BAML-based JSON lineage found for this configuration. "
            "Ensure you have run KG creation and that JSON-backed "
            "subgraphs are configured.",
        )
        return

    grouped = _group_lineage_by_directory(lineage_entries, data_roots)
    selected_entry = _select_markdown_entry(grouped)

    if not selected_entry:
        st.info("No markdown files discovered for lineage.")
        return

    # Tabs for different artifact types
    default = "📄 Markdown ➥"
    tab_src, tab_md, tab_json, tab_schema, tab_chunks = st.tabs(
        [
            "📎 Source Document ➥",
            default,
            "🧱 LLM Structured Output",
            "📋 Schema Code",
            "🧩 Chunks (for Embeddings/RAG)",
        ],
        default=default,
    )

    with tab_md:
        _render_markdown_tab(selected_entry, data_roots)

    with tab_src:
        _render_source_tab(selected_entry, data_roots)

    with tab_chunks:
        _render_chunks_tab(selected_entry, data_roots)

    with tab_json:
        _render_json_tab(selected_entry, data_roots)

    with tab_schema:
        _render_schema_tab()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
