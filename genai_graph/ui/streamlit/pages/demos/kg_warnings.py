"""Streamlit page for Knowledge Graph creation warnings.

This page lets users:
- Select a KG configuration/profile.
- View categorized warnings generated during KG creation.
- Inspect details per category with descriptions and suggestions.
- View the raw warnings log.
"""

from __future__ import annotations

import streamlit as st
from streamlit import session_state as sss

from genai_graph.kg.export.warnings_report import WarningCategory, WarningsReport, categorize_warnings
from genai_graph.kg.manager import get_kg_manager
from genai_graph.ui.streamlit.components.kg_config_selector import (
    init_kg_config_session_state,
    render_kg_config_selector,
)


def _load_warnings_for(profile: str) -> list[str]:
    """Load raw warning lines from the warnings log file for a given profile.

    Returns:
        List of non-empty warning lines (excluding timestamp headers).
    """
    manager = get_kg_manager()
    warnings_file = manager.get_warnings_file_for(profile)

    if not warnings_file.exists():
        return []

    lines: list[str] = []
    with open(str(warnings_file)) as f:
        for raw in f:
            line = raw.strip()
            # Skip empty lines and timestamp headers
            if not line or line.startswith("=== Warnings at"):
                continue
            lines.append(line)
    return lines


def _render_category(category_idx: int, cat: WarningCategory) -> None:
    """Render a single warning category as an expander."""
    with st.expander(f"{cat.title}  —  {len(cat.warnings)} warning(s)", expanded=(category_idx == 0)):
        st.markdown(f"**📋 Description:** {cat.description}")
        st.markdown(f"**💡 Suggestion:** {cat.suggestion}")

        # Structured details table
        if cat.examples:
            st.markdown("#### 📊 Details")

            if cat.category == "duplicate_relationships":
                st.table(
                    [
                        {"From Node": ex["from_node"], "To Node": ex["to_node"], "Relationships": ex["relationships"]}
                        for ex in cat.examples
                    ]
                )
            elif cat.category in ("missing_nodes", "orphaned_nodes"):
                st.table([{"Node Class": ex["class"]} for ex in cat.examples])
            elif cat.category == "schema_failures":
                st.table([{"Subgraph": ex["subgraph"]} for ex in cat.examples])

        # Raw warning messages
        st.markdown("#### 📝 Raw Warnings")
        st.code("\n".join(cat.warnings), language="text")


def _render_report(report: WarningsReport) -> None:
    """Render the full warnings report."""
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Warnings", report.total_warnings)
    col2.metric("Categories", len(report.categories))
    col3.metric("Uncategorized", len(report.uncategorized))

    st.markdown("---")

    # Categorized warnings
    for idx, cat in enumerate(report.categories):
        _render_category(idx, cat)

    # Uncategorized warnings
    if report.uncategorized:
        with st.expander(f"ℹ️ Uncategorized Warnings  —  {len(report.uncategorized)}", expanded=False):
            for warning in report.uncategorized:
                st.markdown(f"- {warning}")


def _render_no_warnings(config_name: str) -> None:
    """Render a success message when no warnings are found."""
    st.success(
        f"""
        ### ✅ No warnings for '{config_name}'

        The knowledge graph was created successfully with no issues.

        - All node configurations are valid
        - All relationships are properly defined
        - No schema validation issues
        """
    )


def _render_missing_log(config_name: str) -> None:
    """Render a message when no warnings log file exists."""
    st.info(
        f"""
        ### 📭 No warnings log for '{config_name}'

        No warnings log file was found. This means the KG has not been created yet,
        or the warnings log was cleared.

        To create the KG, run:

        ```bash
        cli kg create --profile={config_name}
        ```

        Then refresh this page.
        """
    )


def main() -> None:
    """Render the KG warnings page."""
    st.set_page_config(
        page_title="KG Creation Warnings",
        page_icon="⚠️",
        layout="wide",
    )

    init_kg_config_session_state()

    st.title("⚠️ KG Creation Warnings")

    # --- Sidebar: KG configuration selector ---
    with st.sidebar:
        render_kg_config_selector(help="Select the Knowledge Graph configuration whose warnings you want to inspect.")

    # --- Load warnings ---
    raw_warnings = _load_warnings_for(sss.kg_config_selected)

    # --- Sidebar: file info ---
    with st.sidebar:
        st.markdown("---")
        manager = get_kg_manager()
        warnings_file = manager.get_warnings_file_for(sss.kg_config_selected)
        if warnings_file.exists():
            import os

            mtime = os.path.getmtime(str(warnings_file))
            from datetime import datetime

            last_modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"Warnings log: `{warnings_file.name}`")
            st.caption(f"Last modified: `{last_modified}`")
            st.caption(f"Warning count: `{len(raw_warnings)}`")
        else:
            st.caption("No warnings log file found.")

        # Markdown report link
        warnings_md = manager.get_warnings_md_path_for(sss.kg_config_selected)
        if warnings_md.exists():
            st.caption(f"Markdown report: `{warnings_md.name}`")

    # --- Main content ---
    if not warnings_file.exists():
        _render_missing_log(sss.kg_config_selected)
        return

    if not raw_warnings:
        _render_no_warnings(sss.kg_config_selected)
        return

    # Categorize and render
    report = categorize_warnings(raw_warnings)
    _render_report(report)

    # Optional: show the full markdown report if available
    if warnings_md.exists():
        st.markdown("---")
        with st.expander("📄 Full Markdown Report", expanded=False):
            md_content = warnings_md.read_text(encoding="utf-8")
            st.markdown(md_content)


if __name__ == "__main__":
    main()
