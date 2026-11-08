"""Streamlit page for Knowledge Graph schema visualization."""

from __future__ import annotations

import json

import streamlit as st
from streamlit import session_state as sss

from genai_graph.kg.export import export_schema_html, export_schema_json
from genai_graph.kg.manager import get_kg_manager
from genai_graph.ui.streamlit.components.kg_config_selector import (
    init_kg_config_session_state,
    render_kg_config_selector,
)


def _show_missing_schema_message(config_name: str) -> None:
    """Show message when a schema HTML export is missing.

    Args:
        config_name: Name of the KG configuration.
    """

    st.warning(
        f"""
        ### 🧩 Schema for '{config_name}' is Missing

        The schema visualization has not been generated yet.

        To create it, run:

        ```bash
        cli kg create --profile={config_name}
        ```

        Then refresh this page.
        """
    )


def main() -> None:
    """Render the schema visualization page."""

    st.set_page_config(
        page_title="Knowledge Graph Schema",
        page_icon="🧩",
        layout="wide",
    )

    init_kg_config_session_state()

    st.title("🧩 Knowledge Graph Schema")

    with st.sidebar:
        render_kg_config_selector(help="Select the Knowledge Graph configuration whose schema you want to inspect.")

    manager = get_kg_manager()

    schema_html_path = manager.get_schema_html_path_for(sss.kg_config_selected)
    schema_json_path = manager.get_schema_json_path_for(sss.kg_config_selected)

    # Ensure artifacts exist (and refresh them when the rendering/template code changes)
    try:
        import os

        from genai_graph.kg.schema import schema_d3 as schema_d3_module
        from genai_graph.kg.schema import schema_html_template as schema_html_template_module

        dep_paths = [
            getattr(schema_d3_module, "__file__", None),
            getattr(schema_html_template_module, "__file__", None),
        ]
        dep_mtime = max(
            (os.path.getmtime(p) for p in dep_paths if p and os.path.exists(p)),
            default=0,
        )

        html_mtime = os.path.getmtime(str(schema_html_path)) if schema_html_path.exists() else 0
        json_mtime = os.path.getmtime(str(schema_json_path)) if schema_json_path.exists() else 0

        if html_mtime < dep_mtime or json_mtime < dep_mtime:
            export_schema_json(sss.kg_config_selected)
            export_schema_html(sss.kg_config_selected)

        # Reload paths after export (files are written under the same names)
        schema_html_path = manager.get_schema_html_path_for(sss.kg_config_selected)
        schema_json_path = manager.get_schema_json_path_for(sss.kg_config_selected)

    except Exception as exc:  # pragma: no cover - defensive
        st.sidebar.error(f"Failed to generate schema artifacts: {exc}")

    with st.sidebar:
        st.markdown("---")
        if schema_json_path.exists():
            try:
                schema_json_text = schema_json_path.read_text(encoding="utf-8")
                meta = json.loads(schema_json_text).get("meta", {})

                st.caption(f"Schema JSON: `{schema_json_path.name}`")
                if meta:
                    st.caption(f"Generated: `{meta.get('generated_at', '')}`")
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Failed to read schema JSON: {exc}")
        else:
            st.caption("Schema JSON not found.")

    if not schema_html_path.exists():
        _show_missing_schema_message(sss.kg_config_selected)
        return

    try:
        html_content = schema_html_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Failed to read schema HTML: {exc}")
        return

    st.iframe(html_content, height=700)


if __name__ == "__main__":
    main()
