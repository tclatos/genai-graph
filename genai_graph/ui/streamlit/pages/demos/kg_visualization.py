"""Streamlit page for Knowledge Graph HTML visualization."""

from __future__ import annotations

import streamlit as st

from genai_graph.kg.backend import create_backend_from_config
from genai_graph.ui.config.graph import GraphDisplayConfig
from genai_graph.ui.streamlit.components.graph_viewer import render_graph_visualization
from genai_graph.ui.streamlit.components.kg_config_selector import (
    init_kg_config_session_state,
    render_kg_config_selector,
)

st.set_page_config(
    page_title="Knowledge Graph Visualization",
    page_icon="🕸️",
    layout="wide",
)

st.title("🕸️ Knowledge Graph Visualization")
st.caption("Interactive D3 and DAG graph visualizer with sibling limiting and node filtering.")

init_kg_config_session_state()
render_kg_config_selector(help="Select the Knowledge Graph configuration to visualize")
selected_config = st.session_state.get("kg_config_selected", "default")

try:
    backend = create_backend_from_config("default", selected_config)
    if not backend:
        st.error(f"❌ Could not connect to Knowledge Graph configuration '{selected_config}'")
        st.stop()
except Exception as exc:
    st.error(f"❌ Failed to connect to graph database: {exc}")
    st.stop()

config = GraphDisplayConfig.from_config("ui.display.graph")
render_graph_visualization(backend=backend, config=config)
