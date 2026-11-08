"""Reusable Streamlit component for interactive Knowledge Graph visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st
from genai_tk.ui.config.graph import GraphDisplayConfig
from streamlit import session_state as sss

from genai_graph.ui.graph_html import (
    build_filtered_cypher_query,
    get_node_types,
    get_nodes_of_type,
    get_relationship_types,
    get_top_node_types,
    get_top_nodes_of_type,
    render_filtered_graph_html,
)

if TYPE_CHECKING:
    from genai_graph.kg.backend import KgBackend


def render_graph_visualization(
    backend: KgBackend,
    config: GraphDisplayConfig | None = None,
    session_prefix: str = "kg_viz",
    height: int = 800,
) -> None:
    """Render the full interactive graph visualizer (sidebar controls + D3/DAG view).

    Args:
        backend: Connected KgBackend database instance
        config: Optional GraphDisplayConfig overriding defaults/YAML
        session_prefix: State variable prefix in st.session_state
        height: Height in px of the embedded iframe
    """
    cfg = config or GraphDisplayConfig.from_config("ui.display.graph")

    html_key = f"{session_prefix}_html"
    metrics_key = f"{session_prefix}_metrics"
    query_key = f"{session_prefix}_query"

    if html_key not in sss:
        sss[html_key] = None
    if metrics_key not in sss:
        sss[metrics_key] = None
    if query_key not in sss:
        sss[query_key] = ""

    available_node_types = get_node_types(backend)
    available_rel_types = get_relationship_types(backend)

    if not available_node_types and not available_rel_types:
        st.warning("The graph database contains no tables or is empty.")
        return

    # ── Sidebar Controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🎯 Visualization Filters")

        # 1. Top Node Scope (Root Hierarchy Selection)
        top_node_type_options = get_top_node_types(backend, configured_default=cfg.top_node_type)
        selected_top_node_type = st.selectbox(
            "Top Node Type",
            options=top_node_type_options,
            index=0,
            help="Select '(NA)' to display the graph normally without root scoping, or select an entity type (e.g. Opportunity) to anchor the hierarchy.",
        )

        selected_top_node_id = None
        selected_top_node_label = "(NA)"
        if selected_top_node_type not in ("(NA)", "NA", None):
            top_nodes = get_top_nodes_of_type(backend, selected_top_node_type, limit=250)
            top_node_choices = ["(All)", *(n["label"] for n in top_nodes)]
            selected_top_node_label = st.selectbox(
                f"Select {selected_top_node_type}",
                options=top_node_choices,
                index=0,
                help=f"Focus the graph on a specific {selected_top_node_type} and display all child connected nodes.",
            )
            if selected_top_node_label != "(All)":
                for n in top_nodes:
                    if n["label"] == selected_top_node_label:
                        selected_top_node_id = n["id"]
                        break

        # 2. Node Filters
        with st.expander("Node Filters", expanded=False):
            selected_node_types = st.multiselect(
                "Include Node Types",
                options=available_node_types,
                default=[],
                help="Select specific node types to show (leave empty to include all).",
            )

            exclude_options = [t for t in available_node_types if t not in selected_node_types]
            default_exclude = [t for t in cfg.default_exclude_node_types if t in exclude_options]
            selected_exclude_types = st.multiselect(
                "Exclude Node Types",
                options=exclude_options,
                default=default_exclude,
                help="Filter out noisy or unwanted node types.",
            )

            limit = st.number_input(
                "Query Result Limit",
                min_value=50,
                max_value=10000,
                value=cfg.default_limit,
                step=100,
                help="Maximum relationships fetched from the database.",
            )

            filter_orphans = st.checkbox(
                "Remove Orphan Nodes",
                value=cfg.filter_orphan_nodes,
                help="Hide disconnected nodes that have no edges in the filtered view.",
            )

        # Specific node focus (when 1 node type is selected)
        selected_node_name = None
        if len(selected_node_types) == 1:
            target_type = selected_node_types[0]
            nodes_of_type = get_nodes_of_type(backend, target_type, limit=250)
            if nodes_of_type:
                node_choice = st.selectbox(
                    f"Focus on specific {target_type}",
                    options=["(All)", *nodes_of_type],
                    index=0,
                    help=f"Center the graph on a single {target_type} node.",
                )
                selected_node_name = None if node_choice == "(All)" else node_choice

        # 3. Relationship Filters
        with st.expander("Relationship Filters", expanded=False):
            selected_rel_types = st.multiselect(
                "Relationship Types",
                options=available_rel_types,
                default=[],
                help="Select relationship types to display (leave empty for all).",
            )

        # 4. Sibling Limiter
        with st.expander("Sibling Limiter", expanded=False):
            enable_sibling_filter = st.checkbox(
                "Limit Siblings per Parent",
                value=(cfg.max_siblings > 0),
                help="Caps the maximum number of child nodes/subsections displayed per parent to prevent visual clutter.",
            )

            if enable_sibling_filter:
                max_siblings = st.slider(
                    "Max Siblings per Parent",
                    min_value=1,
                    max_value=30,
                    value=cfg.max_siblings or 10,
                    step=1,
                    help="Maximum number of outgoing edges to children of each relationship type.",
                )
                scope_choices = ["All Node Types"] + (["Target types only"] if cfg.target_sibling_types else [])
                sibling_scope = st.radio("Limit Scope", options=scope_choices, index=0)
                target_prune_types = set(cfg.target_sibling_types) if sibling_scope == "Target types only" else None
            else:
                max_siblings = 0
                target_prune_types = None

        # 5. Layout and Result Limit
        st.markdown("**📐 Display Options**")
        default_layout_idx = 1 if cfg.default_layout == "dag" else 0
        layout_choice = st.radio(
            "Graph Layout Engine",
            options=["🕸️ Force-Directed", "🔀 Layered DAG"],
            index=default_layout_idx,
            help="Layered DAG highlights top-down/left-to-right hierarchy. Force-directed simulation highlights clusters.",
        )
        layout = "dag" if "DAG" in layout_choice else "force"

        dag_orientation = cfg.dag_orientation
        if layout == "dag":
            default_ori_idx = 0 if cfg.dag_orientation == "TB" else 1
            dag_ori_label = st.radio(
                "DAG Orientation",
                options=[
                    "⬇️ Top-Down (Vertical, recommended)",
                    "➡️ Left-to-Right (Horizontal)",
                ],
                index=default_ori_idx,
                help="Top-Down places root nodes at the top. Left-to-Right places roots on the left.",
            )
            dag_orientation = "TB" if "Top-Down" in dag_ori_label else "LR"

        st.divider()

        # Render Button
        render_clicked = st.button("🎨 Render Visualization", type="primary", use_container_width=True)

    # ── Execution ────────────────────────────────────────────────────────────
    should_render = render_clicked or sss[html_key] is None

    if should_render:
        with st.spinner("Executing Cypher query and pruning graph siblings..."):
            try:
                query, params = build_filtered_cypher_query(
                    node_types=selected_node_types,
                    node_name=selected_node_name,
                    relationship_types=selected_rel_types,
                    limit=limit,
                    excluded_node_types=selected_exclude_types,
                    top_node_type=selected_top_node_type,
                    top_node_id=selected_top_node_id,
                    max_hops=cfg.max_hops if hasattr(cfg, "max_hops") else 6,
                    backend=backend,
                )

                html_content, n_count, r_count, p_count = render_filtered_graph_html(
                    backend=backend,
                    query=query,
                    parameters=params,
                    max_siblings=max_siblings,
                    target_node_types=target_prune_types,
                    focus_node_id=selected_top_node_id or selected_node_name,
                    layout=layout,
                    filter_orphan_nodes=filter_orphans,
                    selected_node_types=selected_node_types or None,
                    excluded_node_types=selected_exclude_types,
                    dag_orientation=dag_orientation,
                    config=cfg,
                )

                sss[html_key] = html_content
                sss[query_key] = query
                sss[metrics_key] = {
                    "nodes": n_count,
                    "relationships": r_count,
                    "pruned_siblings": p_count,
                    "layout": f"Layered DAG ({dag_orientation})" if layout == "dag" else "Force-Directed",
                    "top_scope": f"{selected_top_node_type}: {selected_top_node_label}"
                    if selected_top_node_id
                    else f"(All {selected_top_node_type}s)"
                    if selected_top_node_type not in ("(NA)", "NA", None)
                    else "All Nodes",
                }
            except Exception as exc:
                st.error(f"Failed to generate visualization: {exc}")
                sss[html_key] = None

    # ── Main Display Area ────────────────────────────────────────────────────
    if sss[html_key] and sss[metrics_key]:
        m = sss[metrics_key]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Displayed Nodes", m["nodes"])
        with col2:
            st.metric("Displayed Relationships", m["relationships"])
        with col3:
            pruned = m["pruned_siblings"]
            st.metric(
                "Pruned Siblings",
                pruned,
                delta=f"-{pruned}" if pruned > 0 else None,
                delta_color="inverse",
            )

        st.iframe(sss[html_key], height=height)

        with st.expander("🔍 Executed Cypher Query", expanded=False):
            st.code(sss[query_key], language="cypher")
