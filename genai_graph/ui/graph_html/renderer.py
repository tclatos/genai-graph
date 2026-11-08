"""High-level rendering orchestration for Knowledge Graph HTML visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from genai_tk.ui.config.graph import GraphDisplayConfig

from genai_graph.ui.graph_html.algorithms import prune_graph_by_max_siblings
from genai_graph.ui.graph_html.dag_html import _generate_dag_html_content
from genai_graph.ui.graph_html.html import _generate_html_content
from genai_graph.ui.graph_html.models import _fetch_graph_data, build_graph_model

if TYPE_CHECKING:
    from genai_graph.kg.backend import KgBackend


def render_filtered_graph_html(
    backend: KgBackend,
    query: str = "MATCH (n)-[r]->(m) RETURN n, r, m",
    parameters: dict[str, Any] | None = None,
    max_siblings: int = 10,
    target_node_types: set[str] | None = None,
    focus_node_id: str | None = None,
    layout: str = "force",
    filter_orphan_nodes: bool = True,
    selected_node_types: list[str] | None = None,
    excluded_node_types: list[str] | None = None,
    custom_colors: dict[str, str] | None = None,
    dag_orientation: str = "TB",
    config: GraphDisplayConfig | None = None,
) -> tuple[str, int, int, int]:
    """Execute Cypher query, apply sibling pruning, and render D3 force or DAG HTML.

    Args:
        backend: Connected KgBackend instance
        query: Cypher query to execute
        parameters: Query parameter dict
        max_siblings: Max children per parent (overridden by config if provided)
        target_node_types: Node types to apply sibling limiting to (None = all)
        focus_node_id: Optional node ID to center/start traversal from
        layout: "force" (D3.js force-directed) or "dag" (d3-dag Sugiyama)
        filter_orphan_nodes: Whether to prune nodes with no relationships
        selected_node_types: Node types explicitly selected
        excluded_node_types: Node types explicitly excluded (pruned before model building)
        custom_colors: Color mapping dict
        dag_orientation: "TB" (top-to-bottom / vertical, default) or "LR" (left-to-right / horizontal)
        config: Optional GraphDisplayConfig instance providing default parameters

    Returns:
        Tuple of (html_content, displayed_node_count, displayed_rel_count, pruned_sibling_count)
    """
    cfg = config or GraphDisplayConfig.from_config("ui.display.graph")

    effective_max_siblings = max_siblings if max_siblings is not None else cfg.max_siblings
    effective_layout = layout or cfg.default_layout
    effective_orientation = dag_orientation or cfg.dag_orientation
    effective_filter_orphans = filter_orphan_nodes if filter_orphan_nodes is not None else cfg.filter_orphan_nodes
    effective_colors = custom_colors or cfg.custom_colors or None

    nodes_data, relationships_data = _fetch_graph_data(
        backend,
        query=query,
        union=True,
        parameters=parameters,
        supplemental_node_types=selected_node_types,
    )

    # Filter out excluded node types
    if excluded_node_types:
        exc_set = set(excluded_node_types)
        nodes_data = [n for n in nodes_data if n.properties.get("type") not in exc_set]
        relationships_data = [r for r in relationships_data if r.from_type not in exc_set and r.to_type not in exc_set]

    pruned_nodes, pruned_rels, pruned_count = prune_graph_by_max_siblings(
        nodes_data,
        relationships_data,
        max_siblings=effective_max_siblings,
        target_node_types=target_node_types,
        focus_node_id=focus_node_id,
    )

    nodes_list, links_list = build_graph_model(
        pruned_nodes,
        pruned_rels,
        custom_colors=effective_colors,
        filter_orphan_nodes=effective_filter_orphans,
        selected_node_types=selected_node_types,
    )

    if effective_layout.startswith("dag"):
        html_content = _generate_dag_html_content(
            nodes_list,
            links_list,
            orientation=effective_orientation,
        )
    else:
        html_content = _generate_html_content(nodes_list, links_list)

    return html_content, len(nodes_list), len(links_list), pruned_count
