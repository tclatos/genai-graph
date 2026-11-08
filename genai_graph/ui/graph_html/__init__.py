"""Graph HTML generation layer: D3 force, d3-dag Sugiyama, algorithms, and schema inspection."""

from __future__ import annotations

from genai_graph.ui.graph_html.algorithms import prune_graph_by_max_siblings
from genai_graph.ui.graph_html.dag_html import generate_dag_html
from genai_graph.ui.graph_html.html import generate_html
from genai_graph.ui.graph_html.models import (
    NodeRecord,
    RelationshipRecord,
    build_graph_model,
)
from genai_graph.ui.graph_html.query_builder import build_filtered_cypher_query
from genai_graph.ui.graph_html.renderer import render_filtered_graph_html
from genai_graph.ui.graph_html.schema_inspect import (
    get_node_types,
    get_nodes_of_type,
    get_opportunity_folders,
    get_primary_key,
    get_relationship_types,
    get_top_node_types,
    get_top_nodes_of_type,
)

get_available_node_types = get_node_types
get_available_relationship_types = get_relationship_types

__all__ = [
    "NodeRecord",
    "RelationshipRecord",
    "build_filtered_cypher_query",
    "build_graph_model",
    "generate_dag_html",
    "generate_html",
    "get_available_node_types",
    "get_available_relationship_types",
    "get_node_types",
    "get_nodes_of_type",
    "get_opportunity_folders",
    "get_primary_key",
    "get_relationship_types",
    "get_top_node_types",
    "get_top_nodes_of_type",
    "prune_graph_by_max_siblings",
    "render_filtered_graph_html",
]
