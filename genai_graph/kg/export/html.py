"""Generate an interactive HTML visualization of a Cypher graph.

This module builds a simple JSON model (shared with the DAG renderer via
``_graph_model``), and embeds it in an HTML page rendered with D3
force-directed layout.

It's inspired from code in Cognee.

Features:
- Interactive D3.js force-directed graph visualization
- Node types legend panel (right side) with click-to-highlight
- Selection details panel (left side) showing node/edge properties
- Zoom controls and drag-to-move functionality
- Tooltips on hover for quick inspection

Usage example:
```python
from genai_graph.kg.export.html import generate_html

# Generate visualization from a graph backend
html_content = generate_html(
    connection=graph_backend,
    destination_file_path="my_graph.html",
)
```
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from genai_graph.kg.backend import KgBackend

# Re-export the shared model helpers so existing imports from this module keep
# working (e.g. ``from genai_graph.kg.export.html import _get_node_color``).
from genai_graph.kg.export._graph_model import (  # noqa: F401
    _fetch_graph_data,
    _get_node_color,
    _get_node_display_name,
    _get_node_raw_name,
    _normalize_graph_obj,
    _serialize_kuzu_id,
    build_graph_model,
)
from genai_graph.kg.export.html_template import HTML_TEMPLATE


def _generate_html_content(nodes_list: list[dict[str, Any]], links_list: list[dict[str, Any]]) -> str:
    """Generate HTML content from nodes and links lists.

    Args:
        nodes_list: List of node dictionaries with id, name, type, color, and properties
        links_list: List of link dictionaries with source, target, relation, weight info

    Returns:
        HTML content as a string with embedded D3.js visualization
    """
    _d3_cdn_tag = '    <script src="https://d3js.org/d3.v5.min.js"></script>'
    _d3_bundle = Path(__file__).parent.parent / "schema" / "d3.v5.min.js"
    if _d3_bundle.exists():
        _d3_script = f"<script>{_d3_bundle.read_text(encoding='utf-8')}</script>"
    else:
        _d3_script = _d3_cdn_tag
    html_content = HTML_TEMPLATE.replace("{nodes}", json.dumps(nodes_list))
    html_content = html_content.replace("{links}", json.dumps(links_list))
    html_content = html_content.replace(_d3_cdn_tag, _d3_script)

    return html_content


def generate_html(
    connection: KgBackend,
    destination_file_path: str | None = None,
    node_configs: list | None = None,
    relation_configs: list | None = None,
    custom_colors: dict[str, str] | None = None,
    query: str = "MATCH (n)-[r]->(m) RETURN n, r, m",
    union: bool = True,
    filter_orphan_nodes: bool = False,
    selected_node_types: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Generate an HTML graph visualization from a graph connection/backend.

    Args:
        connection: Object exposing an execute() method (e.g. KgBackend or kuzu.Connection) connected to a database that uses
            a schema with Node(id, name, type, properties) and EDGE(relationship_name, properties).
        destination_file_path: Optional path to write the HTML file. If omitted,
            the file will be saved as "graph_visualization.html" in the user's home directory.
        node_configs: Optional list of node configurations (legacy or new format)
        relation_configs: Optional list of relation configurations (legacy or new format)
        custom_colors: Optional mapping of node types to hex color codes
        query: Cypher query to fetch relationships and nodes (default: "MATCH (n)-[r]->(m) RETURN n, r, m")
               Can be customized to filter by node types, limit results, etc.
               Must return columns named 'n', 'r', 'm' for source node, relationship, target node.
        union: If True and query contains multiple statements, union the results. Default True.
        filter_orphan_nodes: If True, remove nodes that don't appear in any relationship.
            Useful when using multi-hop queries that may include intermediate nodes. Default False.
            Nodes whose type is in ``selected_node_types`` are always preserved regardless of this flag.
        selected_node_types: Node type labels the user explicitly chose to display.  All instances
            of these types are fetched via a supplemental query so that LIMIT constraints on the
            relationship query cannot leave them out.  Nodes of these types are also exempt from
            ``filter_orphan_nodes`` pruning.
        parameters: Optional query parameters bound with ``$name``-style placeholders in *query*.

    Returns:
        The HTML content as a string.
    """
    nodes_data, relationships_data = _fetch_graph_data(
        connection,
        node_configs,
        relation_configs,
        query,
        union,
        supplemental_node_types=selected_node_types,
        parameters=parameters,
    )

    nodes_list, links_list = build_graph_model(
        nodes_data,
        relationships_data,
        custom_colors=custom_colors,
        filter_orphan_nodes=filter_orphan_nodes,
        selected_node_types=selected_node_types,
    )

    html_content = _generate_html_content(nodes_list, links_list)

    if not destination_file_path:
        home_dir = os.path.expanduser("~")
        destination_file_path = os.path.join(home_dir, "graph_visualization.html")

    os.makedirs(os.path.dirname(destination_file_path) or ".", exist_ok=True)
    with open(destination_file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_content
