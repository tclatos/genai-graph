"""Generate an interactive HTML visualization of a graph db graph from a given connection.

Work in Progress: 2D is not as nice as with D3.js current implementation, and 3D as issues
Usage example:
```

```
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from genai_graph.core.graph_backend import GraphBackend

# Import new schema types

# from genai_graph.demos.ekg.graph_schema import GraphNodeConfig, GraphRelationConfig, GraphSchema


def _generate_node_id(node_type: str, node_name: str, max_length: int = 50) -> str:
    """Generate a consistent node ID from node type and name.

    Args:
        node_type: The node type/table name
        node_name: The node name/display name
        max_length: Maximum length for the ID

    Returns:
        A consistent node ID string
    """
    # Clean the name for use in ID
    clean_name = str(node_name).replace(" ", "_").replace("/", "_").replace("\n", "_")
    # Create ID and ensure it doesn't exceed max length
    node_id = f"{node_type}_{clean_name}"
    if len(node_id) > max_length:
        # Truncate but keep the type intact
        available_length = max_length - len(node_type) - 1  # -1 for the underscore
        if available_length > 0:
            node_id = f"{node_type}_{clean_name[:available_length]}"
        else:
            node_id = node_type[:max_length]
    return node_id


def _get_node_raw_name(node_dict: dict[str, Any], node_type: str) -> str:
    """Extract the raw name for a node without truncation (for ID generation).

    Args:
        node_dict: Node properties dictionary
        node_type: The node type/table name

    Returns:
        Raw name for the node (no truncation)
    """
    # PRIORITY 1: Check for our custom _name field first
    if "_name" in node_dict and node_dict["_name"] is not None:
        value = str(node_dict["_name"]).strip()
        if value:
            return value

    # PRIORITY 2: Common name fields to check in order of preference
    name_fields = ["name", "title", "description", "label", "id"]

    for field in name_fields:
        if field in node_dict and node_dict[field] is not None:
            value = str(node_dict[field]).strip()
            if value:
                return value

    # PRIORITY 3: If no name field found, use the first non-empty string field
    for key, value in node_dict.items():
        if isinstance(value, str) and value.strip() and key not in ["type", "id", "_name"]:
            return str(value)

    # Fallback to node type
    return node_type


def _get_node_display_name(node_dict: dict[str, Any], node_type: str, max_length: int = 30) -> str:
    """Generate a display name for a node based on its properties.

    Args:
        node_dict: Node properties dictionary
        node_type: The node type/table name
        max_length: Maximum length for the display name

    Returns:
        Display name for the node
    """
    # Get the raw name first
    raw_name = _get_node_raw_name(node_dict, node_type)

    # Apply truncation only for display
    if len(raw_name) > max_length:
        return raw_name[:max_length] + "..."
    return raw_name

    # If no name field found, use the first non-empty string field
    for key, value in node_dict.items():
        if isinstance(value, str) and value.strip() and key not in ["type", "id"]:
            truncated = str(value)[:max_length]
            return truncated + ("..." if len(str(value)) > max_length else "")

    # Fallback to node type
    return node_type


def _get_node_color(node_type: str, custom_colors: dict[str, str] | None = None) -> str:
    """Get color for a node type.

    Args:
        node_type: The node type/table name
        custom_colors: Optional custom color mapping

    Returns:
        Hex color code for the node
    """
    if custom_colors and node_type in custom_colors:
        return custom_colors[node_type]

    # Generate a consistent color based on node type hash
    import hashlib

    # Create a hash of the node type
    hash_object = hashlib.md5(node_type.encode())
    hex_hash = hash_object.hexdigest()

    # Use first 6 characters as color, but ensure it's not too dark
    color = "#" + hex_hash[:6]

    # Brighten the color if it's too dark
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)

    # Ensure minimum brightness
    min_brightness = 100
    if r < min_brightness:
        r = min(255, r + min_brightness)
    if g < min_brightness:
        g = min(255, g + min_brightness)
    if b < min_brightness:
        b = min(255, b + min_brightness)

    return f"#{r:02x}{g:02x}{b:02x}"


def _fetch_graph_data(
    connection: GraphBackend,
    node_configs: list | None = None,
    relation_configs: list | None = None,
) -> tuple[list[tuple[str, dict]], list[tuple[str, str, str, dict]]]:
    """Fetch all nodes and edges from the graph database via the provided connection/backend.

    Args:
        connection: Object exposing an execute() method (e.g. GraphBackend )
        node_configs: Optional list of node configurations (legacy or new format)
        relation_configs: Optional list of relation configurations (legacy or new format)

    Returns:
        The function returns two lists:
        - nodes: list of (node_id, properties_dict)
        - edges: list of (source_id, target_id, relationship_name, properties_dict)
    """
    nodes: list[tuple[str, dict]] = []
    edges: list[tuple[str, str, str, dict]] = []

    # Optional filtering based on provided node / relation configs
    allowed_node_labels: set[str] | None = None
    allowed_rel_types: set[str] | None = None

    if node_configs:
        try:
            labels: set[str] = set()
            for cfg in node_configs:
                baml_class = getattr(cfg, "baml_class", None)
                if baml_class is not None and hasattr(baml_class, "__name__"):
                    labels.add(baml_class.__name__)
            if labels:
                allowed_node_labels = labels
        except Exception:
            # Fail open if configs are not in the expected shape
            allowed_node_labels = None

    if relation_configs:
        try:
            rel_types: set[str] = set()
            for cfg in relation_configs:
                name = getattr(cfg, "name", None)
                if isinstance(name, str):
                    rel_types.add(name)
            if rel_types:
                allowed_rel_types = rel_types
        except Exception:
            allowed_rel_types = None

    # Get all tables first to understand the schema
    try:
        tables_result = connection.execute("CALL show_tables() RETURN *")
        tables_df = tables_result.get_as_df()

        node_tables = []

        for _, row in tables_df.iterrows():
            table_name = row["name"]
            table_type = row["type"]
            if table_type == "NODE":
                if allowed_node_labels and table_name not in allowed_node_labels:
                    continue
                node_tables.append(table_name)

        # Create a mapping to store UUID to node data for relationship matching
        uuid_to_node_data = {}

        # Fetch nodes from all node tables using simple RETURN n query
        for table_name in node_tables:
            try:
                # Always use simple query to avoid field extraction issues
                nodes_query = f"MATCH (n:{table_name}) RETURN n"
                nodes_result = connection.execute(nodes_query)
                result_df = nodes_result.get_as_df()

                for _idx, row in result_df.iterrows():
                    node_dict = {}

                    # Extract node data from the first column (the node object)
                    node_obj = row.iloc[0] if len(row) > 0 else None

                    if isinstance(node_obj, dict):
                        # Handle dictionary-based results (most common)
                        for key, val in node_obj.items():
                            # Keep metadata fields like _name, _created_at, _updated_at
                            # but skip graph db internal fields like _id, _label
                            if key in ("_name", "_created_at", "_updated_at"):
                                node_dict[key] = str(val).strip() if str(val).strip() else str(val)
                            elif not key.startswith("_") and val is not None:
                                node_dict[key] = str(val).strip() if str(val).strip() else str(val)
                    else:
                        # Handle object-based results (fallback)
                        try:
                            for attr in dir(node_obj):
                                if not attr.startswith("_") and hasattr(node_obj, attr):
                                    val = getattr(node_obj, attr)
                                    if val is not None and not callable(val):
                                        node_dict[attr] = str(val).strip() if str(val).strip() else str(val)
                        except Exception:
                            # Last resort: skip this node
                            continue

                    # Skip empty nodes
                    if not node_dict:
                        continue

                    # Generate display name and add node metadata
                    node_name = _get_node_display_name(node_dict, table_name)
                    node_dict["type"] = table_name
                    node_dict["name"] = node_name

                    # Generate UUID-based ID for absolute uniqueness and consistency
                    node_uuid = str(uuid.uuid4())

                    # Store mapping for relationship resolution
                    # Use graph db internal ID for perfect consistency
                    kuzu_id = None
                    if isinstance(node_obj, dict) and "_id" in node_obj:
                        kuzu_id = str(node_obj["_id"])  # Use internal graph db ID as key

                    if kuzu_id:
                        uuid_to_node_data[kuzu_id] = {"uuid": node_uuid, "type": table_name, "node_dict": node_dict}

                    nodes.append((node_uuid, node_dict))

            except Exception as e:
                print(f"Error fetching nodes from {table_name}: {e}")
                continue

        # If no nodes, return early
        if not nodes:
            return [], []

        # UUID mapping complete

        # Fetch relationships using explicit queries
        try:
            # Try to get all relationships - basic graph db syntax
            rel_query = "MATCH (n)-[r]->(m) RETURN n, r, m"
            rel_result = connection.execute(rel_query)
            rel_df = rel_result.get_as_df()

            for _, row in rel_df.iterrows():
                # Extract source and destination node data
                src_node = row["n"]
                dst_node = row["m"]
                rel_obj = row["r"]

                # Get relationship type from relationship object (graph db returns dict)
                rel_type = "RELATED_TO"
                edge_props = {}
                if isinstance(rel_obj, dict):
                    if "_label" in rel_obj:
                        rel_type = rel_obj["_label"]
                    # Extract edge properties (non-internal fields)
                    for key, value in rel_obj.items():
                        if not key.startswith("_") and value is not None:
                            edge_props[key] = value
                elif hasattr(rel_obj, "__class__"):
                    rel_type = rel_obj.__class__.__name__.replace("Relationship", "").replace("Record", "")
                    if not rel_type or rel_type == "object":
                        rel_type = "RELATED_TO"

                # Respect relation-type filtering when requested
                if allowed_rel_types and rel_type not in allowed_rel_types:
                    continue

                # Extract node types and names from dictionary-based graph db results
                def extract_node_info(node_obj: dict) -> tuple[str, str]:
                    """Extract node type and name from a graph db node object (dictionary)."""
                    node_type = "Unknown"
                    node_name = "unknown"

                    # Handle dictionary-based graph db results
                    if isinstance(node_obj, dict):
                        # Get node type from _label
                        if "_label" in node_obj:
                            node_type = node_obj["_label"]

                        # Create a clean dictionary for name extraction (exclude internal graph db fields)
                        node_dict = {}
                        for key, value in node_obj.items():
                            if not key.startswith("_") and value is not None:
                                node_dict[key] = value

                        if node_dict:
                            node_name = _get_node_raw_name(node_dict, node_type)
                    else:
                        # Fallback for object-based results (if any)
                        if hasattr(node_obj, "__class__"):
                            class_name = node_obj.__class__.__name__
                            if class_name != "object":
                                node_type = class_name

                        # Extract name using attribute access
                        node_dict = {}
                        for attr in dir(node_obj):
                            if not attr.startswith("_") and hasattr(node_obj, attr):
                                try:
                                    value = getattr(node_obj, attr)
                                    if value is not None and not callable(value):
                                        node_dict[attr] = value
                                except Exception:
                                    continue

                        if node_dict:
                            node_name = _get_node_raw_name(node_dict, node_type)

                    return node_type, node_name

                # Extract graph db internal IDs for perfect matching
                src_kuzu_id = None
                dst_kuzu_id = None

                if isinstance(src_node, dict) and "_id" in src_node:
                    src_kuzu_id = str(src_node["_id"])
                if isinstance(dst_node, dict) and "_id" in dst_node:
                    dst_kuzu_id = str(dst_node["_id"])

                src_uuid = None
                dst_uuid = None

                if src_kuzu_id and src_kuzu_id in uuid_to_node_data:
                    src_uuid = uuid_to_node_data[src_kuzu_id]["uuid"]

                if dst_kuzu_id and dst_kuzu_id in uuid_to_node_data:
                    dst_uuid = uuid_to_node_data[dst_kuzu_id]["uuid"]

                # Only add if we have valid UUIDs for both nodes
                if src_uuid and dst_uuid:
                    edges.append((src_uuid, dst_uuid, rel_type, edge_props))

        except Exception as e:
            print(f"Error fetching relationships: {e}")
            print(f"Error in schema-aware relationship extraction: {e}")

        # Note: UUID-based IDs ensure all relationships are valid

        # If we have nodes but no relationships and relation_configs are provided,
        # we could potentially create logical connections based on the relation configs
        # However, for a truly generic solution, we'll just use what we found

    except Exception as e:
        print(f"Error in _fetch_graph_data: {e}")
        return [], []

    return nodes, edges


def _build_force_graph_html_content(
    nodes_list: list[dict[str, Any]],
    links_list: list[dict[str, Any]],
    use_3d: bool = True,
) -> str:
    """Build HTML content using force-graph / 3d-force-graph.

    Args:
        nodes_list: List of node dictionaries ready for visualization
        links_list: List of link dictionaries ready for visualization
        use_3d: Whether to render the visualization in 3D or 2D

    Returns:
        HTML content as a string.
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset=\"utf-8\">
        <title>graph db Graph Visualization</title>
        <style>
            html, body {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                background: linear-gradient(90deg, #101010, #1a1a2e);
                color: white;
                font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }

            #graph {
                width: 100vw;
                height: 100vh;
            }

            .tooltip {
                position: absolute;
                text-align: left;
                padding: 8px;
                font-size: 10px;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.2s;
                z-index: 1000;
                max-width: 500px;
                word-wrap: break-word;
                max-height: 80vh;
                overflow-y: auto;
            }
        </style>
        <script src=\"https://cdn.jsdelivr.net/npm/3d-force-graph\"></script>
        <script src=\"https://cdn.jsdelivr.net/npm/force-graph\"></script>
        <script src=\"https://unpkg.com/three-spritetext\"></script>
    </head>
    <body>
        <div id=\"graph\"></div>
        <div class=\"tooltip\" id=\"tooltip\"></div>
        <script>
            const PREFERRED_3D = __IS_3D__;

            function hasWebGL() {
                try {
                    if (!window.WebGLRenderingContext) {
                        return false;
                    }
                    const canvas = document.createElement('canvas');
                    return !!(
                        canvas.getContext('webgl') ||
                        canvas.getContext('experimental-webgl')
                    );
                } catch (e) {
                    return false;
                }
            }

            const IS_3D = PREFERRED_3D && hasWebGL();
            if (PREFERRED_3D && !IS_3D) {
                console.warn('3D WebGL not available; falling back to 2D force-graph.');
            }

            const nodes = {nodes};
            const links = {links};

            const graphElem = document.getElementById('graph');
            const tooltip = document.getElementById('tooltip');

            let Graph;
            try {
                if (IS_3D) {
                    Graph = ForceGraph3D()(graphElem);
                } else {
                    Graph = ForceGraph()(graphElem);
                }
            } catch (e) {
                // If 3D/WebGL init fails (common on some environments), fall back to 2D
                Graph = ForceGraph()(graphElem);
            }

            Graph
                .graphData({ nodes, links })
                .nodeId('id')
                .nodeLabel(d => d.name || d.id)
                .nodeColor(d => d.color || '#ffffff')
                .linkWidth(link => {
                    if (link.weight) return Math.max(1, link.weight * 2.5);
                    if (link.all_weights && Object.keys(link.all_weights).length > 0) {
                        const values = Object.values(link.all_weights);
                        const avg = values.reduce((a, b) => a + b, 0) / values.length;
                        return Math.max(1, avg * 2.5);
                    }
                    return 1;
                })
                .linkColor(() => 'rgba(220,220,220,0.8)')
                .linkLabel(link => {
                    let label = link.relation || '';
                    if (link.all_weights && Object.keys(link.all_weights).length > 1) {
                        label += ` (${Object.keys(link.all_weights).length} weights)`;
                    } else if (link.weight != null) {
                        label += ` (${link.weight})`;
                    } else if (link.all_weights && Object.keys(link.all_weights).length === 1) {
                        const singleWeight = Object.values(link.all_weights)[0];
                        label += ` (${singleWeight})`;
                    }
                    return label;
                })
                .backgroundColor('rgba(0,0,0,0)');

            if (!IS_3D) {
                // 2D-specific tweaks
                Graph.nodeRelSize(6);

                // Draw node and link labels directly on the 2D canvas
                Graph
                    .nodeCanvasObjectMode(() => 'after')
                    .nodeCanvasObject((node, ctx, globalScale) => {
                        const label = node.name || node.id;
                        if (!label) return;
                        const fontSize = 10 / globalScale;
                        ctx.font = `${fontSize}px Sans-Serif`;
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillStyle = 'white';
                        const offset = 10 / globalScale;
                        ctx.fillText(label, node.x, node.y - offset);
                    })
                    .linkCanvasObjectMode(() => 'after')
                    .linkCanvasObject((link, ctx, globalScale) => {
                        const label = link.relation;
                        if (!label || !link.source || !link.target) return;
                        const start = link.source;
                        const end = link.target;
                        const textPos = {
                            x: (start.x + end.x) / 2,
                            y: (start.y + end.y) / 2,
                        };
                        const fontSize = 8 / globalScale;
                        ctx.font = `${fontSize}px Sans-Serif`;
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillStyle = 'rgba(255,255,255,0.8)';
                        ctx.fillText(label, textPos.x, textPos.y);
                    });
            } else {
                // 3D-specific tweaks
                Graph.nodeRelSize(4);
                Graph.linkOpacity(0.6);

                // Use 3D sprite text for link labels (text-links example style)
                if (typeof SpriteText !== 'undefined') {
                    Graph
                        .linkThreeObjectExtend(true)
                        .linkThreeObject(link => {
                            const label = link.relation;
                            if (!label) return undefined;
                            const sprite = new SpriteText(label);
                            sprite.color = 'rgba(230,230,230,0.9)';
                            sprite.textHeight = 3;
                            return sprite;
                        })
                        .linkPositionUpdate((sprite, { start, end }) => {
                            if (!sprite) return;
                            const middlePos = {
                                x: start.x + (end.x - start.x) / 2,
                                y: start.y + (end.y - start.y) / 2,
                                z: start.z + (end.z - start.z) / 2,
                            };
                            Object.assign(sprite.position, middlePos);
                        });
                }
            }

            // Keep a simple tooltip that shows rich information similar to the previous D3 version
            let lastMouseX = 0;
            let lastMouseY = 0;

            graphElem.addEventListener('mousemove', evt => {
                lastMouseX = evt.clientX;
                lastMouseY = evt.clientY;
                if (tooltip.style.opacity === '1') {
                    tooltip.style.left = (lastMouseX + 10) + 'px';
                    tooltip.style.top = (lastMouseY - 10) + 'px';
                }
            });

            function createTreeHTML(obj, depth = 0) {
                const INDENT = '&nbsp;'.repeat(depth * 4);
                let html = '';

                Object.keys(obj).forEach(key => {
                    if (['color', 'index', 'id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz'].includes(key)) {
                        return;
                    }
                    const value = obj[key];
                    if (value == null) {
                        return;
                    }

                    if (typeof value === 'object' && !Array.isArray(value)) {
                        html += `${INDENT}<strong>${key}:</strong><br/>`;
                        html += createTreeHTML(value, depth + 1);
                    } else if (Array.isArray(value)) {
                        html += `${INDENT}<strong>${key}:</strong> [${value.length} items]<br/>`;
                        value.forEach((item, idx) => {
                            if (typeof item === 'object') {
                                html += `${INDENT}&nbsp;&nbsp;[${idx}]:<br/>`;
                                html += createTreeHTML(item, depth + 2);
                            } else {
                                html += `${INDENT}&nbsp;&nbsp;[${idx}]: ${item}<br/>`;
                            }
                        });
                    } else {
                        let displayValue = String(value);
                        if (displayValue.length > 100) {
                            displayValue = displayValue.substring(0, 100) + '...';
                        }
                        html += `${INDENT}<strong>${key}:</strong> ${displayValue}<br/>`;
                    }
                });

                return html;
            }

            function showTooltip(content) {
                tooltip.innerHTML = content;
                tooltip.style.left = (lastMouseX + 10) + 'px';
                tooltip.style.top = (lastMouseY - 10) + 'px';
                tooltip.style.opacity = 1;
            }

            function hideTooltip() {
                tooltip.style.opacity = 0;
            }

            Graph
                .onNodeHover(node => {
                    if (!node) {
                        hideTooltip();
                        return;
                    }
                    let content = `<strong style='font-size: 9px;'>${node.type || 'Node'}</strong><br/><br/>`;
                    content += createTreeHTML(node);
                    showTooltip(content);
                })
                .onLinkHover(link => {
                    if (!link) {
                        hideTooltip();
                        return;
                    }
                    let content = '<strong>Edge Information</strong><br/>';
                    content += `Relationship: ${link.relation || ''}<br/>`;

                    if (link.all_weights && Object.keys(link.all_weights).length > 0) {
                        content += '<strong>Weights:</strong><br/>';
                        Object.keys(link.all_weights).forEach(name => {
                            content += `&nbsp;&nbsp;${name}: ${link.all_weights[name]}<br/>`;
                        });
                    } else if (link.weight != null) {
                        content += `Weight: ${link.weight}<br/>`;
                    }

                    if (link.relationship_type) {
                        content += `Type: ${link.relationship_type}<br/>`;
                    }

                    if (link.edge_info) {
                        Object.keys(link.edge_info).forEach(key => {
                            if (
                                key === 'weight' ||
                                key === 'weights' ||
                                key === 'relationship_type' ||
                                key === 'source_node_id' ||
                                key === 'target_node_id' ||
                                key === 'relationship_name' ||
                                key === 'updated_at' ||
                                key.startsWith('weight_')
                            ) {
                                return;
                            }
                            content += `${key}: ${link.edge_info[key]}<br/>`;
                        });
                    }

                    showTooltip(content);
                });

            window.addEventListener('resize', () => {
                Graph.width(window.innerWidth);
                Graph.height(window.innerHeight);
            });
        </script>
    </body>
    </html>
    """

    html_content = html_template.replace("__IS_3D__", "true" if use_3d else "false")
    html_content = html_content.replace("{nodes}", json.dumps(nodes_list))
    html_content = html_content.replace("{links}", json.dumps(links_list))
    return html_content


def generate_html(
    connection: GraphBackend,
    destination_file_path: str | None = None,
    node_configs: list | None = None,
    relation_configs: list | None = None,
    custom_colors: dict[str, str] | None = None,
    use_3d: bool = True,
) -> str:
    """Generate an HTML graph visualization from a graph connection/backend.

    Args:
        connection: Object exposing an execute() method (e.g. GraphBackend) connected to a database that uses
            a schema with Node(id, name, type, properties) and EDGE(relationship_name, properties).
        destination_file_path: Optional path to write the HTML file. If omitted,
            the file will be saved as "graph_visualization.html" in the user's home directory.
        node_configs: Optional list of node configurations (legacy or new format)
        relation_configs: Optional list of relation configurations (legacy or new format)
        custom_colors: Optional mapping of node types to hex color codes

    Returns:
        The HTML content as a string.
    """
    nodes_data, edges_data = _fetch_graph_data(connection, node_configs, relation_configs)

    # Build visualization model using generic color assignment

    nodes_list: list[dict[str, Any]] = []
    for node_id, node_info in nodes_data:
        node_info = dict(node_info)  # shallow copy
        node_info["id"] = str(node_id)
        node_type = node_info.get("type", "Unknown")
        node_info["color"] = _get_node_color(node_type, custom_colors)
        node_info["name"] = node_info.get("name", str(node_id))
        # Trim noisy timestamp fields if present
        node_info.pop("updated_at", None)
        node_info.pop("created_at", None)
        nodes_list.append(node_info)

    links_list: list[dict[str, Any]] = []
    for source, target, relation, edge_info in edges_data:
        source_s = str(source)
        target_s = str(target)

        # Extract weight variations
        all_weights: dict[str, float] = {}
        primary_weight: float | None = None
        edge_info = edge_info or {}

        if "weight" in edge_info:
            try:
                primary_weight = float(edge_info["weight"])  # best effort
                all_weights["default"] = primary_weight
            except (TypeError, ValueError):
                pass

        if "weights" in edge_info and isinstance(edge_info["weights"], dict):
            for k, v in edge_info["weights"].items():
                try:
                    all_weights[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            if primary_weight is None and all_weights:
                primary_weight = next(iter(all_weights.values()))

        for key, value in edge_info.items():
            if key.startswith("weight_"):
                try:
                    all_weights[key[7:]] = float(value)
                except (TypeError, ValueError):
                    continue

        links_list.append(
            {
                "source": source_s,
                "target": target_s,
                "relation": relation,
                "weight": primary_weight,
                "all_weights": all_weights,
                "relationship_type": edge_info.get("relationship_type"),
                "edge_info": edge_info,
            }
        )

    html_content = _build_force_graph_html_content(
        nodes_list=nodes_list,
        links_list=links_list,
        use_3d=use_3d,
    )

    if not destination_file_path:
        home_dir = os.path.expanduser("~")
        destination_file_path = os.path.join(home_dir, "graph_visualization.html")

    os.makedirs(os.path.dirname(destination_file_path) or ".", exist_ok=True)
    with open(destination_file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_content


class KnowledgeGraphHTMLVisualizer:
    """Class-based wrapper for HTML visualization functionality."""

    def __init__(self, custom_colors: dict[str, str] | None = None, use_3d: bool = True) -> None:
        """Initialize the visualizer.

        Args:
            custom_colors: Optional custom color mapping for node types
            use_3d: Whether to render the visualization in 3D or 2D
        """
        self.custom_colors = custom_colors or {}
        self.use_3d = use_3d

    def generate_html(self, nodes: list[tuple[str, dict]], links: list[tuple[str, str, str, dict]]) -> str:
        """Generate HTML visualization from node and link data.

        Args:
            nodes: List of (node_id, properties_dict) tuples
            links: List of (source_id, target_id, relationship_name, properties_dict) tuples

        Returns:
            HTML content as string
        """
        # Convert to the format expected by the HTML template
        nodes_list = []
        for node_id, node_info in nodes:
            node_info = dict(node_info)  # shallow copy
            node_info["id"] = str(node_id)
            node_type = node_info.get("type", "Unknown")
            node_info["color"] = _get_node_color(node_type, self.custom_colors)
            node_info["name"] = node_info.get("name", str(node_id))
            # Trim noisy timestamp fields if present
            node_info.pop("updated_at", None)
            node_info.pop("created_at", None)
            nodes_list.append(node_info)

        links_list = []
        for source, target, relation, edge_info in links:
            source_s = str(source)
            target_s = str(target)

            # Extract weight variations
            all_weights = {}
            primary_weight = None
            edge_info = edge_info or {}

            if "weight" in edge_info:
                try:
                    primary_weight = float(edge_info["weight"])
                    all_weights["default"] = primary_weight
                except (TypeError, ValueError):
                    pass

            if "weights" in edge_info and isinstance(edge_info["weights"], dict):
                for k, v in edge_info["weights"].items():
                    try:
                        all_weights[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                if primary_weight is None and all_weights:
                    primary_weight = next(iter(all_weights.values()))

            for key, value in edge_info.items():
                if key.startswith("weight_"):
                    try:
                        all_weights[key[7:]] = float(value)
                    except (TypeError, ValueError):
                        continue

            links_list.append(
                {
                    "source": source_s,
                    "target": target_s,
                    "relation": relation,
                    "weight": primary_weight,
                    "all_weights": all_weights,
                    "relationship_type": edge_info.get("relationship_type"),
                    "edge_info": edge_info,
                }
            )

        return _build_force_graph_html_content(
            nodes_list=nodes_list,
            links_list=links_list,
            use_3d=self.use_3d,
        )
