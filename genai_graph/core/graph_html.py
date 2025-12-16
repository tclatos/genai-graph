"""Generate an interactive HTML visualization of a Cypher graph

This module  builds a simple JSON model, and embeds it in
an HTML page rendered with D3 force-directed layout.

It's inspired from code in Cognee.

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
from genai_graph.core.graph_core import NodeRecord, RelationshipRecord

# Import new schema types

# from genai_graph.demos.ekg.graph_schema import GraphNode, GraphRelationConfig, GraphSchema


def _get_node_raw_name(node_dict: dict[str, Any], node_type: str) -> str:
    """Extract the raw name for a node without truncation (for ID generation).

    Args:
        node_dict: Node properties dictionary
        node_type: The node type/table name

    Returns:
        Raw name for the node (no truncation)
    """
    # PRIORITY 1: Check for the 'name' field (user-chosen node name)
    if "name" in node_dict and node_dict["name"] is not None:
        value = str(node_dict["name"]).strip()
        if value:
            return value

    # PRIORITY 2: Common name fields to check in order of preference
    name_fields = ["title", "description", "label", "_original_name", "id"]

    for field in name_fields:
        if field in node_dict and node_dict[field] is not None:
            value = str(node_dict[field]).strip()
            if value:
                return value

    # PRIORITY 3: If no name field found, use the first non-empty string field
    for key, value in node_dict.items():
        if isinstance(value, str) and value.strip() and key not in ["type", "id", "name"]:
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


def _serialize_kuzu_id(kuzu_id: Any) -> str:
    """Serialize a Kuzu internal ID to a consistent string format.

    Kuzu IDs can be dicts like {'offset': 0, 'table': 0} or simple values.
    This ensures we get a consistent string representation.

    Args:
        kuzu_id: The Kuzu internal ID (dict or other)

    Returns:
        A consistent string representation
    """
    if isinstance(kuzu_id, dict):
        # Kuzu returns IDs as {'offset': int, 'table': int}
        table = kuzu_id.get("table", 0)
        offset = kuzu_id.get("offset", 0)
        return f"{table}:{offset}"
    return str(kuzu_id)


def _generate_html_content(nodes_list: list[dict[str, Any]], links_list: list[dict[str, Any]]) -> str:
    """Generate HTML content from nodes and links lists.

    This is a helper function to avoid code duplication between generate_html
    and generate_html_from_cypher.

    Args:
        nodes_list: List of node dictionaries with id, name, type, color, and properties
        links_list: List of link dictionaries with source, target, relation, weight info

    Returns:
        HTML content as a string with embedded D3.js visualization
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://d3js.org/d3.v5.min.js"></script>
        <style>
            body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: linear-gradient(90deg, #101010, #1a1a2e); color: white; font-family: 'Inter', sans-serif; }

            svg { width: 100vw; height: 100vh; display: block; }
            .links line { stroke: rgba(255, 255, 255, 0.4); stroke-width: 2px; }
            .links line.weighted { stroke: rgba(255, 215, 0, 0.7); }
            .links line.multi-weighted { stroke: rgba(0, 255, 127, 0.8); }
            .nodes circle { stroke: white; stroke-width: 0.5px; filter: drop-shadow(0 0 5px rgba(255,255,255,0.3)); }
            .node-label { font-size: 8px; font-weight: bold; fill: white; text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', sans-serif; pointer-events: none; }
            .edge-label { font-size: 3px; fill: rgba(255, 255, 255, 0.7); text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', sans-serif; pointer-events: none; }
            
            .tooltip {
                position: absolute;
                text-align: left;
                padding: 8px;
                font-size: 8px;
                background: rgba(0, 0, 0, 0.95);
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
                line-height: 1.4;
            }
            
            /* Larger tooltips when embedded in iframe (Streamlit) */
            body.in-iframe .tooltip {
                padding: 12px;
                font-size: 14px;
            }
            
            .zoom-controls {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 8px;
                padding: 10px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                z-index: 1000;
            }
            
            .zoom-btn {
                width: 36px;
                height: 36px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                color: white;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .zoom-btn:hover {
                background: rgba(255, 255, 255, 0.2);
                border-color: rgba(255, 255, 255, 0.5);
                transform: scale(1.05);
            }
            
            .zoom-btn:active {
                transform: scale(0.95);
            }
        </style>
    </head>
    <body>
        <svg></svg>
        <div class="tooltip" id="tooltip"></div>
        <div class="zoom-controls">
            <button class="zoom-btn" id="zoom-in" title="Zoom In">+</button>
            <button class="zoom-btn" id="zoom-out" title="Zoom Out">−</button>
            <button class="zoom-btn" id="zoom-reset" title="Fit All" style="font-size: 14px;">⊡</button>
        </div>
        <script>
            // Detect if running in iframe (Streamlit) and adjust styles
            var inIframe = window.self !== window.top;
            if (inIframe) {
                document.body.classList.add('in-iframe');
            }
            
            var nodes = {nodes};
            var links = {links};

            var svg = d3.select("svg"),
                width = window.innerWidth,
                height = window.innerHeight;

            var container = svg.append("g");
            var tooltip = d3.select("#tooltip");

            var simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).strength(0.1))
                .force("charge", d3.forceManyBody().strength(-275))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("x", d3.forceX().strength(0.1).x(width / 2))
                .force("y", d3.forceY().strength(0.1).y(height / 2));

            var link = container.append("g")
                .attr("class", "links")
                .selectAll("line")
                .data(links)
                .enter().append("line")
                .attr("stroke-width", d => {
                    if (d.weight) return Math.max(2, d.weight * 5);
                    if (d.all_weights && Object.keys(d.all_weights).length > 0) {
                        var avgWeight = Object.values(d.all_weights).reduce((a, b) => a + b, 0) / Object.values(d.all_weights).length;
                        return Math.max(2, avgWeight * 5);
                    }
                    return 2;
                })
                .attr("class", d => {
                    if (d.all_weights && Object.keys(d.all_weights).length > 1) return "multi-weighted";
                    if (d.weight || (d.all_weights && Object.keys(d.all_weights).length > 0)) return "weighted";
                    return "";
                })
                .on("mouseover", function(d) {
                    // Create tooltip content for edge
                    var content = "<strong>Edge Information</strong><br/>";
                    content += "Relationship: " + d.relation + "<br/>";

                    // Show all weights
                    if (d.all_weights && Object.keys(d.all_weights).length > 0) {
                        content += "<strong>Weights:</strong><br/>";
                        Object.keys(d.all_weights).forEach(function(weightName) {
                            content += "&nbsp;&nbsp;" + weightName + ": " + d.all_weights[weightName] + "<br/>";
                        });
                    } else if (d.weight !== null && d.weight !== undefined) {
                        content += "Weight: " + d.weight + "<br/>";
                    }

                    if (d.relationship_type) {
                        content += "Type: " + d.relationship_type + "<br/>";
                    }
                    // Add other edge properties
                    if (d.edge_info) {
                        Object.keys(d.edge_info).forEach(function(key) {
                            if (key !== 'weight' && key !== 'weights' && key !== 'relationship_type' && 
                                key !== 'source_node_id' && key !== 'target_node_id' && 
                                key !== 'relationship_name' && key !== 'updated_at' && 
                                !key.startsWith('weight_')) {
                                content += key + ": " + d.edge_info[key] + "<br/>";
                            }
                        });
                    }

                    tooltip.html(content)
                        .style("left", (d3.event.pageX + 10) + "px")
                        .style("top", (d3.event.pageY - 10) + "px")
                        .style("opacity", 1);
                })
                .on("mouseout", function(d) {
                    tooltip.style("opacity", 0);
                });

            var edgeLabels = container.append("g")
                .attr("class", "edge-labels")
                .selectAll("text")
                .data(links)
                .enter().append("text")
                .attr("class", "edge-label")
                .text(d => {
                    var label = d.relation;
                    if (d.all_weights && Object.keys(d.all_weights).length > 1) {
                        // Show count of weights for multiple weights
                        label += " (" + Object.keys(d.all_weights).length + " weights)";
                    } else if (d.weight) {
                        label += " (" + d.weight + ")";
                    } else if (d.all_weights && Object.keys(d.all_weights).length === 1) {
                        var singleWeight = Object.values(d.all_weights)[0];
                        label += " (" + singleWeight + ")";
                    }
                    return label;
                });

            var nodeGroup = container.append("g")
                .attr("class", "nodes")
                .selectAll("g")
                .data(nodes)
                .enter().append("g");

            var node = nodeGroup.append("circle")
                .attr("r", 13)
                .attr("fill", d => d.color)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            nodeGroup.append("text")
                .attr("class", "node-label")
                .attr("dy", 4)
                .attr("text-anchor", "middle")
                .text(d => d.name);

            nodeGroup.on("mouseover", function(d) {
                // Helper function to create tree-like HTML representation
                function createTreeHTML(obj, indent = 0) {
                    var html = "";
                    var indentStr = "&nbsp;".repeat(indent * 4);
                    
                    for (var key in obj) {
                        // Filter out unwanted properties
                        if (key === 'color' || key === 'index' || key === 'id' || 
                            key === 'x' || key === 'y' || key === 'vx' || key === 'vy' || 
                            key === 'fx' || key === 'fy') {
                            continue;
                        }
                        
                        var value = obj[key];
                        
                        if (value === null || value === undefined) {
                            continue;
                        }
                        
                        if (typeof value === 'object' && !Array.isArray(value)) {
                            // Nested object
                            html += indentStr + "<strong>" + key + ":</strong><br/>";
                            html += createTreeHTML(value, indent + 1);
                        } else if (Array.isArray(value)) {
                            // Array
                            html += indentStr + "<strong>" + key + ":</strong> [" + value.length + " items]<br/>";
                            value.forEach(function(item, idx) {
                                if (typeof item === 'object') {
                                    html += indentStr + "&nbsp;&nbsp;[" + idx + "]:<br/>";
                                    html += createTreeHTML(item, indent + 2);
                                } else {
                                    html += indentStr + "&nbsp;&nbsp;[" + idx + "]: " + item + "<br/>";
                                }
                            });
                        } else {
                            // Simple value
                            var displayValue = String(value);
                            if (displayValue.length > 100) {
                                displayValue = displayValue.substring(0, 100) + "...";
                            }
                            html += indentStr + "<strong>" + key + ":</strong> " + displayValue + "<br/>";
                        }
                    }
                    
                    return html;
                }
                
                var titleFontSize = inIframe ? '16px' : '10px';
                var content = "<strong style='font-size: " + titleFontSize + ";'>" + d.type + "</strong><br/><br/>";
                content += createTreeHTML(d);
                
                tooltip.html(content)
                    .style("left", (d3.event.pageX + 10) + "px")
                    .style("top", (d3.event.pageY - 10) + "px")
                    .style("opacity", 1);
            })
            .on("mouseout", function(d) {
                tooltip.style("opacity", 0);
            });
            
            simulation.on("tick", function() {
                link.attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                edgeLabels
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2 - 5);

                nodeGroup.attr("transform", d => "translate(" + d.x + "," + d.y + ")");
            });

            var zoom = d3.zoom().on("zoom", function() {
                container.attr("transform", d3.event.transform);
            });
            
            svg.call(zoom);
            
            // Zoom control buttons
            d3.select("#zoom-in").on("click", function() {
                svg.transition().duration(300).call(zoom.scaleBy, 1.3);
            });
            
            d3.select("#zoom-out").on("click", function() {
                svg.transition().duration(300).call(zoom.scaleBy, 0.7);
            });
            
            d3.select("#zoom-reset").on("click", function() {
                // Calculate bounds of all nodes
                var minX = d3.min(nodes, d => d.x);
                var maxX = d3.max(nodes, d => d.x);
                var minY = d3.min(nodes, d => d.y);
                var maxY = d3.max(nodes, d => d.y);
                
                var graphWidth = maxX - minX;
                var graphHeight = maxY - minY;
                var centerX = (minX + maxX) / 2;
                var centerY = (minY + maxY) / 2;
                
                // Calculate scale to fit with padding
                var padding = 100;
                var scaleX = (width - padding * 2) / graphWidth;
                var scaleY = (height - padding * 2) / graphHeight;
                var scale = Math.min(scaleX, scaleY, 1); // Don't zoom in past 1x
                
                // Calculate translation to center
                var translateX = width / 2 - centerX * scale;
                var translateY = height / 2 - centerY * scale;
                
                svg.transition().duration(750).call(
                    zoom.transform,
                    d3.zoomIdentity.translate(translateX, translateY).scale(scale)
                );
            });

            function dragstarted(d) {
                if (!d3.event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(d) {
                d.fx = d3.event.x;
                d.fy = d3.event.y;
            }

            function dragended(d) {
                if (!d3.event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }

            window.addEventListener("resize", function() {
                width = window.innerWidth;
                height = window.innerHeight;
                svg.attr("width", width).attr("height", height);
                simulation.force("center", d3.forceCenter(width / 2, height / 2));
                simulation.alpha(1).restart();
            });
        </script>
    </body>
    </html>
    """

    html_content = html_template.replace("{nodes}", json.dumps(nodes_list))
    html_content = html_content.replace("{links}", json.dumps(links_list))

    return html_content


def _fetch_graph_data(
    connection: GraphBackend,
    node_configs: list | None = None,
    relation_configs: list | None = None,
) -> tuple[list[NodeRecord], list[RelationshipRecord]]:
    """Fetch all nodes and edges from the graph database via the provided connection/backend.

    Args:
        connection: Object exposing an execute() method (e.g. GraphBackend or kuzu.Connection)
        node_configs: Optional list of node configurations (legacy or new format)
        relation_configs: Optional list of relation configurations (legacy or new format)

    Returns:
        Tuple of (nodes, relationships) where:
        - nodes: list of NodeRecord instances
        - relationships: list of RelationshipRecord instances
    """
    nodes: list[NodeRecord] = []
    relationships: list[RelationshipRecord] = []

    # Optional filtering based on provided node / relation configs
    allowed_node_labels: set[str] | None = None
    allowed_rel_types: set[str] | None = None

    if node_configs:
        try:
            labels: set[str] = set()
            for cfg in node_configs:
                node_class = getattr(cfg, "node_class", None)
                if node_class is not None and hasattr(node_class, "__name__"):
                    labels.add(node_class.__name__)
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

    # No special-case Document/SOURCE provenance nodes handled here anymore.

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
                            # Keep metadata fields like _created_at, _updated_at, _original_name
                            # but skip Kuzu internal fields like _id, _label
                            if key in ("_created_at", "_updated_at", "_original_name"):
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
                    # Use Kuzu internal ID for perfect consistency
                    kuzu_id = None
                    if isinstance(node_obj, dict) and "_id" in node_obj:
                        kuzu_id = _serialize_kuzu_id(node_obj["_id"])

                    if kuzu_id:
                        uuid_to_node_data[kuzu_id] = {"uuid": node_uuid, "type": table_name, "node_dict": node_dict}

                    nodes.append(NodeRecord(node_id=node_uuid, properties=node_dict))

            except Exception as e:
                print(f"Error fetching nodes from {table_name}: {e}")
                continue

        # If no nodes, return early
        if not nodes:
            return [], []

        # UUID mapping complete

        # Fetch relationships using explicit queries
        try:
            # Try to get all relationships - basic Kuzu syntax
            rel_query = "MATCH (n)-[r]->(m) RETURN n, r, m"
            rel_result = connection.execute(rel_query)
            rel_df = rel_result.get_as_df()

            for _, row in rel_df.iterrows():
                # Extract source and destination node data
                src_node = row["n"]
                dst_node = row["m"]
                rel_obj = row["r"]

                # Get relationship type from relationship object (Kuzu returns dict)
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

                # Extract Kuzu internal IDs for perfect matching
                src_kuzu_id = None
                dst_kuzu_id = None

                if isinstance(src_node, dict) and "_id" in src_node:
                    src_kuzu_id = _serialize_kuzu_id(src_node["_id"])
                if isinstance(dst_node, dict) and "_id" in dst_node:
                    dst_kuzu_id = _serialize_kuzu_id(dst_node["_id"])

                src_uuid = None
                dst_uuid = None

                if src_kuzu_id and src_kuzu_id in uuid_to_node_data:
                    src_uuid = uuid_to_node_data[src_kuzu_id]["uuid"]

                if dst_kuzu_id and dst_kuzu_id in uuid_to_node_data:
                    dst_uuid = uuid_to_node_data[dst_kuzu_id]["uuid"]

                # Only add if we have valid UUIDs for both nodes
                if src_uuid and dst_uuid:
                    # Extract node types for RelationshipRecord
                    src_type = uuid_to_node_data[src_kuzu_id]["type"] if src_kuzu_id in uuid_to_node_data else "Unknown"
                    dst_type = uuid_to_node_data[dst_kuzu_id]["type"] if dst_kuzu_id in uuid_to_node_data else "Unknown"

                    relationships.append(
                        RelationshipRecord(
                            from_type=src_type,
                            from_id=src_uuid,
                            to_type=dst_type,
                            to_id=dst_uuid,
                            name=rel_type,
                            properties=edge_props,
                        )
                    )

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

    return nodes, relationships


def generate_html(
    connection: GraphBackend,
    destination_file_path: str | None = None,
    node_configs: list | None = None,
    relation_configs: list | None = None,
    custom_colors: dict[str, str] | None = None,
) -> str:
    """Generate an HTML graph visualization from a graph connection/backend.

    Args:
        connection: Object exposing an execute() method (e.g. GraphBackend or kuzu.Connection) connected to a database that uses
            a schema with Node(id, name, type, properties) and EDGE(relationship_name, properties).
        destination_file_path: Optional path to write the HTML file. If omitted,
            the file will be saved as "graph_visualization.html" in the user's home directory.
        node_configs: Optional list of node configurations (legacy or new format)
        relation_configs: Optional list of relation configurations (legacy or new format)
        custom_colors: Optional mapping of node types to hex color codes

    Returns:
        The HTML content as a string.
    """
    nodes_data, relationships_data = _fetch_graph_data(connection, node_configs, relation_configs)

    # Build visualization model using generic color assignment

    nodes_list: list[dict[str, Any]] = []
    for node_record in nodes_data:
        node_info = dict(node_record.properties)  # shallow copy
        node_info["id"] = str(node_record.node_id)
        node_type = node_info.get("type", "Unknown")
        node_info["color"] = _get_node_color(node_type, custom_colors)
        node_info["name"] = node_info.get("name", str(node_record.node_id))
        # Trim noisy timestamp fields if present
        node_info.pop("updated_at", None)
        node_info.pop("created_at", None)
        nodes_list.append(node_info)

    links_list: list[dict[str, Any]] = []
    for rel_record in relationships_data:
        source_s = str(rel_record.from_id)
        target_s = str(rel_record.to_id)
        relation = rel_record.name
        edge_info = rel_record.properties or {}

        # Extract weight variations
        all_weights: dict[str, float] = {}
        primary_weight: float | None = None

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

    html_content = _generate_html_content(nodes_list, links_list)

    if not destination_file_path:
        home_dir = os.path.expanduser("~")
        destination_file_path = os.path.join(home_dir, "graph_visualization.html")

    os.makedirs(os.path.dirname(destination_file_path) or ".", exist_ok=True)
    with open(destination_file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_content


def generate_html_from_cypher(
    connection: GraphBackend,
    cypher_query: str,
    destination_file_path: str | None = None,
    custom_colors: dict[str, str] | None = None,
) -> str:
    """Generate an HTML graph visualization from a custom Cypher query.

    This function executes a Cypher query and visualizes the results as an
    interactive D3.js graph. It reuses the same data fetching logic as generate_html
    but with a custom query instead of fetching all data.

    Args:
        connection: GraphBackend instance connected to the database
        cypher_query: Cypher query to execute (should return nodes/relationships)
        destination_file_path: Optional path to write the HTML file
        custom_colors: Optional mapping of node types to hex color codes

    Returns:
        The HTML content as a string

    Example:
        ```
        backend = create_backend_from_config("default")
        html = generate_html_from_cypher(
            backend,
            "MATCH (n:Person)-[r]->(m) RETURN n, r, m LIMIT 50"
        )
        ```
    """
    # Use the same data fetching logic by creating a custom wrapper
    # that extracts nodes and relationships from the query result

    nodes: list[NodeRecord] = []
    relationships: list[RelationshipRecord] = []

    try:
        result = connection.execute(cypher_query)
        result_df = result.get_as_df()
    except Exception as e:
        raise RuntimeError(f"Failed to execute Cypher query: {e}") from e

    if result_df.empty:
        # Return a simple HTML with a message
        empty_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Empty Result</title></head>
        <body style="font-family: Arial; padding: 20px; background: #1a1a2e; color: white;">
            <h2>No results found</h2>
            <p>The query returned no data to visualize.</p>
        </body>
        </html>
        """
        return empty_html

    # Use the same ID mapping logic as _fetch_graph_data
    uuid_to_node_data = {}

    # Process each row in the result
    for _, row in result_df.iterrows():
        for col_name in result_df.columns:
            value = row[col_name]

            # Check if value is a node (dict with node properties)
            # IMPORTANT: Must check it's NOT a relationship first (relationships also have _id and _label)
            if (
                isinstance(value, dict)
                and "_id" in value
                and "_label" in value
                and "_src" not in value
                and "_dst" not in value
            ):
                # This is a node (not a relationship)
                node_obj = value
                node_label = value.get("_label", "Unknown")
                kuzu_id = _serialize_kuzu_id(node_obj["_id"])

                # Skip if we've already processed this node
                if kuzu_id in uuid_to_node_data:
                    continue

                # Extract properties (same logic as _fetch_graph_data)
                node_dict = {}
                for key, val in node_obj.items():
                    if key in ("_created_at", "_updated_at", "_original_name"):
                        node_dict[key] = str(val).strip() if str(val).strip() else str(val)
                    elif not key.startswith("_") and val is not None:
                        node_dict[key] = str(val).strip() if str(val).strip() else str(val)

                # Skip empty nodes
                if not node_dict:
                    continue

                # Generate display name and add node metadata
                node_name = _get_node_display_name(node_dict, node_label)
                node_dict["type"] = node_label
                node_dict["name"] = node_name

                # Generate UUID-based ID for absolute uniqueness and consistency
                node_uuid = str(uuid.uuid4())

                # Store mapping for relationship resolution
                uuid_to_node_data[kuzu_id] = {"uuid": node_uuid, "type": node_label, "node_dict": node_dict}

                nodes.append(NodeRecord(node_id=node_uuid, properties=node_dict))

    # Now process relationships
    for _, row in result_df.iterrows():
        for col_name in result_df.columns:
            value = row[col_name]

            # Check if it's a relationship
            if isinstance(value, dict) and "_src" in value and "_dst" in value and "_label" in value:
                rel_obj = value
                rel_type = rel_obj.get("_label", "RELATED_TO")

                # Extract Kuzu internal IDs for matching
                src_kuzu_id = _serialize_kuzu_id(rel_obj["_src"])
                dst_kuzu_id = _serialize_kuzu_id(rel_obj["_dst"])

                # Get UUIDs from our mapping
                if src_kuzu_id not in uuid_to_node_data or dst_kuzu_id not in uuid_to_node_data:
                    # Skip relationships where nodes aren't in our result set
                    continue

                src_uuid = uuid_to_node_data[src_kuzu_id]["uuid"]
                dst_uuid = uuid_to_node_data[dst_kuzu_id]["uuid"]
                src_type = uuid_to_node_data[src_kuzu_id]["type"]
                dst_type = uuid_to_node_data[dst_kuzu_id]["type"]

                # Extract edge properties (non-internal fields)
                edge_props = {}
                for key, value_prop in rel_obj.items():
                    if not key.startswith("_") and value_prop is not None:
                        edge_props[key] = value_prop

                relationships.append(
                    RelationshipRecord(
                        from_type=src_type,
                        from_id=src_uuid,
                        to_type=dst_type,
                        to_id=dst_uuid,
                        name=rel_type,
                        properties=edge_props,
                    )
                )

    # Now use the same HTML generation logic as generate_html
    # Build visualization model using generic color assignment
    nodes_list: list[dict[str, Any]] = []
    for node_record in nodes:
        node_info = dict(node_record.properties)  # shallow copy
        node_info["id"] = str(node_record.node_id)
        node_type = node_info.get("type", "Unknown")
        node_info["color"] = _get_node_color(node_type, custom_colors)
        node_info["name"] = node_info.get("name", str(node_record.node_id))
        # Trim noisy timestamp fields if present
        node_info.pop("updated_at", None)
        node_info.pop("created_at", None)
        nodes_list.append(node_info)

    links_list: list[dict[str, Any]] = []
    for rel_record in relationships:
        source_s = str(rel_record.from_id)
        target_s = str(rel_record.to_id)
        relation = rel_record.name
        edge_info = rel_record.properties or {}

        # Extract weight variations
        all_weights: dict[str, float] = {}
        primary_weight: float | None = None

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

    # Generate the HTML content using the exact same template as generate_html
    html_content = _generate_html_content(nodes_list, links_list)

    if destination_file_path:
        os.makedirs(os.path.dirname(destination_file_path) or ".", exist_ok=True)
        with open(destination_file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content


# Alias for backward compatibility
def generate_html_visualization(
    connection: GraphBackend,
    destination_file_path: str | None = None,
    title: str = "Knowledge Graph",
    node_configs: list | None = None,
    relation_configs: list | None = None,
    custom_colors: dict[str, str] | None = None,
) -> str:
    """Generate an HTML graph visualization from a graph connection/backend.

    Alias for generate_kuzu_graph_html for backward compatibility.

    Args:
        connection: Object exposing an execute() method (e.g. GraphBackend or kuzu.Connection)
        destination_file_path: Optional path to write the HTML file
        title: Title for the visualization (currently unused)
        node_configs: Optional list of node configurations
        relation_configs: Optional list of relation configurations
        custom_colors: Optional mapping of node types to hex color codes

    Returns:
        The HTML content as a string.
    """
    return generate_html(
        connection=connection,
        destination_file_path=destination_file_path,
        node_configs=node_configs,
        relation_configs=relation_configs,
        custom_colors=custom_colors,
    )
