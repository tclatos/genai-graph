"""HTML template for the DAG (d3-dag) graph visualization.

Mirrors the look-and-feel of :mod:`genai_graph.kg.export.html_template`
(dark theme, zoom controls, left details panel, right node-type legend with
click-to-highlight, hover tooltips) but renders the graph as a left-to-right
DAG using d3-dag's sugiyama layered layout with default algorithm parameters.

The same ``{nodes}``/``{links}`` JSON placeholders are used, so both renderers
share the model built by :mod:`genai_graph.kg.export._graph_model`.
"""

# Layout orientation. "TB" = top-to-bottom (roots at the top, default);
# "LR" = left-to-right (roots on the left).
DAG_ORIENTATION = "TB"

# Pixels per sugiyama layout unit. With the default nodeSize [1,1] + gap [1,1],
# adjacent layers and adjacent siblings are both 2 units apart.
PX_PER_LAYER_UNIT = 110
PX_PER_SIBLING_UNIT = 55

DAG_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://d3js.org/d3.v5.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3-dag@1.1.0/bundle/d3-dag.iife.min.js"></script>
    <script>
        window.addEventListener('load', function() {
            if (window.parent !== window) {
                window.parent.postMessage({type: 'streamlit:componentReady'}, '*');
            }
        });
    </script>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            min-height: 600px;
            overflow: hidden;
            background: linear-gradient(90deg, #101010, #1a1a2e);
            color: white;
            font-family: 'Inter', sans-serif;
        }

        svg { width: 100%; height: max(600px, 100vh); display: block; }

        /* Links/Edges */
        .links path { fill: none; stroke: rgba(255, 255, 255, 0.4); stroke-width: 2px; cursor: pointer; }
        .links path.weighted { stroke: rgba(255, 215, 0, 0.7); }
        .links path.multi-weighted { stroke: rgba(0, 255, 127, 0.8); }
        .links path.selected { stroke: #fff !important; stroke-width: 4px !important; filter: drop-shadow(0 0 6px rgba(255,255,255,0.8)); }
        .links path.dimmed { opacity: 0.1; }

        /* Nodes */
        .nodes circle { stroke: white; stroke-width: 0.5px; filter: drop-shadow(0 0 5px rgba(255,255,255,0.3)); cursor: pointer; }
        .nodes circle.selected { stroke: #fff; stroke-width: 3px; filter: drop-shadow(0 0 10px rgba(255,255,255,0.8)); }
        .nodes circle.dimmed { opacity: 0.2; }

        /* Labels */
        .node-label { font-size: 8px; font-weight: bold; fill: white; text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', sans-serif; pointer-events: none; }
        .node-label.dimmed { opacity: 0.2; }
        .edge-label { font-size: 3px; fill: rgba(255, 255, 255, 0.7); text-anchor: middle; dominant-baseline: middle; font-family: 'Inter', sans-serif; pointer-events: none; }
        .edge-label.dimmed { opacity: 0.1; }

        /* Tooltip */
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

        body.in-iframe .tooltip { padding: 12px; font-size: 14px; }

        /* Details Panel - Left side */
        .details-panel {
            position: fixed;
            top: 20px;
            left: 20px;
            width: 320px;
            max-height: 80vh;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            padding: 16px;
            overflow-y: auto;
            z-index: 1000;
            font-size: 12px;
            line-height: 1.5;
            display: none;
        }
        .details-panel.visible { display: block; }
        .details-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        .details-panel-title { font-size: 14px; font-weight: bold; color: #fff; }
        .details-panel-close { background: none; border: none; color: rgba(255,255,255,0.6); font-size: 18px; cursor: pointer; padding: 0 4px; line-height: 1; }
        .details-panel-close:hover { color: #fff; }
        .details-content { color: rgba(255,255,255,0.9); }
        body.in-iframe .details-panel { font-size: 14px; width: 380px; }
        body.in-iframe .details-panel-title { font-size: 16px; }

        /* Legend Panel - Right side */
        .legend-panel {
            position: fixed;
            top: 180px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            padding: 12px;
            max-height: 50vh;
            overflow-y: auto;
            z-index: 1000;
            min-width: 140px;
        }
        .legend-title { font-size: 12px; font-weight: bold; margin-bottom: 10px; color: rgba(255,255,255,0.9); padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.2); }
        .legend-item { display: flex; align-items: center; margin-bottom: 6px; cursor: pointer; padding: 4px 6px; border-radius: 4px; transition: background 0.2s; }
        .legend-item:hover { background: rgba(255,255,255,0.1); }
        .legend-item.active { background: rgba(255,255,255,0.2); }
        .legend-color { width: 14px; height: 14px; border-radius: 50%; margin-right: 8px; border: 1px solid rgba(255,255,255,0.3); flex-shrink: 0; }
        .legend-label { font-size: 11px; color: rgba(255,255,255,0.8); }
        .legend-count { font-size: 10px; color: rgba(255,255,255,0.5); margin-left: auto; padding-left: 8px; }

        /* Zoom Controls */
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
            width: 36px; height: 36px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            color: white;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            display: flex; align-items: center; justify-content: center;
            transition: all 0.2s;
        }
        .zoom-btn:hover { background: rgba(255, 255, 255, 0.2); border-color: rgba(255, 255, 255, 0.5); transform: scale(1.05); }
        .zoom-btn:active { transform: scale(0.95); }

        /* Error / empty message */
        .message-banner {
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.9); border: 1px solid rgba(255,255,255,0.3);
            border-radius: 8px; padding: 20px 24px; max-width: 520px; text-align: center;
            font-size: 14px; color: rgba(255,255,255,0.9); z-index: 1001; display: none;
        }
        .message-banner.visible { display: block; }
    </style>
</head>
<body>
    <svg></svg>
    <div class="tooltip" id="tooltip"></div>

    <div class="details-panel" id="details-panel">
        <div class="details-panel-header">
            <div class="details-panel-title" id="details-title">Select an element</div>
            <button class="details-panel-close" id="details-close" title="Close">&times;</button>
        </div>
        <div class="details-content" id="details-content"></div>
    </div>

    <div class="legend-panel" id="legend-panel">
        <div class="legend-title">Node Types</div>
        <div id="legend-items"></div>
    </div>

    <div class="zoom-controls">
        <button class="zoom-btn" id="zoom-in" title="Zoom In">+</button>
        <button class="zoom-btn" id="zoom-out" title="Zoom Out">-</button>
        <button class="zoom-btn" id="zoom-reset" title="Fit All" style="font-size: 14px;">&#x229E;</button>
    </div>

    <div class="message-banner" id="message-banner"></div>

    <script>
        var inIframe = window.self !== window.top;
        if (inIframe) { document.body.classList.add('in-iframe'); }

        var nodes = {nodes};
        var links = {links};

        var ORIENTATION = "__ORIENTATION__";
        var PX_LAYER = __PX_LAYER__;
        var PX_SIBLING = __PX_SIBLING__;
        var NODE_R = 13;

        // Sugiyama lays out with the layer/rank axis on y (top-to-bottom) and the
        // within-layer axis on x. For "LR" we swap so the layer axis becomes
        // horizontal (roots on the left). Node coords are {x,y}; link.points are
        // [x,y] pairs, so read via pX/pY which handle both shapes.
        function pX(p) { return Array.isArray(p) ? p[0] : p.x; }
        function pY(p) { return Array.isArray(p) ? p[1] : p.y; }
        function sx(p) { return ORIENTATION === "LR" ? pY(p) * PX_LAYER : pX(p) * PX_SIBLING; }
        function sy(p) { return ORIENTATION === "LR" ? pX(p) * PX_SIBLING : pY(p) * PX_LAYER; }

        var svg = d3.select("svg"),
            width = window.innerWidth,
            height = window.innerHeight;
        svg.attr("width", width).attr("height", height);

        // arrowhead marker + root container
        var defs = svg.append("defs");
        defs.append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 0 10 10")
            .attr("refX", 9).attr("refY", 5)
            .attr("markerWidth", 6).attr("markerHeight", 6)
            .attr("orient", "auto-start-reverse")
            .append("path")
            .attr("d", "M 0 0 L 10 5 L 0 10 z")
            .attr("fill", "rgba(255,255,255,0.6)");

        var container = svg.append("g");
        var tooltip = d3.select("#tooltip");

        function showMessage(html) {
            d3.select("#message-banner").html(html).classed("visible", true);
        }

        // ========================================
        // Helper: tree-like HTML representation (shared with the force view)
        // ========================================
        function createTreeHTML(obj, indent) {
            indent = indent || 0;
            var html = "";
            var indentStr = "&nbsp;".repeat(indent * 4);
            for (var key in obj) {
                if (key === 'color' || key === 'index' || key === 'id' ||
                    key === 'x' || key === 'y' || key === 'vx' || key === 'vy' ||
                    key === 'fx' || key === 'fy' || key === 'parentIds' || key === '_children') {
                    continue;
                }
                var value = obj[key];
                if (value === null || value === undefined) { continue; }
                if (typeof value === 'object' && !Array.isArray(value)) {
                    html += indentStr + "<strong>" + key + ":</strong><br/>";
                    html += createTreeHTML(value, indent + 1);
                } else if (Array.isArray(value)) {
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
                    var displayValue = String(value);
                    if (displayValue.length > 100) { displayValue = displayValue.substring(0, 100) + "..."; }
                    html += indentStr + "<strong>" + key + ":</strong> " + displayValue + "<br/>";
                }
            }
            return html;
        }

        function generateEdgeContent(d) {
            var orig = d._orig || {};
            var content = "<strong>Relationship:</strong> " + (orig.relation || d.relation || "") + "<br/>";
            content += "<strong>From:</strong> " + (d.source.data.name || d.source.data.id) + "<br/>";
            content += "<strong>To:</strong> " + (d.target.data.name || d.target.data.id) + "<br/><br/>";
            if (orig.all_weights && Object.keys(orig.all_weights).length > 0) {
                content += "<strong>Weights:</strong><br/>";
                Object.keys(orig.all_weights).forEach(function(weightName) {
                    content += "&nbsp;&nbsp;" + weightName + ": " + orig.all_weights[weightName] + "<br/>";
                });
            } else if (orig.weight !== null && orig.weight !== undefined) {
                content += "<strong>Weight:</strong> " + orig.weight + "<br/>";
            }
            if (orig.relationship_type) {
                content += "<strong>Type:</strong> " + orig.relationship_type + "<br/>";
            }
            if (orig.edge_info) {
                var hasExtra = false;
                Object.keys(orig.edge_info).forEach(function(key) {
                    if (key !== 'weight' && key !== 'weights' && key !== 'relationship_type' &&
                        key !== 'source_node_id' && key !== 'target_node_id' &&
                        key !== 'relationship_name' && key !== 'updated_at' &&
                        !key.startsWith('weight_')) {
                        if (!hasExtra) { content += "<br/><strong>Properties:</strong><br/>"; hasExtra = true; }
                        content += "&nbsp;&nbsp;" + key + ": " + orig.edge_info[key] + "<br/>";
                    }
                });
            }
            return content;
        }

        function generateNodeContent(d) {
            var data = d.data || d;
            var titleFontSize = inIframe ? '16px' : '12px';
            var content = "<strong style='font-size: " + titleFontSize + ";'>" + (data.type || 'Node') + "</strong><br/><br/>";
            content += createTreeHTML(data);
            return content;
        }

        // ========================================
        // Build the DAG (left-to-right), roots = nodes with no incoming edge
        // ========================================
        if (!nodes.length) {
            showMessage("No nodes to display. The graph is empty.");
        } else {
            var stratifyData = nodes.map(function(n) {
                return Object.assign({}, n, { id: n.id, parentIds: [] });
            });
            var byId = {};
            stratifyData.forEach(function(d) { byId[d.id] = d; });
            // Roots (no incoming relationship) get an empty parentIds and so land
            // leftmost; each edge adds the source as one of the target's parents.
            links.forEach(function(l) {
                var tgt = byId[l.target];
                if (tgt && byId[l.source]) { tgt.parentIds.push(l.source); }
            });
            // Dedupe parentIds (graphStratify rejects duplicates).
            stratifyData.forEach(function(d) {
                var seen = {}; d.parentIds = d.parentIds.filter(function(p) {
                    return seen.hasOwnProperty(p) ? false : (seen[p] = true);
                });
            });

            var dag = null;
            try {
                var stratify = d3.graphStratify();
                dag = stratify(stratifyData);
                // sugiyama with default layering/decross/coord; nodeSize/gap stay default.
                var layout = d3.sugiyama();
                layout(dag);
            } catch (err) {
                showMessage("<strong>Could not lay out the graph as a DAG.</strong><br/><br/>" +
                    "d3-dag's sugiyama layout requires an acyclic graph. This graph contains a " +
                    "cycle, or the data could not be stratified.<br/><br/>" +
                    "Try the force-directed view instead.<br/><br/><code>" +
                    String(err && err.message ? err.message : err) + "</code>");
            }

            if (dag) {
              try {
                // Look up the original link by (source, target) id for edge details.
                var linkMap = {};
                links.forEach(function(l) {
                    var k = l.source + "::" + l.target;
                    if (!linkMap[k]) { linkMap[k] = l; }
                });

                // dag.nodes()/dag.links() return iterables, not arrays — convert with
                // Array.from so d3's .data() can bind them and we can index/re-iterate.
                var dagNodes = Array.from(dag.nodes());
                var dagLinks = Array.from(dag.links());
                dagNodes.forEach(function(n) {
                    n.screenX = sx(n);
                    n.screenY = sy(n);
                });
                dagLinks.forEach(function(l) {
                    l._orig = linkMap[(l.source.data.id || l.source.id) + "::" + (l.target.data.id || l.target.id)];
                    l.screenPoints = (l.points || []).map(function(p) {
                        return { x: sx(p), y: sy(p) };
                    });
                });

                // Shorten a polyline so it starts/ends at the node edge (for arrowheads).
                function shortenPath(pts, startPad, endPad) {
                    if (pts.length === 0) return pts;
                    var out = pts.map(function(p) { return { x: p.x, y: p.y }; });
                    if (out.length >= 2) {
                        var a = out[0], b = out[1];
                        var dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1;
                        out[0] = { x: a.x + dx / d * startPad, y: a.y + dy / d * startPad };
                        var y = out[out.length - 2], z = out[out.length - 1];
                        var dx2 = z.x - y.x, dy2 = z.y - y.y, d2 = Math.hypot(dx2, dy2) || 1;
                        out[out.length - 1] = { x: z.x - dx2 / d2 * endPad, y: z.y - dy2 / d2 * endPad };
                    }
                    return out;
                }
                function pathFromPoints(pts) {
                    return pts.map(function(p, i) { return (i ? "L" : "M") + p.x + " " + p.y; }).join(" ");
                }

                // Links
                var link = container.append("g")
                    .attr("class", "links")
                    .selectAll("path")
                    .data(dagLinks)
                    .enter().append("path")
                    .attr("stroke-width", function(d) {
                        var o = d._orig || {};
                        if (o.weight) return Math.max(2, o.weight * 5);
                        if (o.all_weights && Object.keys(o.all_weights).length > 0) {
                            var avg = Object.values(o.all_weights).reduce(function(a, b) { return a + b; }, 0) /
                                Object.values(o.all_weights).length;
                            return Math.max(2, avg * 5);
                        }
                        return 2;
                    })
                    .attr("class", function(d) {
                        var o = d._orig || {};
                        if (o.all_weights && Object.keys(o.all_weights).length > 1) return "multi-weighted";
                        if (o.weight || (o.all_weights && Object.keys(o.all_weights).length > 0)) return "weighted";
                        return "";
                    })
                    .attr("d", function(d) {
                        return pathFromPoints(shortenPath(d.screenPoints, NODE_R + 2, NODE_R + 6));
                    })
                    .attr("marker-end", "url(#arrowhead)")
                    .on("mouseover", function(d) {
                        tooltip.html(generateEdgeContent(d))
                            .style("left", (d3.event.pageX + 10) + "px")
                            .style("top", (d3.event.pageY - 10) + "px")
                            .style("opacity", 1);
                    })
                    .on("mouseout", function() { tooltip.style("opacity", 0); })
                    .on("click", function(d) { d3.event.stopPropagation(); selectEdge(d, this); });

                // Edge labels (relation name at the midpoint)
                var edgeLabels = container.append("g")
                    .attr("class", "edge-labels")
                    .selectAll("text")
                    .data(dagLinks)
                    .enter().append("text")
                    .attr("class", "edge-label")
                    .text(function(d) {
                        var o = d._orig || {};
                        var label = o.relation || d.relation || "";
                        if (o.all_weights && Object.keys(o.all_weights).length > 1) {
                            label += " (" + Object.keys(o.all_weights).length + " weights)";
                        } else if (o.weight) {
                            label += " (" + o.weight + ")";
                        } else if (o.all_weights && Object.keys(o.all_weights).length === 1) {
                            label += " (" + Object.values(o.all_weights)[0] + ")";
                        }
                        return label;
                    })
                    .attr("x", function(d) {
                        var pts = d.screenPoints;
                        if (!pts.length) return 0;
                        var mid = pts[Math.floor(pts.length / 2)];
                        return mid ? mid.x : (pts[0].x + pts[pts.length - 1].x) / 2;
                    })
                    .attr("y", function(d) {
                        var pts = d.screenPoints;
                        if (!pts.length) return 0;
                        var mid = pts[Math.floor(pts.length / 2)];
                        return mid ? mid.y - 5 : (pts[0].y + pts[pts.length - 1].y) / 2 - 5;
                    });

                // Nodes
                var nodeGroup = container.append("g")
                    .attr("class", "nodes")
                    .selectAll("g")
                    .data(dagNodes)
                    .enter().append("g")
                    .attr("transform", function(d) { return "translate(" + d.screenX + "," + d.screenY + ")"; });

                nodeGroup.append("circle")
                    .attr("r", NODE_R)
                    .attr("fill", function(d) { return d.data.color; });

                nodeGroup.append("text")
                    .attr("class", "node-label")
                    .attr("dy", 4)
                    .attr("text-anchor", "middle")
                    .text(function(d) { return d.data.name; });

                nodeGroup.on("mouseover", function(d) {
                        tooltip.html(generateNodeContent(d))
                            .style("left", (d3.event.pageX + 10) + "px")
                            .style("top", (d3.event.pageY - 10) + "px")
                            .style("opacity", 1);
                    })
                    .on("mouseout", function() { tooltip.style("opacity", 0); })
                    .on("click", function(d) { d3.event.stopPropagation(); selectNode(d, this); });

                // ========================================
                // Selection
                // ========================================
                function clearSelection() {
                    d3.selectAll('.nodes circle').classed('selected', false);
                    d3.selectAll('.links path').classed('selected', false);
                    document.getElementById('details-panel').classList.remove('visible');
                }
                function selectNode(d, element) {
                    clearSelection();
                    d3.select(element).select('circle').classed('selected', true);
                    document.getElementById('details-title').innerHTML = (d.data.type || 'Node') + ': ' + d.data.name;
                    document.getElementById('details-content').innerHTML = createTreeHTML(d.data);
                    document.getElementById('details-panel').classList.add('visible');
                }
                function selectEdge(d, element) {
                    clearSelection();
                    d3.select(element).classed('selected', true);
                    var o = d._orig || {};
                    document.getElementById('details-title').innerHTML = 'Edge: ' + (o.relation || d.relation || '');
                    document.getElementById('details-content').innerHTML = generateEdgeContent(d);
                    document.getElementById('details-panel').classList.add('visible');
                }
                document.getElementById('details-close').addEventListener('click', clearSelection);
                svg.on("click", function() {
                    if (d3.event.target.tagName === 'svg') { clearSelection(); }
                });

                // ========================================
                // Legend (click-to-highlight a node type)
                // ========================================
                var highlightedType = null;
                function generateLegend() {
                    var typeColorMap = {}, typeCounts = {};
                    dagNodes.forEach(function(n) {
                        var t = n.data.type;
                        if (!typeColorMap[t]) { typeColorMap[t] = n.data.color; typeCounts[t] = 0; }
                        typeCounts[t]++;
                    });
                    var types = Object.keys(typeColorMap).sort();
                    var legendHtml = '';
                    types.forEach(function(type) {
                        legendHtml += '<div class="legend-item" data-type="' + type + '">' +
                            '<div class="legend-color" style="background:' + typeColorMap[type] + '"></div>' +
                            '<span class="legend-label">' + type + '</span>' +
                            '<span class="legend-count">' + typeCounts[type] + '</span></div>';
                    });
                    document.getElementById('legend-items').innerHTML = legendHtml;
                    document.querySelectorAll('.legend-item').forEach(function(item) {
                        item.addEventListener('click', function() { toggleHighlightType(this.getAttribute('data-type')); });
                    });
                }
                function toggleHighlightType(type) {
                    if (highlightedType === type) {
                        highlightedType = null;
                        d3.selectAll('.legend-item').classed('active', false);
                        nodeGroup.select('circle').classed('dimmed', false);
                        nodeGroup.select('text').classed('dimmed', false);
                        link.classed('dimmed', false);
                        edgeLabels.classed('dimmed', false);
                    } else {
                        highlightedType = type;
                        d3.selectAll('.legend-item').classed('active', function() {
                            return this.getAttribute('data-type') === type;
                        });
                        nodeGroup.select('circle').classed('dimmed', function(d) { return d.data.type !== type; });
                        nodeGroup.select('text').classed('dimmed', function(d) { return d.data.type !== type; });
                        link.classed('dimmed', function(d) { return d.source.data.type !== type && d.target.data.type !== type; });
                        edgeLabels.classed('dimmed', function(d) { return d.source.data.type !== type && d.target.data.type !== type; });
                    }
                }
                generateLegend();

                // ========================================
                // Zoom controls + fit-to-all
                // ========================================
                var zoom = d3.zoom().on("zoom", function() { container.attr("transform", d3.event.transform); });
                svg.call(zoom);

                function graphBBox() {
                    var minX = d3.min(dagNodes, function(d) { return d.screenX; }) - NODE_R;
                    var maxX = d3.max(dagNodes, function(d) { return d.screenX; }) + NODE_R;
                    var minY = d3.min(dagNodes, function(d) { return d.screenY; }) - NODE_R;
                    var maxY = d3.max(dagNodes, function(d) { return d.screenY; }) + NODE_R;
                    return { minX: minX, maxX: maxX, minY: minY, maxY: maxY };
                }
                function fitAll() {
                    var b = graphBBox();
                    var gw = b.maxX - b.minX, gh = b.maxY - b.minY;
                    if (!isFinite(gw) || !isFinite(gh)) return;
                    // A zero-width/height (e.g. a linear chain) yields Infinity for that
                    // axis' scale; Math.min clamps it, so we still fit the other axis.
                    var padding = 100;
                    var scale = Math.min((width - padding * 2) / (gw || Infinity), (height - padding * 2) / (gh || Infinity), 1.2);
                    var cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2;
                    var tx = width / 2 - cx * scale, ty = height / 2 - cy * scale;
                    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
                }

                d3.select("#zoom-in").on("click", function() { svg.transition().duration(300).call(zoom.scaleBy, 1.3); });
                d3.select("#zoom-out").on("click", function() { svg.transition().duration(300).call(zoom.scaleBy, 0.7); });
                d3.select("#zoom-reset").on("click", fitAll);

                // Fit on first paint.
                fitAll();
              } catch (renderErr) {
                // Surface rendering errors on the page instead of failing silently.
                showMessage("<strong>DAG rendering error.</strong><br/><br/><code>" +
                    String(renderErr && renderErr.message ? renderErr.message : renderErr) + "</code>");
              }
            }
        }

        window.addEventListener("resize", function() {
            width = window.innerWidth;
            height = window.innerHeight;
            svg.attr("width", width).attr("height", height);
        });
    </script>
</body>
</html>
"""
