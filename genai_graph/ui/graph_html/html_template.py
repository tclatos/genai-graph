"""HTML template for graph visualization.

This module contains the HTML/CSS/JavaScript template used by graph_html.py
to generate interactive D3.js force-directed graph visualizations.
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://d3js.org/d3.v5.min.js"></script>
    <script>
        // Load Streamlit communication API for iframe communication
        window.addEventListener('load', function() {
            // Set iframe height
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
        .links line { stroke: rgba(255, 255, 255, 0.4); stroke-width: 2px; cursor: pointer; }
        .links line.weighted { stroke: rgba(255, 215, 0, 0.7); }
        .links line.multi-weighted { stroke: rgba(0, 255, 127, 0.8); }
        .links line.selected { stroke: #fff !important; stroke-width: 4px !important; filter: drop-shadow(0 0 6px rgba(255,255,255,0.8)); }
        .links line.dimmed { opacity: 0.1; }
        
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
        
        /* Larger tooltips when embedded in iframe (Streamlit) */
        body.in-iframe .tooltip {
            padding: 12px;
            font-size: 14px;
        }
        
        /* Details Panel - Left side for selected element info */
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
        
        .details-panel-title {
            font-size: 14px;
            font-weight: bold;
            color: #fff;
        }
        
        .details-panel-close {
            background: none;
            border: none;
            color: rgba(255,255,255,0.6);
            font-size: 18px;
            cursor: pointer;
            padding: 0 4px;
            line-height: 1;
        }
        
        .details-panel-close:hover { color: #fff; }
        
        .details-content {
            color: rgba(255,255,255,0.9);
        }
        
        /* Larger details panel when in iframe */
        body.in-iframe .details-panel {
            font-size: 14px;
            width: 380px;
        }
        
        body.in-iframe .details-panel-title {
            font-size: 16px;
        }
        
        /* Legend Panel - Right side for node types */
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
        
        .legend-title {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 10px;
            color: rgba(255,255,255,0.9);
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 6px;
            cursor: pointer;
            padding: 4px 6px;
            border-radius: 4px;
            transition: background 0.2s;
        }
        
        .legend-item:hover { background: rgba(255,255,255,0.1); }
        .legend-item.active { background: rgba(255,255,255,0.2); }
        
        .legend-color {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 8px;
            border: 1px solid rgba(255,255,255,0.3);
            flex-shrink: 0;
        }
        
        .legend-label {
            font-size: 11px;
            color: rgba(255,255,255,0.8);
        }
        
        .legend-count {
            font-size: 10px;
            color: rgba(255,255,255,0.5);
            margin-left: auto;
            padding-left: 8px;
        }
        
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
    
    <!-- Details Panel - Left side -->
    <div class="details-panel" id="details-panel">
        <div class="details-panel-header">
            <div class="details-panel-title" id="details-title">Select an element</div>
            <button class="details-panel-close" id="details-close" title="Close">&times;</button>
        </div>
        <div class="details-content" id="details-content"></div>
    </div>
    
    <!-- Legend Panel - Right side -->
    <div class="legend-panel" id="legend-panel">
        <div class="legend-title">Node Types</div>
        <div id="legend-items"></div>
    </div>
    
    <!-- Zoom Controls -->
    <div class="zoom-controls">
        <button class="zoom-btn" id="zoom-in" title="Zoom In">+</button>
        <button class="zoom-btn" id="zoom-out" title="Zoom Out">-</button>
        <button class="zoom-btn" id="zoom-reset" title="Fit All" style="font-size: 14px;">&#x229E;</button>
    </div>
    
    <script>
        // Detect if running in iframe (Streamlit) and adjust styles
        var inIframe = window.self !== window.top;
        if (inIframe) {
            document.body.classList.add('in-iframe');
        }
        
        var nodes = {nodes};
        var links = {links};

        // State variables
        var selectedElement = null;
        var highlightedType = null;

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

        // ========================================
        // Helper function: Create tree-like HTML representation
        // ========================================
        function createTreeHTML(obj, indent) {
            indent = indent || 0;
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

        // ========================================
        // Generate edge content (for tooltip and details panel)
        // ========================================
        function generateEdgeContent(d) {
            var content = "<strong>Relationship:</strong> " + d.relation + "<br/>";
            content += "<strong>From:</strong> " + (d.source.name || d.source) + "<br/>";
            content += "<strong>To:</strong> " + (d.target.name || d.target) + "<br/><br/>";

            // Show all weights
            if (d.all_weights && Object.keys(d.all_weights).length > 0) {
                content += "<strong>Weights:</strong><br/>";
                Object.keys(d.all_weights).forEach(function(weightName) {
                    content += "&nbsp;&nbsp;" + weightName + ": " + d.all_weights[weightName] + "<br/>";
                });
            } else if (d.weight !== null && d.weight !== undefined) {
                content += "<strong>Weight:</strong> " + d.weight + "<br/>";
            }

            if (d.relationship_type) {
                content += "<strong>Type:</strong> " + d.relationship_type + "<br/>";
            }
            
            // Add other edge properties
            if (d.edge_info) {
                var hasExtra = false;
                Object.keys(d.edge_info).forEach(function(key) {
                    if (key !== 'weight' && key !== 'weights' && key !== 'relationship_type' && 
                        key !== 'source_node_id' && key !== 'target_node_id' && 
                        key !== 'relationship_name' && key !== 'updated_at' && 
                        !key.startsWith('weight_')) {
                        if (!hasExtra) {
                            content += "<br/><strong>Properties:</strong><br/>";
                            hasExtra = true;
                        }
                        content += "&nbsp;&nbsp;" + key + ": " + d.edge_info[key] + "<br/>";
                    }
                });
            }
            
            return content;
        }

        // ========================================
        // Generate node content (for tooltip and details panel)
        // ========================================
        function generateNodeContent(d) {
            var titleFontSize = inIframe ? '16px' : '12px';
            var content = "<strong style='font-size: " + titleFontSize + ";'>" + d.type + "</strong><br/><br/>";
            content += createTreeHTML(d);
            return content;
        }

        // ========================================
        // Links/Edges
        // ========================================
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
                tooltip.html(generateEdgeContent(d))
                    .style("left", (d3.event.pageX + 10) + "px")
                    .style("top", (d3.event.pageY - 10) + "px")
                    .style("opacity", 1);
            })
            .on("mouseout", function(d) {
                tooltip.style("opacity", 0);
            })
            .on("click", function(d) {
                d3.event.stopPropagation();
                selectEdge(d, this);
            });

        // ========================================
        // Edge Labels
        // ========================================
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

        // ========================================
        // Nodes
        // ========================================
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

        var nodeLabels = nodeGroup.append("text")
            .attr("class", "node-label")
            .attr("dy", 4)
            .attr("text-anchor", "middle")
            .text(d => d.name);

        nodeGroup.on("mouseover", function(d) {
            tooltip.html(generateNodeContent(d))
                .style("left", (d3.event.pageX + 10) + "px")
                .style("top", (d3.event.pageY - 10) + "px")
                .style("opacity", 1);
        })
        .on("mouseout", function(d) {
            tooltip.style("opacity", 0);
        })
        .on("click", function(d) {
            d3.event.stopPropagation();
            selectNode(d, this);
        });

        // ========================================
        // Selection Functions
        // ========================================
        function clearSelection() {
            selectedElement = null;
            d3.selectAll('.nodes circle').classed('selected', false);
            d3.selectAll('.links line').classed('selected', false);
            document.getElementById('details-panel').classList.remove('visible');
        }

        function selectNode(d, element) {
            clearSelection();
            selectedElement = { type: 'node', data: d };
            d3.select(element).select('circle').classed('selected', true);
            
            document.getElementById('details-title').innerHTML = d.type + ': ' + d.name;
            document.getElementById('details-content').innerHTML = createTreeHTML(d);
            document.getElementById('details-panel').classList.add('visible');
        }

        function selectEdge(d, element) {
            clearSelection();
            selectedElement = { type: 'edge', data: d };
            d3.select(element).classed('selected', true);
            
            document.getElementById('details-title').innerHTML = 'Edge: ' + d.relation;
            document.getElementById('details-content').innerHTML = generateEdgeContent(d);
            document.getElementById('details-panel').classList.add('visible');
        }

        // Close button for details panel
        document.getElementById('details-close').addEventListener('click', function() {
            clearSelection();
        });

        // Click on SVG background to clear selection
        svg.on("click", function() {
            if (d3.event.target.tagName === 'svg') {
                clearSelection();
            }
        });

        // ========================================
        // Legend Functions
        // ========================================
        function generateLegend() {
            var typeColorMap = {};
            var typeCounts = {};
            
            nodes.forEach(function(n) {
                if (!typeColorMap[n.type]) {
                    typeColorMap[n.type] = n.color;
                    typeCounts[n.type] = 0;
                }
                typeCounts[n.type]++;
            });
            
            var types = Object.keys(typeColorMap).sort();
            var legendHtml = '';
            
            types.forEach(function(type) {
                legendHtml += '<div class="legend-item" data-type="' + type + '">' +
                    '<div class="legend-color" style="background.' + typeColorMap[type] + '"></div>' +
                    '<span class="legend-label">' + type + '</span>' +
                    '<span class="legend-count">' + typeCounts[type] + '</span>' +
                    '</div>';
            });
            
            document.getElementById('legend-items').innerHTML = legendHtml;
            
            // Add click handlers for highlight
            document.querySelectorAll('.legend-item').forEach(function(item) {
                item.addEventListener('click', function() {
                    var type = this.getAttribute('data-type');
                    toggleHighlightType(type);
                });
            });
        }

        function toggleHighlightType(type) {
            if (highlightedType === type) {
                // Clear highlight
                highlightedType = null;
                d3.selectAll('.legend-item').classed('active', false);
                node.classed('dimmed', false);
                nodeLabels.classed('dimmed', false);
                link.classed('dimmed', false);
                edgeLabels.classed('dimmed', false);
            } else {
                // Apply highlight
                highlightedType = type;
                d3.selectAll('.legend-item').classed('active', function() {
                    return this.getAttribute('data-type') === type;
                });
                node.classed('dimmed', function(d) { return d.type !== type; });
                nodeLabels.classed('dimmed', function(d) { return d.type !== type; });
                link.classed('dimmed', function(d) {
                    return d.source.type !== type && d.target.type !== type;
                });
                edgeLabels.classed('dimmed', function(d) {
                    return d.source.type !== type && d.target.type !== type;
                });
            }
        }

        // Generate legend on load
        generateLegend();

        // ========================================
        // Simulation Tick
        // ========================================
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

        // ========================================
        // Zoom Controls
        // ========================================
        var zoom = d3.zoom().on("zoom", function() {
            container.attr("transform", d3.event.transform);
        });
        
        svg.call(zoom);
        
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

        // ========================================
        // Drag Functions
        // ========================================
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

        // ========================================
        // Window Resize
        // ========================================
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
