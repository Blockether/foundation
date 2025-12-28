"""Graph visualization presentation layer using D3.js and TailwindCSS."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from blockether_foundation.graph.common import (
    ENTITY_TYPE_DEFINITIONS,
    RELATIONSHIP_TYPE_DEFINITIONS,
)
from blockether_foundation.palette import (
    ENTITY_TYPE_COLORS,
    GRAY_50,
    GRAY_100,
    GRAY_200,
    GRAY_700,
    PRIMARY_YELLOW,
    PRIMARY_YELLOW_DARK,
    PRIMARY_YELLOW_LIGHT,
    WHITE,
)

if TYPE_CHECKING:
    from blockether_foundation.graph.database import GraphDatabase

DEFAULT_NODE_COLOR = PRIMARY_YELLOW


def _escape_js_string(s: str) -> str:
    """Escape a string for safe inclusion in JavaScript."""
    return html.escape(s).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def generate_graph_html(
    db: "GraphDatabase",  # noqa: UP037
    title: str = "Knowledge Graph",
    width: str = "100%",
    height: str = "100vh",
) -> str:
    """Generate an interactive HTML visualization of the graph database."""
    nodes_data: list[dict[str, str | int]] = []
    links_data: list[dict[str, str]] = []

    entity_type_counts: Counter[str] = Counter()
    relationship_type_counts: Counter[str] = Counter()

    for entity_id, entity in db.index.entity_by_id.items():
        degree = db.get_entity_degree(entity_id)
        color = ENTITY_TYPE_COLORS.get(entity.type, DEFAULT_NODE_COLOR)
        entity_type_counts[entity.type] += 1
        nodes_data.append(
            {
                "id": entity_id,
                "name": entity.name,
                "type": entity.type,
                "content": entity.content or "",
                "color": color,
                "degree": degree,
            }
        )

    for rel_id, rel in db.index.relationship_by_id.items():
        relationship_type_counts[rel.type] += 1
        links_data.append(
            {
                "id": rel_id,
                "source": rel.source,
                "target": rel.target,
                "type": rel.type,
            }
        )

    metrics = db.get_graph_density_metrics()
    nodes_json = json.dumps(nodes_data)
    links_json = json.dumps(links_data)
    entity_definitions_json = json.dumps(ENTITY_TYPE_DEFINITIONS)
    relationship_definitions_json = json.dumps(RELATIONSHIP_TYPE_DEFINITIONS)

    entity_type_items = "\n".join(
        f"""<div class="entity-type-item flex items-center justify-between py-1 px-2 rounded cursor-pointer hover:bg-yellow-50 transition-colors"
                  data-type="entity" data-value="{etype}" title="{html.escape(ENTITY_TYPE_DEFINITIONS.get(etype, "No definition available"))}">
            <div class="flex items-center gap-2">
                <div class="w-3 h-3 rounded-full" style="background-color: {ENTITY_TYPE_COLORS.get(etype, DEFAULT_NODE_COLOR)}"></div>
                <span class="text-sm text-gray-700 capitalize">{etype}</span>
            </div>
            <span class="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">{count}</span>
        </div>"""
        for etype, count in sorted(entity_type_counts.items(), key=lambda x: -x[1])
    )

    relationship_type_items = "\n".join(
        f"""<div class="relationship-type-item flex items-center justify-between py-1 px-2 rounded cursor-pointer hover:bg-yellow-50 transition-colors"
                  data-type="relationship" data-value="{rtype}" title="{html.escape(RELATIONSHIP_TYPE_DEFINITIONS.get(rtype, "No definition available"))}">
            <span class="text-sm text-gray-700">{rtype}</span>
            <span class="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">{count}</span>
        </div>"""
        for rtype, count in sorted(relationship_type_counts.items(), key=lambda x: -x[1])
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_js_string(title)}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {{
            --primary-yellow: {PRIMARY_YELLOW};
            --primary-yellow-dark: {PRIMARY_YELLOW_DARK};
            --primary-yellow-light: {PRIMARY_YELLOW_LIGHT};
        }}

        body {{ font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
        .node {{ cursor: pointer; }}
        .node circle {{ transition: all 0.15s ease; }}
        .node:hover circle {{ filter: brightness(1.1); }}
        .link {{ stroke: {GRAY_200}; stroke-opacity: 0.6; fill: none; }}
        .node-label {{ font-size: 11px; font-weight: 500; pointer-events: none; text-shadow: 0 1px 2px rgba(255,255,255,0.9); }}
        .graph-container {{ background: linear-gradient(135deg, {WHITE} 0%, {GRAY_50} 100%); }}
        .stats-card {{ background: {WHITE}; border: 1px solid {GRAY_100}; border-left: 4px solid {PRIMARY_YELLOW}; }}
        .header-accent {{ background: linear-gradient(90deg, {PRIMARY_YELLOW} 0%, {PRIMARY_YELLOW_LIGHT} 100%); }}
        #graph-svg {{ width: {width}; height: {height}; }}
        .sidebar {{ transition: transform 0.3s ease, width 0.3s ease; }}
        .sidebar.collapsed {{ transform: translateX(-100%); width: 0; padding: 0; overflow: hidden; }}
        .sidebar.collapsed + .graph-wrapper {{ margin-left: 0; }}
        .right-sidebar {{ transition: transform 0.3s ease; }}
        .right-sidebar.hidden {{ transform: translateX(100%); }}
        .type-item-active {{ background-color: {PRIMARY_YELLOW_LIGHT} !important; }}
        .type-item-active span {{ font-weight: 600; }}
    </style>
</head>
<body class="bg-white min-h-screen overflow-hidden">
    <header class="bg-white border-b border-gray-100 sticky top-0 z-50">
        <div class="header-accent h-1"></div>
        <div class="px-4 py-3">
            <div class="flex items-center justify-between gap-4">
                <div class="flex items-center gap-3 flex-shrink-0">
                    <button id="toggle-sidebar" class="p-2 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors" title="Toggle Sidebar">
                        <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/>
                        </svg>
                    </button>
                    <div class="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0" style="background-color: {PRIMARY_YELLOW}">
                        <svg viewBox="113.12 113.12 1003.25 1154.58" aria-hidden="true" preserveAspectRatio="xMinYMid meet"><path d="M614.79,125.623l-489.167,282.375l0,564.792l489.125,282.417l489.125,-282.417l0,-564.792l-489.083,-282.375Zm0,912.834l-301.417,-174.042l0,-122.958l409.209,-0l193.666,122.958l-301.458,174.042Zm107.75,-399.125l-409.167,-0l0,-122.959l301.417,-174.041l301.417,174.041l-193.667,122.959Z" class="stroke-25 transition-all duration-300 stroke-neutral-950" fill="#fc0"></path></svg>
                    </div>
                    <div class="hidden sm:block">
                        <h1 class="text-lg font-bold text-gray-900 leading-tight">{_escape_js_string(title)}</h1>
                    </div>
                </div>

                <div class="flex-1 max-w-xl mx-4">
                    <div class="relative">
                        <svg class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                        </svg>
                        <input type="text" id="search-input" placeholder="Search entities..."
                            class="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:border-transparent bg-gray-50 focus:bg-white transition-colors">
                    </div>
                </div>

                <div class="flex items-center gap-2 flex-shrink-0">
                    <button id="zoom-in" class="p-2 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors" title="Zoom In">
                        <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
                        </svg>
                    </button>
                    <button id="zoom-out" class="p-2 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors" title="Zoom Out">
                        <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/>
                        </svg>
                    </button>
                    <button id="reset-view" class="p-2 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors" title="Reset View">
                        <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                        </svg>
                    </button>
                    <div class="w-px h-6 bg-gray-200 hidden sm:block"></div>
                    <button id="toggle-labels" class="hidden sm:block px-3 py-2 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors text-sm font-medium text-gray-700">
                        Labels
                    </button>
                </div>
            </div>
        </div>
    </header>

    <div class="flex relative" style="height: calc(100vh - 57px);">
        <aside id="left-sidebar" class="sidebar w-64 border-r border-gray-100 bg-white p-4 overflow-y-auto flex-shrink-0 z-40">
            <div class="mb-5">
                <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Overview</h2>
                <div class="grid grid-cols-2 gap-2">
                    <div class="stats-card rounded-lg p-2.5">
                        <div class="text-xl font-bold text-gray-900">{metrics.get("total_entities", 0)}</div>
                        <div class="text-xs text-gray-500">Entities</div>
                    </div>
                    <div class="stats-card rounded-lg p-2.5">
                        <div class="text-xl font-bold text-gray-900">{metrics.get("total_relationships", 0)}</div>
                        <div class="text-xs text-gray-500">Relations</div>
                    </div>
                    <div class="stats-card rounded-lg p-2.5">
                        <div class="text-xl font-bold text-gray-900">{metrics.get("average_degree", 0):.1f}</div>
                        <div class="text-xs text-gray-500">Avg Conns</div>
                    </div>
                    <div class="stats-card rounded-lg p-2.5">
                        <div class="text-xl font-bold text-gray-900">{metrics.get("density", 0):.3f}</div>
                        <div class="text-xs text-gray-500">Density</div>
                    </div>
                </div>
            </div>

            <div class="mb-5">
                <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Entity Types</h2>
                <div class="space-y-0.5">{entity_type_items}</div>
            </div>

            <div>
                <h2 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Relationship Types</h2>
                <div class="space-y-0.5">{relationship_type_items}</div>
            </div>
        </aside>

        <main class="flex-1 graph-container relative overflow-hidden">
            <svg id="graph-svg"></svg>
        </main>

        <aside id="right-sidebar" class="right-sidebar hidden absolute right-0 top-0 w-80 h-full border-l border-gray-100 bg-white shadow-lg z-40 flex flex-col">
            <div class="flex items-center justify-between p-4 border-b border-gray-100">
                <h2 class="text-sm font-semibold text-gray-900 uppercase tracking-wider">Entity Details</h2>
                <button id="close-details" class="p-1 rounded hover:bg-gray-100 transition-colors">
                    <svg class="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                    </svg>
                </button>
            </div>
            <div id="node-info-content" class="flex-1 overflow-y-auto p-4"></div>
        </aside>
    </div>

    <script>
        const nodes = {nodes_json};
        const links = {links_json};
        const entityDefinitions = {entity_definitions_json};
        const relationshipDefinitions = {relationship_definitions_json};

        let showLabels = true;
        let selectedNode = null;
        let leftSidebarCollapsed = false;
        let activeFilter = {{ type: null, value: null }};

        const svg = d3.select("#graph-svg");
        const container = svg.node().parentElement;
        let width = container.clientWidth;
        let height = container.clientHeight;

        svg.attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");

        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {{ g.attr("transform", event.transform); }});

        svg.call(zoom);

        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "-0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .append("path")
            .attr("d", "M 0,-5 L 10,0 L 0,5")
            .attr("fill", "{GRAY_200}");

        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));

        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", 1.5)
            .attr("marker-end", "url(#arrowhead)");

        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", d => Math.max(8, Math.min(20, 8 + d.degree * 2)))
            .attr("fill", d => d.color)
            .attr("stroke", "{WHITE}")
            .attr("stroke-width", 2);

        const labels = node.append("text")
            .attr("class", "node-label")
            .attr("dy", d => Math.max(8, Math.min(20, 8 + d.degree * 2)) + 14)
            .attr("text-anchor", "middle")
            .attr("fill", "{GRAY_700}")
            .text(d => d.name.length > 15 ? d.name.substring(0, 15) + "..." : d.name);

        node.on("click", function(event, d) {{
            event.stopPropagation();
            selectNode(d);
        }});

        svg.on("click", function() {{
            deselectNode();
        }});

        function applyFilter(filterType, filterValue) {{
            if (activeFilter.type === filterType && activeFilter.value === filterValue) {{
                clearFilter();
                return;
            }}

            activeFilter = {{ type: filterType, value: filterValue }};

            if (filterType === 'entity') {{
                node.style("opacity", d => d.type === filterValue ? 1 : 0.1);
                link.style("opacity", d => {{
                    const sourceMatch = d.source.type === filterValue;
                    const targetMatch = d.target.type === filterValue;
                    return (sourceMatch || targetMatch) ? 1 : 0.05;
                }});
            }} else if (filterType === 'relationship') {{
                node.style("opacity", d => {{
                    const connectedLinks = links.filter(l => l.source.id === d.id || l.target.id === d.id);
                    const hasMatchingLink = connectedLinks.some(l => l.type === filterValue);
                    return hasMatchingLink ? 1 : 0.1;
                }});
                link.style("opacity", d => d.type === filterValue ? 1 : 0.05);
            }}

            updateTypeItemStyles();
        }}

        function clearFilter() {{
            activeFilter = {{ type: null, value: null }};
            node.style("opacity", 1);
            link.style("opacity", 1);
            updateTypeItemStyles();
        }}

        function updateTypeItemStyles() {{
            document.querySelectorAll(".entity-type-item, .relationship-type-item").forEach(item => {{
                const itemType = item.dataset.type;
                const itemValue = item.dataset.value;
                if (activeFilter.type === itemType && activeFilter.value === itemValue) {{
                    item.classList.add("type-item-active");
                }} else {{
                    item.classList.remove("type-item-active");
                }}
            }});
        }}

        function selectNode(d) {{
            selectedNode = d;

            node.selectAll("circle")
                .attr("stroke", n => n.id === d.id ? "{PRIMARY_YELLOW}" : "{WHITE}")
                .attr("stroke-width", n => n.id === d.id ? 4 : 2);

            link.attr("stroke", l => (l.source.id === d.id || l.target.id === d.id) ? "{PRIMARY_YELLOW}" : "{GRAY_200}")
                .attr("stroke-width", l => (l.source.id === d.id || l.target.id === d.id) ? 2.5 : 1.5);

            updateNodeInfo(d);
            document.getElementById("right-sidebar").classList.remove("hidden");
        }}

        function deselectNode() {{
            selectedNode = null;

            node.selectAll("circle")
                .attr("stroke", "{WHITE}")
                .attr("stroke-width", 2);

            link.attr("stroke", "{GRAY_200}")
                .attr("stroke-width", 1.5);

            document.getElementById("right-sidebar").classList.add("hidden");
        }}

        function updateNodeInfo(d) {{
            const content = document.getElementById("node-info-content");

            const connectedLinks = links.filter(l => l.source.id === d.id || l.target.id === d.id);

            const outgoing = connectedLinks.filter(l => l.source.id === d.id).map(l => {{
                return `<div class="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-gray-50 cursor-pointer" onclick="focusNode('${{l.target.id}}')">
                    <div class="w-2 h-2 rounded-full" style="background-color: ${{l.target.color}}"></div>
                    <span class="text-sm text-gray-700 flex-1">${{l.target.name}}</span>
                    <span class="text-xs text-gray-400">${{l.type}}</span>
                </div>`;
            }}).join("");

            const incoming = connectedLinks.filter(l => l.target.id === d.id).map(l => {{
                return `<div class="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-gray-50 cursor-pointer" onclick="focusNode('${{l.source.id}}')">
                    <div class="w-2 h-2 rounded-full" style="background-color: ${{l.source.color}}"></div>
                    <span class="text-sm text-gray-700 flex-1">${{l.source.name}}</span>
                    <span class="text-xs text-gray-400">${{l.type}}</span>
                </div>`;
            }}).join("");

            content.innerHTML = `
                <div class="mb-4">
                    <div class="flex items-center gap-3 mb-3">
                        <div class="w-10 h-10 rounded-full flex items-center justify-center" style="background-color: ${{d.color}}">
                            <span class="text-white font-bold text-sm">${{d.name.charAt(0).toUpperCase()}}</span>
                        </div>
                        <div class="flex-1 min-w-0">
                            <div class="font-semibold text-gray-900 truncate">${{d.name}}</div>
                            <div class="text-xs text-gray-500 uppercase">${{d.type}}</div>
                        </div>
                    </div>
                </div>

                ${{d.content ? `
                <div class="mb-4">
                    <h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Content</h3>
                    <p class="text-sm text-gray-600 leading-relaxed">${{d.content}}</p>
                </div>
                ` : ''}}

                <div class="mb-4">
                    <h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Outgoing (${{connectedLinks.filter(l => l.source.id === d.id).length}})</h3>
                    <div class="space-y-0.5">
                        ${{outgoing || '<div class="text-sm text-gray-400 py-1">None</div>'}}
                    </div>
                </div>

                <div>
                    <h3 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Incoming (${{connectedLinks.filter(l => l.target.id === d.id).length}})</h3>
                    <div class="space-y-0.5">
                        ${{incoming || '<div class="text-sm text-gray-400 py-1">None</div>'}}
                    </div>
                </div>
            `;
        }}

        window.focusNode = function(nodeId) {{
            const targetNode = nodes.find(n => n.id === nodeId);
            if (targetNode) {{
                selectNode(targetNode);
                const transform = d3.zoomIdentity.translate(width/2 - targetNode.x, height/2 - targetNode.y);
                svg.transition().duration(500).call(zoom.transform, transform);
            }}
        }};

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});

        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        document.getElementById("toggle-sidebar").addEventListener("click", () => {{
            const sidebar = document.getElementById("left-sidebar");
            leftSidebarCollapsed = !leftSidebarCollapsed;
            sidebar.classList.toggle("collapsed", leftSidebarCollapsed);
            setTimeout(() => {{
                width = container.clientWidth;
                svg.attr("viewBox", [0, 0, width, height]);
                simulation.force("center", d3.forceCenter(width / 2, height / 2));
                simulation.alpha(0.1).restart();
            }}, 300);
        }});

        document.getElementById("close-details").addEventListener("click", () => {{
            deselectNode();
        }});

        document.getElementById("zoom-in").addEventListener("click", () => {{
            svg.transition().duration(300).call(zoom.scaleBy, 1.3);
        }});

        document.getElementById("zoom-out").addEventListener("click", () => {{
            svg.transition().duration(300).call(zoom.scaleBy, 0.7);
        }});

        document.getElementById("reset-view").addEventListener("click", () => {{
            svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
            simulation.alpha(0.3).restart();
        }});

        document.getElementById("toggle-labels").addEventListener("click", function() {{
            showLabels = !showLabels;
            labels.style("display", showLabels ? "block" : "none");
            this.style.backgroundColor = showLabels ? "" : "{GRAY_100}";
        }});

        document.getElementById("search-input").addEventListener("input", function(e) {{
            const query = e.target.value.toLowerCase();
            clearFilter();

            node.style("opacity", d => {{
                if (!query) return 1;
                return d.name.toLowerCase().includes(query) ||
                       d.type.toLowerCase().includes(query) ||
                       (d.content && d.content.toLowerCase().includes(query)) ? 1 : 0.15;
            }});

            link.style("opacity", d => {{
                if (!query) return 1;
                const sourceMatch = d.source.name.toLowerCase().includes(query);
                const targetMatch = d.target.name.toLowerCase().includes(query);
                return (sourceMatch || targetMatch) ? 1 : 0.05;
            }});
        }});

        window.addEventListener("resize", () => {{
            width = container.clientWidth;
            height = container.clientHeight;
            svg.attr("viewBox", [0, 0, width, height]);
            simulation.force("center", d3.forceCenter(width / 2, height / 2));
            simulation.alpha(0.1).restart();
        }});

        document.querySelectorAll(".entity-type-item, .relationship-type-item").forEach(item => {{
            item.addEventListener("click", function() {{
                const filterType = this.dataset.type;
                const filterValue = this.dataset.value;
                applyFilter(filterType, filterValue);
            }});
        }});
    </script>
</body>
</html>"""

    return html_content


def export_graph_to_html(
    db: "GraphDatabase",  # noqa: UP037
    output_path: str | Path,
    title: str = "Knowledge Graph",
) -> Path:
    """Export graph database to an interactive HTML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    html_content = generate_graph_html(db, title=title)
    path.write_text(html_content, encoding="utf-8")

    return path
