"""Consensus process visualization using D3.js and TailwindCSS."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from blockether_foundation.palette import (
    CHART_COLORS,
    ERROR_RED,
    GRAY_100,
    GRAY_200,
    GRAY_50,
    GRAY_500,
    GRAY_700,
    GRAY_800,
    GRAY_900,
    INFO_BLUE,
    PRIMARY_YELLOW,
    PRIMARY_YELLOW_DARK,
    PRIMARY_YELLOW_LIGHT,
    SUCCESS_GREEN,
    WARNING_AMBER,
    WHITE,
)

if TYPE_CHECKING:
    from blockether_foundation.agents.hooks.consensus.core import ConsensusResult


def _escape_js_string(s: str) -> str:
    return html.escape(s).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def _get_score_color(score: float) -> str:
    if score >= 0.8:
        return SUCCESS_GREEN
    elif score >= 0.6:
        return WARNING_AMBER
    else:
        return ERROR_RED


def generate_consensus_report_html(
    result: "ConsensusResult",
    title: str = "Consensus Decision Report",
) -> str:
    model_data: list[dict[str, Any]] = [
        {
            "name": mc.model_name,
            "importance": mc.importance,
            "perspective": mc.perspective or "General",
            "contributions": mc.key_contributions,
            "incorporated": mc.insights_incorporated,
            "rejected": mc.insights_rejected,
            "color": CHART_COLORS[i % len(CHART_COLORS)],
        }
        for i, mc in enumerate(result.model_contributions)
    ]
    models_json = json.dumps(model_data)
    chart_colors_json = json.dumps(CHART_COLORS)

    confidence_color = _get_score_color(result.consensus_confidence)
    judge_color = _get_score_color(result.judge_score)

    escaped_output = html.escape(result.final_output)
    escaped_gen_summary = html.escape(result.generation_summary)
    escaped_critique_summary = html.escape(result.critique_summary)
    escaped_synthesis_summary = html.escape(result.synthesis_summary)
    escaped_judge_summary = html.escape(result.judge_summary)

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
        .header-accent {{ background: linear-gradient(90deg, {PRIMARY_YELLOW} 0%, {PRIMARY_YELLOW_LIGHT} 100%); }}
        .card {{ background: {WHITE}; border: 1px solid {GRAY_100}; }}
        .card-accent {{ border-left: 4px solid {PRIMARY_YELLOW}; }}
        .phase-dot {{ width: 40px; height: 40px; }}
        .phase-line {{ height: 4px; background: {GRAY_200}; }}
        .phase-line.completed {{ background: {PRIMARY_YELLOW}; }}
        .score-ring {{ stroke-linecap: round; }}
        .accordion-content {{ max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; }}
        .accordion-content.open {{ max-height: 2000px; }}
        .tooltip {{
            position: absolute;
            background: {GRAY_800};
            color: {WHITE};
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 100;
            max-width: 300px;
        }}
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div id="tooltip" class="tooltip"></div>

    <header class="bg-white border-b border-gray-100 sticky top-0 z-50">
        <div class="header-accent h-1"></div>
        <div class="max-w-6xl mx-auto px-4 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 rounded-lg flex items-center justify-center" style="background-color: {PRIMARY_YELLOW}">
                        <svg class="w-6 h-6 text-gray-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold text-gray-900">{_escape_js_string(title)}</h1>
                        <p class="text-sm text-gray-500">{len(result.model_contributions)} models, {result.refinement_iterations} refinement iteration(s)</p>
                    </div>
                </div>
                <div class="flex items-center gap-6">
                    <div class="text-center">
                        <div class="text-2xl font-bold" style="color: {confidence_color}">{result.consensus_confidence:.0%}</div>
                        <div class="text-xs text-gray-500 uppercase tracking-wider">Confidence</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold" style="color: {judge_color}">{result.judge_score:.0%}</div>
                        <div class="text-xs text-gray-500 uppercase tracking-wider">Judge Score</div>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-4 py-8">
        <!-- Process Timeline -->
        <section class="mb-8">
            <h2 class="text-lg font-semibold text-gray-900 mb-4">Decision Process</h2>
            <div class="card rounded-xl p-6">
                <div class="flex items-center justify-between">
                    <div class="flex flex-col items-center flex-1">
                        <div class="phase-dot rounded-full flex items-center justify-center text-white font-bold" style="background-color: {PRIMARY_YELLOW}">1</div>
                        <div class="mt-2 text-sm font-medium text-gray-900">Generation</div>
                        <div class="text-xs text-gray-500 text-center mt-1 max-w-32">{len(result.model_contributions)} models generated outputs</div>
                    </div>
                    <div class="phase-line completed flex-1 -mt-8"></div>
                    <div class="flex flex-col items-center flex-1">
                        <div class="phase-dot rounded-full flex items-center justify-center text-white font-bold" style="background-color: {PRIMARY_YELLOW}">2</div>
                        <div class="mt-2 text-sm font-medium text-gray-900">Critique</div>
                        <div class="text-xs text-gray-500 text-center mt-1 max-w-32">Self + peer review</div>
                    </div>
                    <div class="phase-line completed flex-1 -mt-8"></div>
                    <div class="flex flex-col items-center flex-1">
                        <div class="phase-dot rounded-full flex items-center justify-center text-white font-bold" style="background-color: {PRIMARY_YELLOW}">3</div>
                        <div class="mt-2 text-sm font-medium text-gray-900">Synthesis</div>
                        <div class="text-xs text-gray-500 text-center mt-1 max-w-32">Weighted combination</div>
                    </div>
                    <div class="phase-line completed flex-1 -mt-8"></div>
                    <div class="flex flex-col items-center flex-1">
                        <div class="phase-dot rounded-full flex items-center justify-center text-white font-bold" style="background-color: {PRIMARY_YELLOW}">4</div>
                        <div class="mt-2 text-sm font-medium text-gray-900">Judge</div>
                        <div class="text-xs text-gray-500 text-center mt-1 max-w-32">{result.refinement_iterations} iteration(s)</div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Metrics and Model Contributions -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <!-- Confidence Gauges -->
            <section>
                <h2 class="text-lg font-semibold text-gray-900 mb-4">Quality Metrics</h2>
                <div class="card rounded-xl p-6">
                    <div class="flex justify-around">
                        <div class="text-center">
                            <svg width="120" height="120" viewBox="0 0 120 120" class="mx-auto">
                                <circle cx="60" cy="60" r="50" fill="none" stroke="{GRAY_200}" stroke-width="10"/>
                                <circle cx="60" cy="60" r="50" fill="none" stroke="{confidence_color}" stroke-width="10"
                                    class="score-ring" stroke-dasharray="{314 * result.consensus_confidence} 314"
                                    transform="rotate(-90 60 60)"/>
                                <text x="60" y="60" text-anchor="middle" dy="0.35em" class="text-2xl font-bold" fill="{GRAY_900}">{result.consensus_confidence:.0%}</text>
                            </svg>
                            <div class="mt-2 text-sm font-medium text-gray-700">Consensus Confidence</div>
                        </div>
                        <div class="text-center">
                            <svg width="120" height="120" viewBox="0 0 120 120" class="mx-auto">
                                <circle cx="60" cy="60" r="50" fill="none" stroke="{GRAY_200}" stroke-width="10"/>
                                <circle cx="60" cy="60" r="50" fill="none" stroke="{judge_color}" stroke-width="10"
                                    class="score-ring" stroke-dasharray="{314 * result.judge_score} 314"
                                    transform="rotate(-90 60 60)"/>
                                <text x="60" y="60" text-anchor="middle" dy="0.35em" class="text-2xl font-bold" fill="{GRAY_900}">{result.judge_score:.0%}</text>
                            </svg>
                            <div class="mt-2 text-sm font-medium text-gray-700">Judge Score</div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Model Contributions Pie Chart -->
            <section>
                <h2 class="text-lg font-semibold text-gray-900 mb-4">Model Contributions</h2>
                <div class="card rounded-xl p-6">
                    <div class="flex items-center gap-6">
                        <div id="model-chart" class="flex-shrink-0"></div>
                        <div id="model-legend" class="flex-1 space-y-2"></div>
                    </div>
                </div>
            </section>
        </div>

        <!-- Agreements, Conflicts, Uncertainties -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <section>
                <h2 class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <span class="w-3 h-3 rounded-full" style="background-color: {SUCCESS_GREEN}"></span>
                    Key Agreements ({len(result.key_agreements)})
                </h2>
                <div class="card card-accent rounded-xl p-4" style="border-left-color: {SUCCESS_GREEN}">
                    <ul id="agreements-list" class="space-y-2 text-sm text-gray-700">
                        {_generate_list_items(result.key_agreements, "No key agreements recorded")}
                    </ul>
                </div>
            </section>

            <section>
                <h2 class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <span class="w-3 h-3 rounded-full" style="background-color: {WARNING_AMBER}"></span>
                    Resolved Conflicts ({len(result.resolved_conflicts)})
                </h2>
                <div class="card card-accent rounded-xl p-4" style="border-left-color: {WARNING_AMBER}">
                    <ul id="conflicts-list" class="space-y-2 text-sm text-gray-700">
                        {_generate_list_items(result.resolved_conflicts, "No conflicts to resolve")}
                    </ul>
                </div>
            </section>

            <section>
                <h2 class="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <span class="w-3 h-3 rounded-full" style="background-color: {GRAY_500}"></span>
                    Uncertainties ({len(result.remaining_uncertainties)})
                </h2>
                <div class="card card-accent rounded-xl p-4" style="border-left-color: {GRAY_500}">
                    <ul id="uncertainties-list" class="space-y-2 text-sm text-gray-700">
                        {_generate_list_items(result.remaining_uncertainties, "No remaining uncertainties")}
                    </ul>
                </div>
            </section>
        </div>

        <!-- Phase Summaries (Accordions) -->
        <section class="mb-8">
            <h2 class="text-lg font-semibold text-gray-900 mb-4">Phase Summaries</h2>
            <div class="space-y-3">
                {_generate_accordion("generation", "1. Generation Phase", escaped_gen_summary, INFO_BLUE)}
                {_generate_accordion("critique", "2. Critique Phase", escaped_critique_summary, PRIMARY_YELLOW)}
                {_generate_accordion("synthesis", "3. Synthesis Phase", escaped_synthesis_summary, SUCCESS_GREEN)}
                {_generate_accordion("judge", "4. Judge & Refine Phase", escaped_judge_summary, WARNING_AMBER)}
            </div>
        </section>

        <!-- Final Output -->
        <section class="mb-8">
            <h2 class="text-lg font-semibold text-gray-900 mb-4">Final Consensus Output</h2>
            <div class="card card-accent rounded-xl p-6">
                <div class="prose prose-gray max-w-none">
                    <pre class="whitespace-pre-wrap text-sm text-gray-800 font-sans leading-relaxed">{escaped_output}</pre>
                </div>
            </div>
        </section>
    </main>

    <footer class="border-t border-gray-100 bg-white py-4 mt-8">
        <div class="max-w-6xl mx-auto px-4 text-center text-sm text-gray-500">
            Generated by Blockether Foundation Consensus Engine
        </div>
    </footer>

    <script>
        const models = {models_json};
        const chartColors = {chart_colors_json};

        const width = 150, height = 150, radius = Math.min(width, height) / 2;

        const svg = d3.select("#model-chart")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", `translate(${{width/2}}, ${{height/2}})`);

        const pie = d3.pie().value(d => d.importance).sort(null);
        const arc = d3.arc().innerRadius(radius * 0.5).outerRadius(radius * 0.9);

        const tooltip = d3.select("#tooltip");

        const arcs = svg.selectAll("path")
            .data(pie(models))
            .join("path")
            .attr("d", arc)
            .attr("fill", d => d.data.color)
            .attr("stroke", "{WHITE}")
            .attr("stroke-width", 2)
            .style("cursor", "pointer")
            .on("mouseover", function(event, d) {{
                d3.select(this).attr("opacity", 0.8);
                tooltip.style("opacity", 1)
                    .html(`<strong>${{d.data.name}}</strong><br/>Importance: ${{(d.data.importance * 100).toFixed(0)}}%<br/>Incorporated: ${{d.data.incorporated}}<br/>Rejected: ${{d.data.rejected}}`);
            }})
            .on("mousemove", function(event) {{
                tooltip.style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            }})
            .on("mouseout", function() {{
                d3.select(this).attr("opacity", 1);
                tooltip.style("opacity", 0);
            }});

        const legend = d3.select("#model-legend");
        models.forEach(m => {{
            legend.append("div")
                .attr("class", "flex items-center gap-2")
                .html(`
                    <div class="w-3 h-3 rounded-full flex-shrink-0" style="background-color: ${{m.color}}"></div>
                    <div class="flex-1 min-w-0">
                        <div class="text-sm font-medium text-gray-900 truncate">${{m.name}}</div>
                        <div class="text-xs text-gray-500">${{m.perspective}}</div>
                    </div>
                    <div class="text-sm font-bold text-gray-700">${{(m.importance * 100).toFixed(0)}}%</div>
                `);
        }});

        document.querySelectorAll('.accordion-header').forEach(header => {{
            header.addEventListener('click', function() {{
                const content = this.nextElementSibling;
                const icon = this.querySelector('.accordion-icon');
                content.classList.toggle('open');
                icon.classList.toggle('rotate-180');
            }});
        }});
    </script>
</body>
</html>"""

    return html_content


def _generate_list_items(items: list[str], empty_message: str) -> str:
    if not items:
        return f'<li class="text-gray-400 italic">{empty_message}</li>'
    return "\n".join(
        f'<li class="flex items-start gap-2"><span class="text-gray-400">•</span><span>{html.escape(item)}</span></li>'
        for item in items
    )


def _generate_accordion(id_prefix: str, title: str, content: str, color: str) -> str:
    return f"""
    <div class="card rounded-xl overflow-hidden">
        <button class="accordion-header w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors">
            <div class="flex items-center gap-3">
                <div class="w-2 h-2 rounded-full" style="background-color: {color}"></div>
                <span class="font-medium text-gray-900">{title}</span>
            </div>
            <svg class="accordion-icon w-5 h-5 text-gray-500 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
            </svg>
        </button>
        <div class="accordion-content">
            <div class="px-4 pb-4 pt-0 text-sm text-gray-700 border-t border-gray-100">
                <p class="pt-3">{content}</p>
            </div>
        </div>
    </div>"""


def export_consensus_report_to_html(
    result: "ConsensusResult",
    output_path: str | Path,
    title: str = "Consensus Decision Report",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    html_content = generate_consensus_report_html(result, title=title)
    path.write_text(html_content, encoding="utf-8")

    return path
