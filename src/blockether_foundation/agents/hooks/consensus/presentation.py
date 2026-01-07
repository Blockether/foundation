"""Consensus process visualization using D3.js and TailwindCSS."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from blockether_foundation.palette import (
    CHART_COLORS,
    D3_CHART_INIT_SCRIPT,
    GRAY_100,
    GRAY_200,
    GRAY_300,
    GRAY_400,
    GRAY_500,
    GRAY_700,
    GRAY_800,
    GRAY_900,
    HTML_REPORT_FOOTER,
    HTML_REPORT_HEAD,
    HTML_REPORT_HEADER_TEMPLATE,
    PRIMARY_YELLOW,
    SUCCESS_GREEN,
    SUCCESS_GREEN_LIGHT,
    WARNING_AMBER,
    WARNING_AMBER_LIGHT,
    WHITE,
)

# Define colors not imported from palette
GRAY_50 = "#F9FAFB"
GRAY_600 = "#4B5563"

if TYPE_CHECKING:
    from blockether_foundation.agents.hooks.consensus.core import (
        ConsensusResult,
        CritiqueSummary,
    )


def _escape_js_string(s: str) -> str:
    return html.escape(s).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def _get_score_color(score: float) -> str:
    if score >= 0.8:
        return GRAY_900
    elif score >= 0.6:
        return GRAY_700
    else:
        return GRAY_500


def _generate_step_navigation() -> str:
    """Generate sticky step navigation."""
    return """    <nav class="step-nav">
        <div class="step-nav-inner">
            <div class="step-item active" data-step-target="goal">
                <span class="step-number">1</span>
                <span>Goal</span>
            </div>
            <div class="step-item" data-step-target="acceptance">
                <span class="step-number">2</span>
                <span>Acceptance Criteria</span>
            </div>
            <div class="step-item" data-step-target="results">
                <span class="step-number">3</span>
                <span>Results</span>
            </div>
            <div class="step-item" data-step-target="gossip">
                <span class="step-number">4</span>
                <span>Gossip</span>
            </div>
            <div class="step-item" data-step-target="evaluation">
                <span class="step-number">5</span>
                <span>Evaluation</span>
            </div>
            <div class="step-item" data-step-target="conflicts">
                <span class="step-number">6</span>
                <span>Conflicts Resolution</span>
            </div>
            <div class="step-item" data-step-target="decision">
                <span class="step-number">7</span>
                <span>Decision</span>
            </div>
        </div>
    </nav>"""


def _generate_task_section(result: ConsensusResult) -> str:
    """Generate task section - shows the original user request."""
    escaped_task = html.escape(result.user_input)

    return f"""    <!-- Task - Original user request -->
    <section class="report-section mb-8" data-step="goal">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Goal</h2>
        <div class="card p-6" style="border-left: 4px solid {GRAY_900};">
            <pre class="raw-output text-sm text-gray-700">{escaped_task}</pre>
        </div>
    </section>"""


def _generate_judge_criteria_section(result: ConsensusResult) -> str:
    """Generate judge criteria section - shown first to establish evaluation framework."""
    if not result.judge_criteria_used:
        return ""

    criteria_items: list[str] = []
    for jc in result.judge_criteria_used:
        threshold_color = _get_score_color(jc.threshold)
        criteria_items.append(f"""            <div class="border border-gray-200 p-4 hover:border-gray-400 transition-colors">
                <div class="flex items-start justify-between gap-4">
                    <div class="flex-1">
                        <div class="flex items-center gap-2 mb-2">
                            <span class="font-semibold text-gray-900">{html.escape(jc.name)}</span>
                            <span class="status-badge" style="background: {GRAY_100}; color: {GRAY_700};">Weight: {(f"{jc.weight:.1f}" if jc.weight is not None else "N/A")}</span>
                        </div>
                        <p class="text-sm text-gray-600">{html.escape(jc.description)}</p>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold" style="color: {threshold_color}">{jc.threshold:.0%}</div>
                        <div class="text-xs text-gray-500">Threshold</div>
                    </div>
                </div>
            </div>""")

    return f"""    <!-- Judge Criteria - What we're evaluating -->
    <section class="report-section mb-8" data-step="acceptance">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Acceptance Criteria <span class="text-sm text-gray-500">({len(result.judge_criteria_used)} criteria)</span></h2>
        <div class="card p-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
{"\n".join(criteria_items)}
            </div>
        </div>
    </section>"""


def _generate_summary_section(result: ConsensusResult) -> str:
    """Generate summary section with quality scores, model weights, and consensus table."""
    confidence_color = _get_score_color(result.consensus_confidence)
    judge_color = _get_score_color(result.judge_score)

    # Build 3-column consensus table: Agreements | Conflicts | Uncertainties
    # Format each column's content with reasoning and confidence/importance
    agreements_html = (
        "\n".join(
            f'<div class="mb-3 p-2 border-l-2 border-green-400">'
            f'<div class="font-medium text-sm">• {html.escape(a.point)}</div>'
            f'<div class="text-xs text-gray-500 mt-1">{html.escape(a.reasoning or "")}</div>'
            f'<div class="text-xs text-gray-400 mt-1">Confidence: {(f"{a.confidence:.0%}" if a.confidence is not None else "N/A")}</div>'
            f"</div>"
            for a in result.key_agreements
        )
        if result.key_agreements
        else '<div class="text-gray-400 italic">None</div>'
    )

    conflicts_html = (
        "\n".join(
            f'<div class="mb-3 p-2 border-l-2 border-amber-400">'
            f'<div class="font-medium text-sm">• {html.escape(c.topic)}</div>'
            f'<div class="text-xs text-gray-600 mt-1">{html.escape(c.resolution)}</div>'
            f'<div class="text-xs text-gray-500 mt-1 italic">{html.escape(c.resolution_rationale)}</div>'
            f"</div>"
            for c in result.resolved_conflicts
        )
        if result.resolved_conflicts
        else '<div class="text-gray-400 italic">None</div>'
    )

    uncertainties_html = (
        "\n".join(
            f'<div class="mb-3 p-2 border-l-2 border-gray-300">'
            f'<div class="font-medium text-sm">• {html.escape(u.area)}</div>'
            + (
                f'<div class="text-xs text-gray-500 mt-1">{html.escape(u.reasoning)}</div>'
                if u.reasoning
                else ""
            )
            + f'<div class="text-xs text-gray-400 mt-1">Importance: {(f"{u.importance:.0%}" if u.importance is not None else "N/A")}</div>'
            f"</div>"
            for u in result.remaining_uncertainties
        )
        if result.remaining_uncertainties
        else '<div class="text-gray-400 italic">None</div>'
    )

    return f"""    <!-- Summary with Quality Scores, Model Weights, and Consensus Table -->
    <section class="report-section mb-8" data-step="summary">
        <div class="card p-6 mb-6">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <!-- Quality Scores -->
                <div>
                    <h3 class="text-base font-semibold text-gray-900 mb-4">Quality Scores</h3>
                    <div class="flex items-center justify-center gap-8">
                        <div class="text-center">
                            <svg width="100" height="100" viewBox="0 0 100 100" class="mx-auto">
                                <circle cx="50" cy="50" r="40" fill="none" stroke="{GRAY_200}" stroke-width="8"/>
                                <circle cx="50" cy="50" r="40" fill="none" stroke="{confidence_color}" stroke-width="8"
                                    class="score-ring" stroke-dasharray="{251 * result.consensus_confidence} 251"
                                    transform="rotate(-90 50 50)"/>
                                <text x="50" y="50" text-anchor="middle" dy="0.35em" class="text-lg font-bold" fill="{GRAY_900}">{result.consensus_confidence:.0%}</text>
                            </svg>
                            <div class="mt-1 text-xs font-medium text-gray-600">Consensus Confidence</div>
                        </div>
                        <div class="text-center">
                            <svg width="100" height="100" viewBox="0 0 100 100" class="mx-auto">
                                <circle cx="50" cy="50" r="40" fill="none" stroke="{GRAY_200}" stroke-width="8"/>
                                <circle cx="50" cy="50" r="40" fill="none" stroke="{judge_color}" stroke-width="8"
                                    class="score-ring" stroke-dasharray="{251 * result.judge_score} 251"
                                    transform="rotate(-90 50 50)"/>
                                <text x="50" y="50" text-anchor="middle" dy="0.35em" class="text-lg font-bold" fill="{GRAY_900}">{result.judge_score:.0%}</text>
                            </svg>
                            <div class="mt-1 text-xs font-medium text-gray-600">Judge Score</div>
                        </div>
                    </div>
                </div>

                <!-- Model Weights -->
                <div>
                    <h3 class="text-base font-semibold text-gray-900 mb-4">Model Weights</h3>
                    <div class="flex items-center justify-center gap-6">
                        <div id="model-chart"></div>
                        <div id="model-legend" class="space-y-2"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Consensus Table - 3 columns: Agreements | Conflicts | Uncertainties -->
        <div class="card p-6">
            <h3 class="text-base font-semibold text-gray-900 mb-4">Consensus Summary</h3>
            <table class="consensus-table">
                <thead>
                    <tr>
                        <th style="width: 33%; padding: 0.75rem 1rem; background: {SUCCESS_GREEN_LIGHT}; color: {SUCCESS_GREEN};">Agreements ({len(result.key_agreements)})</th>
                        <th style="width: 33%; padding: 0.75rem 1rem; background: {WARNING_AMBER_LIGHT}; color: {WARNING_AMBER};">Conflicts ({len(result.resolved_conflicts)})</th>
                        <th style="width: 33%; padding: 0.75rem 1rem; background: {GRAY_100}; color: {GRAY_500};">Uncertainties ({len(result.remaining_uncertainties)})</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="vertical-align: top; padding: 1rem; background: {SUCCESS_GREEN_LIGHT}20;" class="text-sm text-gray-700">
                            {agreements_html}
                        </td>
                        <td style="vertical-align: top; padding: 1rem; background: {WARNING_AMBER_LIGHT}20;" class="text-sm text-gray-700">
                            {conflicts_html}
                        </td>
                        <td style="vertical-align: top; padding: 1rem; background: {GRAY_100}40;" class="text-sm text-gray-700">
                            {uncertainties_html}
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </section>"""


def _generate_raw_outputs_section(result: ConsensusResult) -> str:
    """Generate model results section - shows what each model produced."""
    if not result.raw_generation_outputs:
        return ""

    outputs_html: list[str] = []
    for gen_output in result.raw_generation_outputs:
        escaped_output = html.escape(gen_output.output)

        # Metadata badges
        badges_html = f"""                <div class="flex flex-wrap items-center gap-2">
                    <span class="text-sm font-medium text-gray-900">{html.escape(gen_output.model_name)}</span>
                    <span class="status-badge" style="background: {GRAY_100}; color: {GRAY_700};">{gen_output.importance * 100:.0f}% weight</span>
                    {f'<span class="status-badge" style="background: {GRAY_100}; color: {GRAY_700};">{html.escape(gen_output.perspective)}</span>' if gen_output.perspective else ""}
                </div>"""

        assumptions_html = ""
        if gen_output.assumptions:
            assumptions_list: list[str] = []
            for a in gen_output.assumptions:
                assumptions_list.append(
                    f'<li class="flex items-start gap-2 py-1"><span class="text-gray-400">-</span>'
                    f"<span>{html.escape(a)}</span></li>"
                )
            assumptions_html = f"""                <div class="mt-3 pt-3 border-t border-gray-200">
                    <h4 class="text-xs font-semibold text-gray-700 mb-2">Assumptions ({len(gen_output.assumptions)})</h4>
                    <ul class="space-y-3 text-sm text-gray-600">
                        {"".join(assumptions_list)}
                    </ul>
                </div>"""

        alternatives_html = ""
        if gen_output.considered_alternatives:
            alternatives_list: list[str] = []
            for a in gen_output.considered_alternatives:
                alternatives_list.append(f"""<li class="border border-gray-200 p-3">
                    <div class="font-medium text-gray-900 text-sm">{html.escape(a.approach)}</div>
                    <div class="text-xs text-gray-500 mt-1">Value: {(f"{a.potential_value:.0%}" if a.potential_value is not None else "N/A")}</div>
                    <div class="text-xs text-gray-600 mt-1 italic">{html.escape(a.why_not_chosen or "N/A")}</div>
                </li>""")
            alternatives_html = f"""                <div class="mt-3 pt-3 border-t border-gray-200">
                    <h4 class="text-xs font-semibold text-gray-700 mb-2">Alternatives Considered ({len(gen_output.considered_alternatives)})</h4>
                    <ul class="space-y-3 text-sm">
                        {"".join(alternatives_list)}
                    </ul>
                </div>"""

        confidence_html = ""
        if gen_output.confidence_breakdown:
            cb = gen_output.confidence_breakdown
            confidence_html = f"""                <div class="mt-3 pt-3 border-t border-gray-200">
                    <h4 class="text-xs font-semibold text-gray-700 mb-2">Confidence Breakdown</h4>
                    <div class="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                        <div class="flex items-center gap-2">
                            <span class="text-gray-500 w-28">Factual:</span>
                            <div class="flex items-center gap-2 flex-1">
                                <div class="flex-1 bg-gray-200 h-1.5">
                                    <div class="bg-gray-700 h-1.5" style="width: {cb.factual_accuracy * 100:.0f}%"></div>
                                </div>
                                <span class="text-gray-700">{cb.factual_accuracy:.0%}</span>
                            </div>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-gray-500 w-28">Complete:</span>
                            <div class="flex items-center gap-2 flex-1">
                                <div class="flex-1 bg-gray-200 h-1.5">
                                    <div class="bg-gray-700 h-1.5" style="width: {cb.completeness * 100:.0f}%"></div>
                                </div>
                                <span class="text-gray-700">{cb.completeness:.0%}</span>
                            </div>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-gray-500 w-28">Logical:</span>
                            <div class="flex items-center gap-2 flex-1">
                                <div class="flex-1 bg-gray-200 h-1.5">
                                    <div class="bg-gray-700 h-1.5" style="width: {cb.logical_coherence * 100:.0f}%"></div>
                                </div>
                                <span class="text-gray-700">{cb.logical_coherence:.0%}</span>
                            </div>
                        </div>
                        <div class="flex items-center gap-2">
                            <span class="text-gray-500 w-28">Relevant:</span>
                            <div class="flex items-center gap-2 flex-1">
                                <div class="flex-1 bg-gray-200 h-1.5">
                                    <div class="bg-gray-700 h-1.5" style="width: {cb.relevance_to_task * 100:.0f}%"></div>
                                </div>
                                <span class="text-gray-700">{cb.relevance_to_task:.0%}</span>
                            </div>
                        </div>
                    </div>
                </div>"""

        insights_html = ""
        if gen_output.key_insights:
            insights_list: list[str] = [
                f'<li class="flex items-start gap-2 py-2 border-l-2 border-gray-300 pl-2">'
                f'<span class="text-gray-500">+</span>'
                f"<div>"
                f'<span class="font-medium">{html.escape(i.insight)}</span>'
                + (
                    f'<div class="text-xs text-gray-500 mt-1">{html.escape(i.reasoning)}</div>'
                    if i.reasoning
                    else ""
                )
                + f'<div class="text-xs text-gray-400">Importance: {(f"{i.importance:.0%}" if i.importance is not None else "N/A")}{f" | Evidence: {html.escape(i.evidence)}" if i.evidence else ""}</div>'
                f"</div>"
                f"</li>"
                for i in gen_output.key_insights
            ]
            insights_html = f"""                <div class="mt-3 pt-3 border-t border-gray-200">
                    <h4 class="text-xs font-semibold text-gray-700 mb-2">Key Insights ({len(gen_output.key_insights)})</h4>
                    <ul class="space-y-1 text-sm text-gray-600">
                        {"".join(insights_list)}
                    </ul>
                </div>"""

        outputs_html.append(f"""        <div class="card overflow-hidden">
            <button class="accordion-header w-full flex items-center justify-between p-4 text-left">
                <div class="flex-1">
{badges_html}
                </div>
                <svg class="accordion-icon w-5 h-5 text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                </svg>
            </button>
            <div class="accordion-content">
                <div class="px-4 pb-4 border-t border-gray-200">
                    <div class="mt-3">
                        <h4 class="text-xs font-semibold text-gray-700 mb-2">Output</h4>
                        <pre class="raw-output bg-gray-50 p-3 mt-2">{escaped_output}</pre>
                    </div>
{assumptions_html}
{alternatives_html}
{confidence_html}
{insights_html}
                </div>
            </div>
        </div>""")

    return f"""    <!-- Model Results - What each model produced -->
    <section class="report-section mb-8" data-step="results">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Model Results <span class="text-sm text-gray-500">({len(result.raw_generation_outputs)} models)</span></h2>
        <div class="space-y-4">
{"".join(outputs_html)}
        </div>
    </section>"""



def _generate_critique_matrix_section(result: ConsensusResult) -> str:
    """Generate critique matrix visualization showing strengths/weaknesses between models with verified claims."""
    if not result.critique_matrix:
        return ""

    # Generate table rows
    table_rows: list[str] = []
    for cs in result.critique_matrix:
        critique_type = "Self" if cs.is_self_critique else "Peer"
        quality_color = _get_score_color(cs.overall_quality_score)

        # Build summary text
        summary_parts: list[str] = []

        if cs.strengths:
            top_strength = cs.strengths[0]
            summary_parts.append(f"<strong>+</strong> {html.escape(top_strength.description)}")

        if cs.weaknesses:
            top_weakness = cs.weaknesses[0]
            summary_parts.append(f"<strong>-</strong> {html.escape(top_weakness.description)}")

        if cs.flawed_assumptions:
            fa = cs.flawed_assumptions[0]
            summary_parts.append(
                f"<strong>!</strong> {html.escape(fa.assumption)} ({(f'{fa.importance:.0%}' if fa.importance is not None else 'N/A')} importance)"
            )

        summary_text = (
            "<br/>".join(summary_parts)
            if summary_parts
            else "<span class='text-gray-400 italic'>No details</span>"
        )

        table_rows.append(f"""            <tr>
                <td>
                    <div class="font-medium text-gray-900">{html.escape(cs.reviewer_name)}</div>
                    <div class="text-xs text-gray-500">reviewing</div>
                    <div class="font-medium text-gray-700">{html.escape(cs.target_name)}</div>
                </td>
                <td><span class="status-badge">{critique_type}</span></td>
                <td>
                    <div class="flex items-center gap-2">
                        <div class="flex-1 bg-gray-200 h-2 w-16">
                            <div class="h-2" style="background-color: {quality_color}; width: {cs.agreement_level * 100:.0f}%"></div>
                        </div>
                        <span class="text-sm font-medium">{cs.agreement_level:.0%}</span>
                    </div>
                </td>
                <td>
                    <div class="text-lg font-bold" style="color: {quality_color}">{cs.overall_quality_score:.0%}</div>
                </td>
                <td class="text-sm text-gray-600 max-w-md">
                    {summary_text}
                </td>
            </tr>""")

    return f"""    <!-- Critique Matrix - How models critiqued each other -->
    <section class="report-section mb-8" data-step="gossip">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Gossip <span class="text-sm text-gray-500">({len(result.critique_matrix)} reviews)</span></h2>
        <div class="card overflow-hidden">
            <table class="consensus-table">
                <thead>
                    <tr>
                        <th style="width: 140px;">Reviewer &rarr; Target</th>
                        <th style="width: 80px;">Type</th>
                        <th style="width: 120px;">Agreement</th>
                        <th style="width: 80px;">Quality</th>
                        <th>Key Points</th>
                    </tr>
                </thead>
                <tbody>
{"".join(table_rows)}
                </tbody>
            </table>
        </div>
    </section>"""


def _generate_judge_results_section(result: ConsensusResult) -> str:
    """Generate detailed judge results section with per-iteration verdicts."""
    if not result.judge_results:
        return ""

    iterations_html: list[str] = []
    for jr in result.judge_results:
        iteration_color = GRAY_900 if jr.passed else GRAY_500
        status_text = "PASSED" if jr.passed else "NEEDS REFINEMENT"
        status_badge = f"background: {GRAY_200}; color: {GRAY_700};"

        verdicts_parts: list[str] = []
        for v in jr.verdicts:
            verdict_color = GRAY_900 if v.passed else GRAY_500
            verdict_items: list[str] = [
                f"""                <div class="border border-gray-200 p-3" style="border-left: 4px solid {verdict_color}">
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center gap-2">
                            <span class="font-medium text-gray-900">{html.escape(v.criterion_name)}</span>
                            <span class="status-badge" style="background: {GRAY_200}; color: {GRAY_700};">{"Pass" if v.passed else "Fail"}</span>
                        </div>
                        <span class="text-lg font-bold" style="color: {_get_score_color(v.score)}">{(f"{v.score:.0%}" if v.score is not None else "N/A")}</span>
                    </div>
                    <p class="text-sm text-gray-600">{html.escape(v.reasoning)}</p>"""
            ]
            if v.specific_issues:
                issues_list = "\n".join(
                    f'<li class="flex items-start gap-2 py-2 border-l-2 border-red-300 pl-2">'
                    f'<span class="text-gray-600">-</span>'
                    f"<div>"
                    f'<span class="text-sm font-medium">{html.escape(i.issue)}</span>'
                    + (
                        f'<div class="text-xs text-gray-500">{html.escape(i.reasoning)}</div>'
                        if i.reasoning
                        else ""
                    )
                    + f'<div class="text-xs text-gray-400">Severity: {(f"{i.importance:.0%}" if i.importance is not None else "N/A")}</div>'
                    f"</div>"
                    f"</li>"
                    for i in v.specific_issues
                )
                verdict_items.append(f"""                    <div class="mt-2 pt-2 border-t border-gray-200">
                        <h5 class="text-xs font-semibold text-gray-700 mb-1">Issues Found</h5>
                        <ul class="space-y-1">{issues_list}</ul>
                    </div>""")
            if v.improvement_suggestions:
                suggestions_list = "\n".join(
                    f'<li class="flex items-start gap-2 py-2 border-l-2 border-blue-300 pl-2">'
                    f'<span class="text-gray-600">-></span>'
                    f"<div>"
                    f'<span class="text-sm font-medium">{html.escape(s.suggestion)}</span>'
                    + (
                        f'<div class="text-xs text-gray-500">{html.escape(s.reasoning)}</div>'
                        if s.reasoning
                        else ""
                    )
                    + f'<div class="text-xs text-gray-400">Expected impact: {(f"{s.importance:.0%}" if s.importance is not None else "N/A")}</div>'
                    f"</div>"
                    f"</li>"
                    for s in v.improvement_suggestions
                )
                verdict_items.append(f"""                    <div class="mt-2 pt-2 border-t border-gray-200">
                        <h5 class="text-xs font-semibold text-gray-700 mb-1">Suggested Improvements</h5>
                        <ul class="space-y-1">{suggestions_list}</ul>
                    </div>""")
            verdict_items.append("                </div>")
            verdicts_parts.append("".join(verdict_items))

        refinements_html = ""
        if jr.refinement_actions:
            actions_list: list[str] = []
            for ra in jr.refinement_actions:
                status_icon = "✓" if ra.was_addressed else "✗"
                status_color = "green" if ra.was_addressed else "red"
                importance_badge = (
                    f"Severity: {(f'{ra.importance:.0%}' if ra.importance is not None else 'N/A')}"
                )

                action_parts = [
                    f'<span style="color: {status_color}; font-weight: bold;">{status_icon}</span>',
                    f'<span class="font-medium text-gray-900">{html.escape(ra.criterion_name)}</span>',
                    f'<span class="status-badge" style="background: {GRAY_200}; color: {GRAY_700};">{importance_badge}</span>',
                ]

                if ra.was_addressed:
                    if ra.reasoning:
                        action_parts.append(
                            f'<span class="text-sm text-gray-700">{html.escape(ra.reasoning)}</span>'
                        )
                    if ra.changes_made:
                        action_parts.append(
                            f'<span class="text-xs text-gray-500 italic">{html.escape(ra.changes_made)}</span>'
                        )
                else:
                    if ra.reasoning:
                        action_parts.append(
                            f'<span class="text-sm text-gray-600">{html.escape(ra.reasoning)}</span>'
                        )

                actions_list.append(
                    f'<li class="flex items-start gap-2 py-2 border-l-2" style="border-color: {status_color if ra.was_addressed else GRAY_400}">'
                    f'<div class="flex-1 space-y-1">'
                    f"{' '.join(action_parts)}"
                    f"</div>"
                    f"</li>"
                )
            refinements_html = f"""                <div class="mt-4 p-3 bg-gray-50">
                    <h4 class="text-sm font-semibold text-gray-800 mb-2">Refinement Actions ({len(jr.refinement_actions)} issues)</h4>
                    <ul class="space-y-2">
                        {"".join(actions_list)}
                    </ul>
                </div>"""

        refined_output_html = ""
        if jr.refined_output:
            refined_output_html = f"""                <div class="mt-4 p-3 bg-gray-50">
                    <h4 class="text-sm font-semibold text-gray-800 mb-2">Refined Output</h4>
                    <pre class="raw-output text-sm text-gray-700">{html.escape(jr.refined_output)}</pre>
                </div>"""

        iterations_html.append(f"""        <div class="card p-5 mb-4" style="border-left: 4px solid {iteration_color}">
            <div class="flex items-center justify-between mb-4">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 flex items-center justify-center text-white font-bold" style="background-color: {iteration_color};">
                        {jr.iteration}
                    </div>
                    <div>
                        <h3 class="font-semibold text-gray-900">Iteration {jr.iteration}</h3>
                        <div class="flex items-center gap-2 mt-1">
                            <span class="status-badge" style="{status_badge}">{status_text}</span>
                            <span class="text-sm text-gray-500">Overall Score:</span>
                            <span class="text-xl font-bold" style="color: {_get_score_color(jr.overall_score)}">{jr.overall_score:.0%}</span>
                        </div>
                    </div>
                </div>
            </div>
            <div>
                <h4 class="text-sm font-semibold text-gray-800 mb-3">Criterion Verdicts</h4>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
{"".join(verdicts_parts)}
                </div>
            </div>
{refinements_html}
{refined_output_html}
        </div>""")

    return f"""    <!-- Judge Results - Decision iterations -->
    <section class="report-section mb-8" data-step="evaluation">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Evaluation <span class="text-sm text-gray-500">({len(result.judge_results)} iteration{"" if len(result.judge_results) == 1 else "s"})</span></h2>
        <div class="space-y-4">
{"".join(iterations_html)}
        </div>
    </section>"""


def _generate_conflict_resolutions_section(result: ConsensusResult) -> str:
    """Generate conflict resolutions section."""
    if not result.conflict_resolutions:
        return ""

    cards_html: list[str] = []
    for cr in result.conflict_resolutions:
        positions_list: list[str] = []
        for model, position in cr.conflicting_positions.items():
            positions_list.append(f"""                <div class="border border-gray-200 p-3 bg-gray-50">
                    <div class="font-medium text-gray-900 text-sm">{html.escape(model)}</div>
                    <div class="text-gray-700 text-sm mt-1">{html.escape(position)}</div>
                </div>""")

        winning_html = ""
        if cr.winning_model:
            winning_html = f"""                <div class="mt-3">
                    <span class="text-sm text-gray-600">Winning Position:</span>
                    <span class="ml-2 status-badge" style="background: {GRAY_200}; color: {GRAY_900};">{html.escape(cr.winning_model)}</span>
                </div>"""

        cards_html.append(f"""        <div class="card p-4" style="border-left: 4px solid {GRAY_500};">
            <h3 class="font-semibold text-gray-900 mb-3">
                <span class="text-lg font-bold text-gray-700">!</span> {html.escape(cr.topic)}
            </h3>
            <div class="mb-4">
                <h4 class="text-sm font-semibold text-gray-800 mb-2">Conflicting Positions</h4>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
{"".join(positions_list)}
                </div>
            </div>
            <div class="p-3 bg-gray-100">
                <h4 class="text-sm font-semibold text-gray-800 mb-2">Resolution</h4>
                <p class="text-sm text-gray-700 mb-2">{html.escape(cr.resolution)}</p>
                <p class="text-sm text-gray-600 mb-2"><span class="font-medium">Rationale:</span> {html.escape(cr.resolution_rationale)}</p>
{winning_html}
            </div>
        </div>""")

    return f"""    <!-- Conflict Resolutions -->
    <section class="report-section mb-8" data-step="conflicts">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Conflict Resolutions <span class="text-sm text-gray-500">({len(result.conflict_resolutions)} conflict{"" if len(result.conflict_resolutions) == 1 else "s"})</span></h2>
        <div class="grid grid-cols-1 gap-4">
{"".join(cards_html)}
        </div>
    </section>"""


def generate_consensus_report_html(
    result: ConsensusResult,
    title: str = "Consensus Decision Report",
) -> str:
    """Generate a complete HTML report for the consensus process."""
    # Prepare model data for chart
    model_data: list[dict[str, Any]] = [
        {
            "name": mc.model_name,
            "importance": mc.importance,
            "perspective": mc.perspective or "General",
            "contributions": [
                {
                    "contribution": c.contribution,
                    "reasoning": c.reasoning,
                    "importance": c.importance,
                    "uniqueness": c.uniqueness,
                }
                for c in mc.key_contributions
            ],
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

    # Build the header
    header = HTML_REPORT_HEADER_TEMPLATE.format(
        title=_escape_js_string(title),
        subtitle=f"{len(result.model_contributions)} models, {result.refinement_iterations} iteration{'' if result.refinement_iterations == 1 else 's'}",
        confidence_color=confidence_color,
        judge_color=judge_color,
        confidence_score=f"{result.consensus_confidence:.0%}",
        judge_score=f"{result.judge_score:.0%}",
        PRIMARY_YELLOW=PRIMARY_YELLOW,
    )

    # Build step navigation
    step_nav = _generate_step_navigation()

    # Build all sections
    sections = "\n".join(
        filter(
            None,
            [
                _generate_task_section(result),
                _generate_judge_criteria_section(result),
                _generate_raw_outputs_section(result),
                _generate_summary_section(result),
                _generate_critique_matrix_section(result),
                _generate_judge_results_section(result),
                _generate_conflict_resolutions_section(result),
            ],
        )
    )

    # Build the final output section
    final_output_section = f"""    <!-- Final Consensus Output -->
    <section class="report-section" data-step="decision">
        <h2 class="text-lg font-semibold text-gray-900 mb-4">Decision</h2>
        <div class="card p-6" style="border-left: 4px solid {GRAY_900};">
            <pre class="raw-output">{escaped_output}</pre>
        </div>
    </section>"""

    # Build the footer
    footer = HTML_REPORT_FOOTER.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Build the D3 chart script
    d3_script = D3_CHART_INIT_SCRIPT.format(
        models_json=models_json,
        chart_colors_json=chart_colors_json,
        WHITE=WHITE,
    )

    # Combine everything - D3 script goes BEFORE footer (which closes HTML)
    return f"""{
        HTML_REPORT_HEAD.format(
            title=_escape_js_string(title),
            PRIMARY_YELLOW=PRIMARY_YELLOW,
            PRIMARY_YELLOW_DARK="#E6B800",
            PRIMARY_YELLOW_LIGHT="#FFE066",
            WHITE=WHITE,
            GRAY_50=GRAY_50,
            GRAY_100=GRAY_100,
            GRAY_200=GRAY_200,
            GRAY_300=GRAY_300,
            GRAY_400=GRAY_400,
            GRAY_500=GRAY_500,
            GRAY_600=GRAY_600,
            GRAY_700=GRAY_700,
            GRAY_800=GRAY_800,
            GRAY_900=GRAY_900,
            SUCCESS_GREEN=SUCCESS_GREEN,
            SUCCESS_GREEN_LIGHT=SUCCESS_GREEN_LIGHT,
            WARNING_AMBER=WARNING_AMBER,
            WARNING_AMBER_LIGHT=WARNING_AMBER_LIGHT,
        )
    }
{header}
{step_nav}
<div class="max-w-6xl mx-auto px-4 py-8">
{sections}
{final_output_section}
</div>
{d3_script}
{footer}
</main>"""


def export_consensus_report_to_html(
    result: ConsensusResult,
    output_path: str | Path,
    title: str = "Consensus Decision Report",
) -> Path:
    """Export the consensus report to an HTML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    html_content = generate_consensus_report_html(result, title=title)
    path.write_text(html_content, encoding="utf-8")

    return path
