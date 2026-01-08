"""Prompt templates for consensus hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import (
    ConfidenceBreakdown,
    ConsensusResult,
    CritiqueFeedback,
    HITLQuestionnaire,
    JudgeResult,
    JudgeVerdict,
    UncertaintyArea,
)

TRIAGE_EXAMPLES = """
<examples>
    <example input="Hello, how are you?">
        <requires_consensus>false</requires_consensus>
        <reason>Simple greeting, no research needed</reason>
        <category>greeting</category>
    </example>
    <example input="Yes, please proceed with the implementation">
        <requires_consensus>false</requires_consensus>
        <reason>User confirmation to proceed, no analysis needed</reason>
        <category>confirmation</category>
    </example>
    <example input="I want you to perform consensus on this topic [topic]">
        <requires_consensus>true</requires_consensus>
        <reason>User explicitly requested consensus</reason>
        <category>consensus_request</category>
    </example>
    <example input="Can you explain what you meant by X?">
        <requires_consensus>false</requires_consensus>
        <reason>Clarification question about previous response</reason>
        <category>clarification</category>
    </example>
    <example input="I need to implement a new authentication system">
        <requires_consensus>true</requires_consensus>
        <reason>Complex implementation requiring research and planning</reason>
        <category>implementation</category>
    </example>
    <example input="Analyze the codebase and find security vulnerabilities">
        <requires_consensus>true</requires_consensus>
        <reason>Deep analysis task requiring multiple perspectives</reason>
        <category>research</category>
    </example>
</examples>
"""


def build_triage_instructions(main_agent_context: str) -> str:
    return f"""{main_agent_context}

<triage_instructions>
    <task>Quickly determine if this request requires multi-model consensus research.</task>
    <thinking_process>
        Before making a triage decision, you MUST use <thinking> tags to analyze:
        1. What type of request is this? (greeting, confirmation, question, task, etc.)
        2. Does it require deep research, multiple perspectives, or complex analysis?
        3. Could a simple, direct response adequately address this request?
        4. Is there explicit language requesting consensus or research?
        5. What is the potential value of consensus vs. direct response?
    </thinking_process>
    <requirements>
        <requirement>ALWAYS use <thinking> tags before your decision to show your analysis</requirement>
        <requirement>Analyze the user's request to understand its nature</requirement>
        <requirement>Determine if it needs deep research, analysis, or implementation planning</requirement>
        <requirement>Direct requests to perform consensus should trigger consensus</requirement>
        <requirement>Simple requests should NOT go through consensus</requirement>
    </requirements>
    <skip_consensus_for>
        <type>Greetings and casual conversation (hello, hi, how are you)</type>
        <type>Simple confirmations (yes, no, proceed, go ahead, ok)</type>
        <type>Clarification questions about previous responses</type>
        <type>Requests to summarize or repeat information</type>
        <type>Simple factual questions with straightforward answers</type>
        <type>Thank you messages or acknowledgments</type>
        <type>Small talk or off-topic chat</type>
    </skip_consensus_for>
    <require_consensus_for>
        <type>Complex research questions requiring multiple perspectives</type>
        <type>Implementation planning for code changes</type>
        <type>Analysis of codebases, architectures, or systems</type>
        <type>Problem-solving requiring deep thought</type>
        <type>Tasks that benefit from critique and synthesis</type>
        <type>Requests explicitly asking for research or analysis</type>
    </require_consensus_for>
    <output_format>
        <thinking>
        [Your analysis of the request type and whether consensus is needed]
        </thinking>

        [Your triage decision]
    </output_format>
</triage_instructions>
"""


def build_generation_instructions(main_agent_context: str, perspective: str | None) -> str:
    perspective_xml = ""
    if perspective:
        perspective_xml = f"<model_perspective>{perspective}</model_perspective>"

    return f"""{main_agent_context}

<generation_instructions>
    <task>Generate a response to given task with step-by-step reasoning.</task>
    {perspective_xml}
    <thinking_process>
        Before generating your output, you MUST use <thinking> tags to plan your approach:
        1. Analyze the user's request and identify the core question
        2. Consider multiple approaches and their trade-offs
        3. Identify what assumptions you need to make
        4. Plan the structure of your response
        5. Identify key insights unique to your perspective
    </thinking_process>
    <requirements>
        <requirement>ALWAYS use <thinking> tags before your output to show your reasoning process</requirement>
        <requirement>DO NOT use markdown formatting - use plain text with proper spacing and newlines</requirement>
        <requirement>Consider multiple approaches before settling on one</requirement>
        <requirement>Document your assumptions explicitly with criticality scores</requirement>
        <requirement>Break down your confidence by different aspects (factual, completeness, coherence, relevance)</requirement>
        <requirement>Identify key insights you bring to this task</requirement>
    </requirements>
    <output_format>
        <thinking>
        [Your step-by-step reasoning and planning goes here]
        </thinking>

        [Your actual output goes here - NO markdown, just structured text]
    </output_format>
</generation_instructions>
"""


def build_generation_example() -> str:
    return """
<example task="What is the best programming language for beginners?">
    <output>Python is the best programming language for beginners because of its readable syntax, extensive learning resources, and versatile applications.</output>
    <assumptions>
        <assumption criticality="0.7" assumption="Beginner means someone with no prior programming experience" />
        <assumption criticality="0.5" assumption="User wants a general-purpose language, not domain-specific" />
    </assumptions>
    <considered_alternatives>
        <alternative potential_value="0.6" approach="JavaScript - widely used for web development" why_not_chosen="Async patterns and browser APIs add complexity for beginners" />
        <alternative potential_value="0.4" approach="Scratch - visual programming for absolute beginners" why_not_chosen="Limited real-world applicability after learning" />
    </considered_alternatives>
    <confidence_breakdown factual_accuracy="0.85" completeness="0.75" logical_coherence="0.90" relevance_to_task="0.95" />
    <key_insights>
        <key_insight importance="0.9" insight="Learning curve matters more than language power for beginners" reasoning="Beginners are more likely to give up if they encounter too much complexity early on" evidence="Python's popularity in educational settings like Code.org and university intro courses" />
        <key_insight importance="0.7" insight="Community support accelerates learning" reasoning="Having access to help and resources reduces frustration and speeds up problem-solving" evidence="Stack Overflow shows Python has the most beginner questions with high-quality answers" />
    </key_insights>
</example>
"""


def build_critique_instructions(
    main_agent_context: str,
    critique_type: str,
    perspective_text: str,
    importance_weight: float,
    final_instruction: str,
) -> str:
    return f"""{main_agent_context}

<critique_instructions type="{critique_type}">
    <reviewer_context>
        <perspective>{perspective_text}</perspective>
        <importance_weight>{importance_weight}</importance_weight>
    </reviewer_context>
    <thinking_process>
        Before critiquing, you MUST use <thinking> tags to plan your analysis:
        1. Identify what the target output is trying to achieve
        2. Scan for strengths - what does this output do well?
        3. Scan for weaknesses - what could be improved?
        4. Look for blind spots - what's missing or overlooked?
        5. Check assumptions - are they valid or flawed?
        6. Extract factual claims and verify each one
        7. Consider alternative approaches
    </thinking_process>
    <analysis_requirements>
        <requirement>ALWAYS use <thinking> tags before your critique to show your analysis process</requirement>
        <requirement>DO NOT use markdown formatting - use plain text with proper spacing and newlines</requirement>
        <requirement>Identify specific strengths with evidence from the output</requirement>
        <requirement>Identify specific weaknesses with evidence from the output</requirement>
        <requirement>Note any missing considerations or blind spots</requirement>
        <requirement>Evaluate validity of stated assumptions</requirement>
        <requirement>Suggest concrete, actionable improvements</requirement>
        <requirement>Consider alternative approaches that might be better</requirement>
        <requirement>VERIFY factual claims: Extract key factual statements and verify each for accuracy</requirement>
        <requirement>For each claim, determine if accurate and provide correction if not</requirement>
    </analysis_requirements>
    <guidance>{final_instruction}</guidance>
    <output_format>
        <thinking>
        [Your analysis and planning goes here - identify strengths, weaknesses, blind spots, verify claims]
        </thinking>

        [Your structured critique output goes here]
    </output_format>
</critique_instructions>
"""


def build_critique_example(is_self: bool, target_model: str) -> str:
    return f"""
<example>
    <is_self_critique>{str(is_self).lower()}</is_self_critique>
    <target_model>{target_model}</target_model>
    <critique type="example" reviewer="example" quality_score="0.70" agreement="0.75">
        <strengths>
            <strength importance="0.8" description="Clear, well-structured response with logical flow" />
        </strengths>
        <weaknesses>
            <weakness importance="0.6" description="Missing consideration of edge cases" />
        </weaknesses>
        <missing_considerations>
            <missing_consideration importance="0.7" consideration="Performance implications not addressed" />
        </missing_considerations>
        <flawed_assumptions>
            <flawed_assumption importance="0.5" assumption="Assumes all users have admin privileges" />
        </flawed_assumptions>
        <suggested_improvements>
            <improvement impact="0.7" what_to_change="Add error handling section" />
        </suggested_improvements>
    </critique>
</example>
"""


def build_synthesis_instructions(
    main_agent_context: str, weights_xml: str, hitl_feedback: str | None = None
) -> str:
    hitl_section = ""
    if hitl_feedback:
        hitl_section = f"""
    <hitl_user_feedback>
        <description>User has provided feedback to resolve uncertainties. Incorporate this into your synthesis.</description>
{hitl_feedback}
    </hitl_user_feedback>"""

    return f"""{main_agent_context}

<synthesis_instructions>
    <task>Synthesize multiple model outputs into a final consensus.</task>
    <model_weights>
{weights_xml}
    </model_weights>{hitl_section}
    <thinking_process>
        Before synthesizing, you MUST use <thinking> tags to plan your approach:
        1. Read through all model outputs and understand their key points
        2. Review all critiques - what strengths and weaknesses were identified?
        3. Identify conflicts between models - where do they disagree?
        4. Determine which insights from each model are most valuable
        5. Plan how to resolve conflicts based on model weights and critique validity
        6. Identify areas of strong agreement vs remaining uncertainty
    </thinking_process>
    <requirements>
        <requirement>ALWAYS use <thinking> tags before your synthesis to show your reasoning process</requirement>
        <requirement>DO NOT use markdown formatting - use plain text with proper spacing and newlines</requirement>
        <requirement>Analyze all outputs and critiques thoroughly</requirement>
        <requirement>Weight contributions by model importance</requirement>
        <requirement>Resolve conflicts considering importance, critique feedback, and self-critique admissions</requirement>
        <requirement>Incorporate best insights from all models</requirement>
        <requirement>Document rejected approaches with reasons</requirement>
        <requirement>Identify areas of strong agreement vs remaining uncertainty</requirement>
    </requirements>
    <goal>Create a synthesis that is BETTER than any individual output.</goal>
    <output_format>
        <thinking>
        [Your synthesis planning goes here - analyze outputs, identify conflicts, plan resolution strategy]
        </thinking>

        [Your synthesized consensus output goes here]
    </output_format>
</synthesis_instructions>
"""


def build_synthesis_example() -> str:
    return """
<example task="Best programming language for web development">
    <synthesized_output>JavaScript/TypeScript is the recommended choice for web development due to its universal browser support, rich ecosystem, and full-stack capabilities with Node.js. For type safety and larger projects, TypeScript adds significant value.</synthesized_output>
    <synthesis_approach>Weighted combination prioritizing practical applicability</synthesis_approach>
    <incorporated_insights>
        <insight from_model="ModelA" weight_applied="0.4" how_used="Adopted TypeScript recommendation for type safety"/>
        <insight from_model="ModelB" weight_applied="0.35" how_used="Incorporated ecosystem analysis"/>
    </incorporated_insights>
    <conflict_resolutions>
        <conflict_resolution topic="TypeScript vs JavaScript" resolution="TypeScript for larger projects, JavaScript for prototypes" resolution_rationale="Balance type safety with development speed" />
    </conflict_resolutions>
    <rejected_approaches>
        <rejected_approach from_model="ModelC" approach="WebAssembly exclusively" rejection_reason="Limited browser API access and steep learning curve" />
    </rejected_approaches>
    <consensus_confidence>0.85</consensus_confidence>
    <areas_of_strong_agreement>
        <agreement_point confidence="0.95" point="JavaScript ecosystem maturity" reasoning="All models cited npm package availability and community size as key factors" />
        <agreement_point confidence="0.90" point="Full-stack capability importance" reasoning="Models agreed that using one language across stack reduces complexity" />
    </areas_of_strong_agreement>
        <uncertainty_area importance="0.6" area="Best framework choice (React vs Vue vs Svelte)" reasoning="Models provided conflicting recommendations based on different criteria" />
</example>
"""


def build_hitl_instructions(hitl_max_questions: int) -> str:
    return f"""
<hitl_questionnaire_instructions>
    <task>Additionally, generate a questionnaire for human-in-the-loop feedback.</task>
    <thinking_process>
        Before generating questions, you MUST use <thinking> tags to analyze:
        1. What are the genuine uncertainties in the synthesis?
        2. Which uncertainties can be resolved by user input vs. which require more research?
        3. Is the synthesis confidence high enough to skip HITL entirely?
        4. What specific information would be most valuable from the user?
        5. How can questions be framed to get actionable, clear responses?
    </thinking_process>
    <requirements>
        <requirement>ALWAYS use <thinking> tags before generating questions to analyze uncertainties</requirement>
        <requirement>ONLY generate questions if there are genuine uncertainties that user input can resolve</requirement>
        <requirement>If synthesis confidence is high (>90%) and no significant uncertainties, set hitl_questionnaire to null or leave questions empty</requirement>
        <requirement>Maximum {hitl_max_questions} questions</requirement>
        <requirement>Each question should have clear, actionable options derived from the uncertainties</requirement>
        <requirement>DO NOT fabricate questions - only ask about real uncertainties</requirement>
    </requirements>
    <output_format>
        <thinking>
        [Your analysis of uncertainties and which questions would be most valuable]
        </thinking>

        [Your questionnaire output - or indication that HITL should be skipped]
    </output_format>
</hitl_questionnaire_instructions>
"""


def build_judge_instructions(main_agent_context: str) -> str:
    return f"""{main_agent_context}

<judge_instructions>
    <task>Evaluate synthesis quality against defined criteria and provide verdict.</task>
    <thinking_process>
        Before evaluating, you MUST use <thinking> tags to analyze:
        1. Review each evaluation criterion - what does it measure?
        2. Examine the synthesis output systematically against each criterion
        3. Identify specific evidence of strengths and weaknesses for each criterion
        4. Consider the threshold requirements - is the output close or far from passing?
        5. If refinement is needed, what specific issues should be addressed?
    </thinking_process>
    <requirements>
        <requirement>ALWAYS use <thinking> tags before your verdict to show your evaluation process</requirement>
        <requirement>Evaluate each criterion independently with specific reasoning</requirement>
        <requirement>Provide concrete evidence for scores, not vague assessments</requirement>
        <requirement>Identify specific issues that prevent criteria from passing</requirement>
        <requirement>Be fair but rigorous - maintain quality standards</requirement>
    </requirements>
    <output_format>
        <thinking>
        [Your systematic evaluation of the synthesis against each criterion]
        </thinking>

        [Your structured verdict with scores and reasoning]
    </output_format>
</judge_instructions>
"""


def build_judge_example() -> str:
    return """
<example>
    <verdicts>
        <verdict criterion_name="Accuracy" score="0.85" passed="true" reasoning="Factual claims are well-supported and accurate" />
        <verdict criterion_name="Clarity" score="0.72" passed="false" reasoning="Some sections lack specific examples" specific_issues="Lacked examples for type safety, Missing framework comparison details" />
    </verdicts>
    <overall_score>0.79</overall_score>
    <meets_threshold>false</meets_threshold>
    <requires_refinement>true</requires_refinement>
</example>
"""


def build_refinement_instructions(main_agent_context: str, failed_criteria_xml: str) -> str:
    return f"""{main_agent_context}

<refinement_instructions>
    <task>Refine consensus output to address failed quality criteria.</task>
    <failed_criteria>
        {failed_criteria_xml}
    </failed_criteria>
    <thinking_process>
        Before refining, you MUST use <thinking> tags to plan your approach:
        1. Review each failed criterion and understand why it failed
        2. Analyze the original output to identify what needs to change
        3. Plan specific fixes for each issue - what concrete changes will you make?
        4. Identify which issues you CAN fix vs which you CANNOT fix (be honest)
        5. Consider any trade-offs - fixing one issue might affect another
        6. Plan how to maintain the key insights while improving quality
    </thinking_process>
    <requirements>
        <requirement>ALWAYS use <thinking> tags before refining to show your planning process</requirement>
        <requirement>DO NOT use markdown formatting - use plain text with proper spacing and newlines</requirement>
        <requirement>Maintain all key insights from original output</requirement>
        <requirement>Focus specifically on fixing the identified issues</requirement>
        <requirement>Preserve overall structure and flow</requirement>
        <requirement>Be honest about which issues you can and cannot address</requirement>
    </requirements>
    <output_format>
        <thinking>
        [Your refinement planning goes here - analyze failures, plan fixes, identify what's possible]
        </thinking>

        [Your structured refinement result output goes here]
    </output_format>
</refinement_instructions>
"""


def build_refinement_example() -> str:
    return """
<description>
    Return a structured refinement result that includes:
    - The refined output text (improved version)
    - For each issue that WAS addressed: explain how and what changes were made
    - For each issue that was NOT addressed: explain why (e.g., insufficient info, trade-offs)
    - Any other improvements made beyond the specific criteria
    Be honest about limitations.
</description>
<example>
    <failed_criteria>
        <failed_criterion name="Completeness" score="0.5" reasoning="Lacks specific examples"/>
        <failed_criterion name="Accuracy" score="0.4" reasoning="Contains outdated information"/>
    </failed_criteria>
    <refinement_result>
        <refined_output>Python is ideal for beginners due to its readable syntax that resembles English, extensive documentation and tutorials, active community support, and versatile applications from web development to data science.</refined_output>
        <issues_addressed>
            <issue_addressed criterion_name="Completeness" original_issue="Lacks specific examples" how_fixed="Added specific examples like documentation, community support, and applications" changes_made="Expanded from 6 words to 30+ words with concrete examples" />
        </issues_addressed>
        <issues_not_addressed>
            <issue_not_addressed criterion_name="Accuracy" original_issue="Contains outdated information" why_not_addressed="Cannot verify current information without external sources" suggested_alternative="Consult official Python documentation for latest version details" />
        </issues_not_addressed>
        <additional_improvements>Improved sentence flow and readability</additional_improvements>
    </refinement_result>
</example>
"""


# ============================================================================
# XML Construction Helper Functions
# ============================================================================


def build_critique_target_xml(
    target_config_name: str,
    target_perspective: str,
    target_importance: float,
    target_output: str,
    assumptions: list[str],
    considered_alternatives: list[Any],
    key_insights: list[Any],
    confidence_breakdown: ConfidenceBreakdown,
    html_escape_func: Callable[[str], str],
) -> str:
    """Build the XML input for critique target."""
    assumptions_xml = (
        " ".join(f'<assumption assumption="{html_escape_func(a or "")}" />' for a in assumptions)
        if assumptions
        else '<assumption assumption="None documented" />'
    )

    if considered_alternatives:
        alternatives_xml = " ".join(
            f'<considered_alternative potential_value="{(f"{a.potential_value:.2f}" if a.potential_value is not None else "N/A")}" approach="{html_escape_func(a.approach or "")}" why_not_chosen="{html_escape_func(a.why_not_chosen or "This was chosen")}" />'
            for a in considered_alternatives
        )
    else:
        alternatives_xml = '<considered_alternative approach="None documented" />'

    if not key_insights:
        insights_xml = "        <key_insight>None documented</key_insight>"
    else:
        insights_parts: list[str] = []
        for ki in key_insights:
            reasoning_attr = (
                f' reasoning="{html_escape_func(ki.reasoning or "")}"' if ki.reasoning else ""
            )
            evidence_attr = (
                f' evidence="{html_escape_func(ki.evidence or "")}"' if ki.evidence else ""
            )
            insights_parts.append(
                f"""        <key_insight importance="{ki.importance:.2f}" insight="{html_escape_func(ki.insight or "")}"{reasoning_attr}{evidence_attr} />"""
            )
        insights_xml = "\n".join(insights_parts)

    return f"""
    <critique_target>
        <model name="{target_config_name}" perspective="{target_perspective}" importance="{target_importance}" />
        <output_to_critique>{target_output}</output_to_critique>
        <assumptions>{assumptions_xml}</assumptions>
        <alternatives_considered>{alternatives_xml}</alternatives_considered>
        <key_insights>
{insights_xml}
        </key_insights>
        <confidence_breakdown factual_accuracy="{confidence_breakdown.factual_accuracy:.2f}" completeness="{confidence_breakdown.completeness:.2f}" logical_coherence="{confidence_breakdown.logical_coherence:.2f}" relevance_to_task="{confidence_breakdown.relevance_to_task:.2f}" />
    </critique_target>
    """


def build_model_output_xml(
    config_name: str,
    config_importance: float,
    perspective: str,
    output: str,
    key_insights: list[Any],
    html_escape_func: Callable[[str], str],
) -> str:
    """Build XML for a single model's output in synthesis phase."""
    if not key_insights:
        insights_xml = "            <key_insight>None</key_insight>"
    else:
        insights_parts: list[str] = []
        for ki in key_insights:
            reasoning_attr = (
                f' reasoning="{html_escape_func(ki.reasoning or "")}"' if ki.reasoning else ""
            )
            evidence_attr = (
                f' evidence="{html_escape_func(ki.evidence or "")}"' if ki.evidence else ""
            )
            insights_parts.append(
                f"""            <key_insight importance="{ki.importance:.2f}" insight="{html_escape_func(ki.insight or "")}"{reasoning_attr}{evidence_attr} />"""
            )
        insights_xml = "\n".join(insights_parts)

    return f"""        <model_output name="{config_name}" importance="{config_importance}" perspective="{perspective}">
            <output>{html_escape_func(output or "")}</output>
            <key_insights>
{insights_xml}
            </key_insights>
        </model_output>"""


def build_critique_feedback_xml(
    reviewer_name: str,
    is_self_critique: bool,
    feedback: CritiqueFeedback,
    html_escape_func: Callable[[str], str],
) -> str:
    """Build XML for a single critique feedback entry."""
    ctype = "self" if is_self_critique else "peer"

    strengths_xml = (
        " ".join(
            f'<strength importance="{s.importance:.2f}" description="{html_escape_func(s.description or "")}" />'
            for s in feedback.strengths
        )
        if feedback.strengths
        else '<strength description="None identified" />'
    )

    weaknesses_xml = (
        " ".join(
            f'<weakness importance="{w.importance:.2f}" description="{html_escape_func(w.description or "")}" />'
            for w in feedback.weaknesses
        )
        if feedback.weaknesses
        else '<weakness description="None identified" />'
    )

    # Build optional sections
    optional_sections: list[str] = []
    if feedback.missing_considerations:
        missing_xml = " ".join(
            f'<missing_consideration importance="{m.importance:.2f}" consideration="{html_escape_func(m.consideration or "")}" />'
            for m in feedback.missing_considerations
        )
        optional_sections.append(f"<missing_considerations>{missing_xml}</missing_considerations>")

    if feedback.flawed_assumptions:
        flawed_xml = " ".join(
            f'<flawed_assumption importance="{f.importance:.2f}" assumption="{html_escape_func(f.assumption or "")}" />'
            for f in feedback.flawed_assumptions
        )
        optional_sections.append(f"<flawed_assumptions>{flawed_xml}</flawed_assumptions>")

    if feedback.suggested_improvements:
        improvements_xml = " ".join(
            f'<improvement impact="{i.expected_impact:.2f}" what_to_change="{html_escape_func(i.what_to_change or "")}" />'
            for i in feedback.suggested_improvements
        )
        optional_sections.append(
            f"<suggested_improvements>{improvements_xml}</suggested_improvements>"
        )

    optional_xml = " ".join(optional_sections) if optional_sections else ""

    return f"""<critique type="{ctype}" reviewer="{reviewer_name}" quality_score="{feedback.overall_quality_score:.2f}" agreement="{feedback.agreement_level:.2f}">
    <strengths>{strengths_xml}</strengths>
    <weaknesses>{weaknesses_xml}</weaknesses>{f" {optional_xml}" if optional_xml else ""}
</critique>"""


def build_failed_criteria_xml(failed_criteria: list[JudgeVerdict]) -> str:
    """Build XML for failed judge criteria."""
    return "\n".join(
        f'            <fail name="{v.criterion_name}" score="{(f"{v.score:.2f}" if v.score is not None else "N/A")}" reason="{v.reasoning}" />'
        for v in failed_criteria
    )


# ============================================================================
# Consensus Result Injection XML Helper Functions
# ============================================================================


def build_contributions_xml(
    model_contributions: list[Any], html_escape_func: Callable[[str], str]
) -> str:
    """Build XML for model contributions."""
    return "\n".join(
        f'        <model_contribution name="{mc.model_name}" importance="{(f"{mc.importance:.0%}" if mc.importance is not None else "N/A")}" insights_incorporated="{mc.insights_incorporated}" insights_rejected="{mc.insights_rejected}" perspective="{html_escape_func(mc.perspective or "General")}" key_contributions="{", ".join(c.contribution for c in mc.key_contributions) if mc.key_contributions else "None"}" />'
        for mc in model_contributions
    )


def build_agreements_xml(agreements: list[Any]) -> str:
    """Build XML for key agreements."""
    return (
        "\n".join(
            f'        <agreement_point confidence="{(f"{a.confidence:.0%}" if a.confidence is not None else "N/A")}" point="{a.point}" reasoning="{a.reasoning}" />'
            for a in agreements
        )
        if agreements
        else "        <agreement_point>None</agreement_point>"
    )


def build_conflicts_xml(conflicts: list[Any]) -> str:
    """Build XML for resolved conflicts."""
    return (
        "\n".join(
            f'        <resolved_conflict topic="{cr.topic}" resolution="{cr.resolution}" resolution_rationale="{cr.resolution_rationale}" />'
            for cr in conflicts
        )
        if conflicts
        else "        <resolved_conflict>None</resolved_conflict>"
    )


def build_uncertainties_xml(uncertainties: list[Any]) -> str:
    """Build XML for remaining uncertainties."""
    return (
        "\n".join(
            f'        <remaining_uncertainty importance="{(f"{u.importance:.0%}" if u.importance is not None else "N/A")}" area="{u.area}" reasoning="{u.reasoning}" />'
            for u in uncertainties
        )
        if uncertainties
        else "        <remaining_uncertainty>None</remaining_uncertainty>"
    )


def build_judge_iteration_xml(
    judge_result: JudgeResult, html_escape_func: Callable[[str], str]
) -> str:
    """Build XML for a single judge iteration."""
    # Build verdicts with optional issues
    verdicts_parts: list[str] = []
    for v in judge_result.verdicts:
        issues_attr = (
            f' specific_issues="{", ".join(si.issue for si in v.specific_issues)}"'
            if v.specific_issues
            else ""
        )
        verdicts_parts.append(
            f"""                <verdict criterion_name="{v.criterion_name}" score="{(f"{v.score:.2f}" if v.score is not None else "N/A")}" passed="{v.passed}" reasoning="{v.reasoning}"{issues_attr} />"""
        )
    verdicts_xml = "\n".join(verdicts_parts)

    # Build refinements with optional reasoning and changes
    if judge_result.refinement_actions:
        refinements_parts: list[str] = []
        for ra in judge_result.refinement_actions:
            reasoning_attr = f' reasoning="{ra.reasoning}"' if ra.reasoning else ""
            changes_attr = f' changes_made="{ra.changes_made}"' if ra.changes_made else ""
            refinements_parts.append(
                f"""                <refinement_action criterion_name="{ra.criterion_name}" was_addressed="{str(ra.was_addressed).lower()}" importance="{(f"{ra.importance:.2f}" if ra.importance is not None else "N/A")}" issue_description="{ra.issue_description}"{reasoning_attr}{changes_attr} />"""
            )
        refinements_xml = "\n".join(refinements_parts)
    else:
        refinements_xml = "                <refinement_action>None</refinement_action>"

    # Build the iteration XML with optional refined output
    refined_output_xml = (
        f"                <refined_output>{html_escape_func(judge_result.refined_output)}</refined_output>"
        if judge_result.refined_output
        else ""
    )

    return f"""            <iteration number="{judge_result.iteration}" overall_score="{(f"{judge_result.overall_score:.2f}" if judge_result.overall_score is not None else "N/A")}" passed="{judge_result.passed}">
                <verdicts>
{verdicts_xml}
                </verdicts>
                <refinement_actions>
{refinements_xml}
                </refinement_actions>
{refined_output_xml}
            </iteration>"""


def build_judge_results_xml(
    judge_results: list[JudgeResult], html_escape_func: Callable[[str], str]
) -> str:
    """Build XML for all judge results."""
    if not judge_results:
        return "            <iteration>None</iteration>"

    iterations_xml = "\n".join(
        build_judge_iteration_xml(jr, html_escape_func) for jr in judge_results
    )
    return iterations_xml


def build_consensus_result_xml(
    consensus_result: ConsensusResult, html_escape_func: Callable[[str], str]
) -> str:
    """Build the complete consensus result XML for injection."""
    contributions_xml = build_contributions_xml(
        consensus_result.model_contributions, html_escape_func
    )
    agreements_xml = build_agreements_xml(consensus_result.key_agreements)
    conflicts_xml = build_conflicts_xml(consensus_result.resolved_conflicts)
    uncertainties_xml = build_uncertainties_xml(consensus_result.remaining_uncertainties)
    judge_results_xml = build_judge_results_xml(consensus_result.judge_results, html_escape_func)

    return f"""
    <multi_model_consensus confidence="{consensus_result.consensus_confidence:.2%}" judge_score="{consensus_result.judge_score:.2%}" iterations="{consensus_result.refinement_iterations}">
        <explanation>
            Multi-model consensus: GENERATED independently, CRITIQUED by all models,
            SYNTHESIZED with weighting, JUDGED until quality thresholds met.
            Use this as authoritative context. Build upon it rather than contradicting.
        </explanation>
        <final_output>{consensus_result.final_output}</final_output>
        <model_contributions>
{contributions_xml}
        </model_contributions>
        <key_agreements>
{agreements_xml}
        </key_agreements>
        <resolved_conflicts>
{conflicts_xml}
        </resolved_conflicts>
        <remaining_uncertainties>
{uncertainties_xml}
        </remaining_uncertainties>
        <judge_evaluation_details>
{judge_results_xml}
        </judge_evaluation_details>
        <process_summary generation="{consensus_result.generation_summary}" critique="{consensus_result.critique_summary}" synthesis="{consensus_result.synthesis_summary}" judge="{consensus_result.judge_summary}" />
    </multi_model_consensus>
    """


def build_synthesis_context_xml(user_input: str, outputs_xml: str, critiques_xml: str) -> str:
    """Build the synthesis context XML wrapper."""
    return f"""
    <synthesis_context task="{user_input}">
        <model_outputs>
{outputs_xml}
        </model_outputs>
        <critiques>
{critiques_xml}
        </critiques>
    </synthesis_context>
    """


def build_refinement_context_xml(
    current_output: str, hitl_questionnaire: HITLQuestionnaire | None = None
) -> str:
    """Build the refinement context XML wrapper.

    Args:
        current_output: The current output to refine.
        hitl_questionnaire: Optional HITL questionnaire to include in the context.
    """
    hitl_xml = ""
    if hitl_questionnaire and hitl_questionnaire.questions:
        # Build XML for questions
        questions_xml: list[str] = []
        for q in hitl_questionnaire.questions:
            options_xml = "\n                ".join(
                [
                    f'<option id="{opt.option_id}"><label>{opt.label}</label>'
                    + (f"<description>{opt.description}</description>" if opt.description else "")
                    + "</option>"
                    for opt in q.options
                ]
            )
            questions_xml.append(f"""
                <question id="{q.question_id}">
                    <text>{q.question_text}</text>
                    <options>
                        {options_xml}
                    </options>
                </question>""")

        all_questions_xml = "\n            ".join(questions_xml)

        hitl_xml = f"""
                <hitl_questionnaire>
                    <synthesis_confidence>{hitl_questionnaire.synthesis_confidence:.0%}</synthesis_confidence>
                        <description>Review the questions below to understand the uncertainties</description>
                    <questions>
                        {all_questions_xml}
                    </questions>
                    <instructions>
                        When refining the output, you can use get_user_input to ask the user for feedback on these uncertainties.
                        The questionnaire above shows the key areas where the synthesis lacks confidence.
                        Use get_user_input to gather user preferences or clarification on these specific questions.
                    </instructions>
                </hitl_questionnaire>"""

    return f"""
            <refinement_context>
                <original_output>{current_output}</original_output>
                {hitl_xml}
            </refinement_context>
            """


def build_hitl_toolkit_instructions() -> str:
    """Build instructions for the Consensus HITL toolkit.

    Returns instructions for gathering human feedback during consensus refinement.
    Uses Agno's native Dynamic User Input pattern with chain-of-thought guidance.
    """
    return """<consensus_hitl>
    <purpose>Gather human feedback to improve the consensus output during refinement iterations.</purpose>

    <thinking_before_acting>
        <step>Think carefully about what information you need from the user</step>
        <step>Review the synthesis uncertainties and questionnaire provided in the context</step>
        <step>Plan your questions before calling get_user_input - be specific and focused</step>
        <step>Consider what follow-up questions might be needed based on potential user responses</step>
        <step>Chain of thought: Analyze uncertainties → Identify needed clarifications → Formulate clear questions → Get user input → Incorporate feedback</step>
    </thinking_before_acting>

    <workflow>
        <step>Review the synthesis uncertainties and questionnaire provided in the context</step>
        <step>Think about what specific information would help resolve these uncertainties</step>
        <step>Use get_user_input to ask the user questions when you need clarification or feedback</step>
        <step>You can call get_user_input multiple times - the system will pause and wait for user input each time</step>
        <step>Incorporate the user's feedback into your refinement of the consensus output</step>
    </workflow>

    <get_user_input_capabilities>
        <purpose>When you have a tool/function where you don't have enough information, don't say you can't do it - use get_user_input to get the information you need from the user.</purpose>
        <usage>Call get_user_input with the fields you require the user to fill in for you to continue your task.</usage>

        <important_guidelines>
            <guideline>**Don't respond and ask the user for information.** Just use the get_user_input tool to get the information you need.</guideline>
            <guideline>**Don't make up information you don't have.** If you don't have the information, use get_user_input to get it from the user.</guideline>
            <guideline>**Include only the required fields.** Only include fields in the user_input_fields parameter that you genuinely need information for.</guideline>
            <guideline>**Provide a clear and concise description of the field.** Clearly describe the field in the field_description parameter.</guideline>
            <guideline>**Provide a type for the field.** Fill the field_type parameter with the type of the field (str, int, float, bool, etc.).</guideline>
        </important_guidelines>

        <input_validation>
            <validation type="boolean">
                <description>For boolean fields, only explicit positive responses are considered True:</description>
                <true_values>'true', 'yes', 'y', '1', 'on', 't', 'True', 'YES', 'Y', 'T'</true_values>
                <false_values>Everything else including 'false', 'no', 'n', '0', 'off', 'f', empty strings, unanswered fields</false_values>
                <critical>Empty/unanswered fields should be treated as False (not selected)</critical>
            </validation>
            <validation type="general">
                <rule>Users can leave fields unanswered - empty responses are valid</rule>
                <rule>NEVER ask for the same field twice - accept whatever the user provides</rule>
                <rule>DO NOT validate or re-request input - accept what the user provides and convert it appropriately</rule>
                <rule>Proceed with only the fields that were explicitly answered as True - skip False/unanswered fields</rule>
                <rule>Complete the task immediately after receiving all user inputs - no confirmation or re-validation</rule>
            </validation>
        </input_validation>
    </get_user_input_capabilities>

    <guidelines>
        <guideline>Only use get_user_input when the questionnaire indicates genuine uncertainty</guideline>
        <guideline>Think first: What specific information do I need? How will it help resolve the uncertainty?</guideline>
        <guideline>Be specific in your questions - reference specific aspects of the synthesis</guideline>
        <guideline>Accept both option selections (a, b, c) and free-form text responses</guideline>
        <guideline>If the user provides feedback, incorporate it into the next refinement iteration</guideline>
        <guideline>Chain of thought: Analyze → Identify needs → Formulate questions → Get input → Incorporate</guideline>
    </guidelines>
</consensus_hitl>"""


def build_critiques_for_wrapper_xml(target_name: str, critique_items: list[str]) -> str:
    """Build XML wrapper for critiques targeting a specific model."""
    critiques_content = "\n".join(critique_items)
    return f"""        <critiques_for target="{target_name}">
{critiques_content}
        </critiques_for>"""


def build_model_weights_xml(normalized_weights: dict[str, float]) -> str:
    """Build XML for model weights in synthesis instructions."""
    weight_items = "\n".join(
        f'        <model_weight name="{name}" weight="{weight:.2f}" />'
        for name, weight in normalized_weights.items()
    )
    return weight_items


def build_consensus_skipped_xml(reason: str, category: str) -> str:
    """Build XML note for when consensus is skipped due to triage."""
    return f"""
    <consensus_skipped>
        <reason>{reason}</reason>
        <category>{category}</category>
        <note>This request was determined to not require multi-model consensus. Proceeding with direct response.</note>
    </consensus_skipped>
    """


def build_hitl_feedback_context_xml(questionnaire: HITLQuestionnaire) -> str:
    """Build XML context for HITL feedback to be incorporated into re-synthesis."""
    questions_xml: list[str] = []
    for q in questionnaire.questions:
        options_xml = "\n            ".join(
            f'<option id="{opt.option_id}" label="{opt.label}"'
            + (f' description="{opt.description}"' if opt.description else "")
            + " />"
            for opt in q.options
        )
        questions_xml.append(f"""        <question id="{q.question_id}">
            <text>{q.question_text}</text>
            <options>
            {options_xml}
            </options>
        </question>""")

    all_questions = "\n".join(questions_xml)

    return f"""        <synthesis_confidence>{questionnaire.synthesis_confidence:.0%}</synthesis_confidence>
        <questions_for_user>
{all_questions}
        </questions_for_user>
        <instruction>Use get_user_input tool to ask the user about these uncertainties before finalizing synthesis.</instruction>"""
