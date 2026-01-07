"""Consensus hooks for Agno agents - Multi-model agreement through structured rounds."""

from __future__ import annotations

import asyncio
import html
from pathlib import Path
from textwrap import dedent
from typing import Any, cast

from agno.agent.agent import Agent
from agno.eval.agent_as_judge import (
    AgentAsJudgeEval,
    AgentAsJudgeEvaluation,
    AgentAsJudgeResult,
)
from agno.models.base import Model
from agno.run.agent import RunInput
from agno.session import AgentSession, TeamSession
from agno.team import Team
from agno.utils.log import log_debug, log_error  # type: ignore
from pydantic import Field, field_validator

from ....concurrency import ConcurrentProcessor
from ....models import BaseModelSerializable, ChainOfThoughts
from ....utils import (
    AgnoPreHook,
    DebugMode,
    UserId,
    create_agent_with_instructions,
    format_main_agent_context,
    inject_context_to_run_input,
)
from .hitl import ConsensusHITLToolkit, HITLQuestionnaire


class KeyInsight(ChainOfThoughts):
    """A key insight with reasoning about why it matters."""

    insight: str = Field(description="The insight itself.")
    evidence: str | None = Field(
        default=None, description="Supporting evidence or examples for this insight."
    )


class MissingConsideration(ChainOfThoughts):
    """An important aspect that was missed, with reasoning about its importance."""

    consideration: str = Field(description="The consideration that was missed.")
    suggested_action: str | None = Field(
        default=None, description="What should be done to address this missing consideration."
    )


class FlawedAssumption(ChainOfThoughts):
    """An incorrect or questionable assumption with explanation."""

    assumption: str = Field(description="The flawed assumption that was made.")
    correct_understanding: str | None = Field(
        default=None, description="What the correct assumption or understanding should be."
    )


class AgreementPoint(ChainOfThoughts):
    """A point of agreement between models with reasoning."""

    point: str = Field(description="The point of agreement.")
    supporting_models: list[str] = Field(
        default=[], description="Names of models that support this agreement."
    )


class UncertaintyArea(ChainOfThoughts):
    """An area of remaining uncertainty with analysis."""

    area: str = Field(description="The area where uncertainty remains.")
    potential_resolution: str | None = Field(
        default=None, description="How this uncertainty might be resolved."
    )


class SpecificIssue(ChainOfThoughts):
    """A specific issue identified during evaluation."""

    issue: str = Field(description="Description of the issue.")
    location: str | None = Field(default=None, description="Where in the output this issue occurs.")


class ImprovementSuggestion(ChainOfThoughts):
    """A suggestion for improvement with expected importance."""

    suggestion: str = Field(description="The improvement suggestion.")
    implementation_hint: str | None = Field(
        default=None, description="Hint on how to implement this improvement."
    )


class Contribution(ChainOfThoughts):
    """A key contribution from a model with importance analysis."""

    contribution: str = Field(description="Description of the contribution.")
    uniqueness: str | None = Field(
        default=None, description="What makes this contribution unique or valuable."
    )


class ConsideredAlternative(BaseModelSerializable):
    approach: str = Field(description="Description of the alternative approach.")
    why_not_chosen: str | None = Field(
        default=None,
        description="Reason this approach was not selected. None if this IS the chosen approach.",
    )
    potential_value: float = Field(
        ge=0.0, le=1.0, description="How valuable this alternative might be (0.0-1.0)."
    )


class ConfidenceBreakdown(ChainOfThoughts):
    factual_accuracy: float = Field(
        ge=0.0, le=1.0, description="Confidence in factual correctness."
    )
    completeness: float = Field(
        ge=0.0, le=1.0, description="Confidence that all aspects are covered."
    )
    logical_coherence: float = Field(
        ge=0.0, le=1.0, description="Confidence in logical consistency."
    )
    relevance_to_task: float = Field(
        ge=0.0, le=1.0, description="Confidence output addresses the task."
    )


class GenerationOutput(ChainOfThoughts):
    output: str = Field(description="The model's main output/answer to the task.")
    assumptions: list[str] = Field(default=[], description="Assumptions made while generating.")
    considered_alternatives: list[ConsideredAlternative] = Field(
        default=[], description="Alternative approaches considered."
    )
    confidence_breakdown: ConfidenceBreakdown = Field(...)
    key_insights: list[KeyInsight] = Field(
        default=[], description="Main insights this model brings with reasoning and impact."
    )


# ============================================================================
class TriageDecision(ChainOfThoughts):
    """Decision on whether a request requires full multi-model consensus research."""

    requires_consensus: bool = Field(
        description="True if the request requires multi-model consensus research, False otherwise."
    )
    reason: str = Field(description="Brief explanation for the decision.")
    category: str = Field(
        description="Category of request: 'research', 'implementation', 'clarification', "
        "'confirmation', 'greeting', 'simple_question', or 'other'."
    )


class StrengthWeakness(BaseModelSerializable):
    description: str = Field(description="Description of the strength or weakness.")
    importance: float = Field(ge=0.0, le=1.0, description="Importance or value score (0.0-1.0).")
    evidence: str = Field(description="Quote or reference from the output.")


class CheckedClaim(BaseModelSerializable):
    claim_text: str = Field(description="The factual claim being checked.")
    is_accurate: bool = Field(description="True if claim is factually accurate.")
    correction: str | None = Field(
        default=None, description="If inaccurate, the correct information."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in verification (0.0-1.0).")


class SuggestedImprovement(BaseModelSerializable):
    what_to_change: str = Field(description="What should be changed.")
    why: str = Field(description="Why this change is needed.")
    proposed_fix: str = Field(description="The proposed fix or improvement.")
    expected_impact: float = Field(ge=0.0, le=1.0, description="Expected impact (0.0-1.0).")


class CritiqueFeedback(ChainOfThoughts):
    is_self_critique: bool = Field(description="True if critiquing own output.")
    target_model: str = Field(description="Name of the model being critiqued.")
    strengths: list[StrengthWeakness] = Field(default=[], description="Identified strengths.")
    weaknesses: list[StrengthWeakness] = Field(default=[], description="Identified weaknesses.")
    missing_considerations: list[MissingConsideration] = Field(
        default=[], description="Important aspects missed with reasoning and impact."
    )
    flawed_assumptions: list[FlawedAssumption] = Field(
        default=[], description="Incorrect assumptions with reasoning and impact."
    )
    suggested_improvements: list[SuggestedImprovement] = Field(
        default=[], description="Improvements."
    )
    alternative_approaches: list[ConsideredAlternative] = Field(
        default=[], description="Better approaches."
    )
    checked_claims: list[CheckedClaim] = Field(
        default=[],
        description="Factual claims checked for accuracy (includes both accurate and inaccurate claims).",
    )
    accurate_claims: list[CheckedClaim] = Field(
        default=[], description="Factual claims that passed verification."
    )
    inaccurate_claims: list[CheckedClaim] = Field(
        default=[], description="Factual claims that failed verification with corrections."
    )
    factual_accuracy_score: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Overall factual accuracy (0.0-1.0)."
    )
    agreement_level: float = Field(ge=0.0, le=1.0, description="Agreement with reviewed output.")
    overall_quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality score of reviewed output."
    )


class ConflictResolution(BaseModelSerializable):
    topic: str = Field(description="What the disagreement was about.")
    conflicting_positions: dict[str, str] = Field(
        description="Each model's position (name -> position)."
    )
    resolution: str = Field(description="How the conflict was resolved.")
    resolution_rationale: str = Field(description="Why this resolution was chosen.")
    winning_model: str | None = Field(default=None, description="Model whose position was adopted.")


class IncorporatedInsight(BaseModelSerializable):
    from_model: str = Field(description="Model that provided this insight.")
    insight: str = Field(description="The insight that was incorporated.")
    weight_applied: float = Field(ge=0.0, le=1.0, description="Weight based on model importance.")
    how_used: str = Field(description="How this insight was used.")


class RejectedApproach(BaseModelSerializable):
    from_model: str = Field(description="Model that proposed this approach.")
    approach: str = Field(description="The rejected approach.")
    rejection_reason: str = Field(description="Why it was rejected.")
    critique_references: list[str] = Field(
        default=[], description="Critiques that led to rejection."
    )


class ConsensusSynthesis(ChainOfThoughts):
    synthesized_output: str = Field(description="The synthesized final output.")
    synthesis_approach: str = Field(description="How the outputs were combined.")
    incorporated_insights: list[IncorporatedInsight] = Field(
        default=[], description="Incorporated insights."
    )
    conflict_resolutions: list[ConflictResolution] = Field(
        default=[], description="Resolved conflicts."
    )
    rejected_approaches: list[RejectedApproach] = Field(
        default=[], description="Rejected approaches."
    )
    model_contribution_weights: dict[str, float] = Field(
        default={}, description="Final weights per model."
    )
    consensus_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in consensus.")
    areas_of_strong_agreement: list[AgreementPoint] = Field(
        default=[], description="Areas of strong agreement with reasoning and confidence."
    )
    areas_of_uncertainty: list[UncertaintyArea] = Field(
        default=[], description="Areas that remain uncertain with reasoning and impact."
    )


class ConsensusSynthesisWithHITL(ConsensusSynthesis):
    """Synthesis with optional HITL questionnaire - generated in single model call.

    When HITL is enabled, the model generates both the synthesis AND the questionnaire
    based on uncertainties in a single call, reducing latency.
    """

    hitl_questionnaire: HITLQuestionnaire | None = Field(
        default=None,
        description="Questionnaire for user feedback. None or empty questions if no uncertainties.",
    )


class JudgeCriteria(BaseModelSerializable):
    name: str = Field(description="Name of the criterion.")
    description: str = Field(description="What this criterion evaluates.")
    weight: float = Field(default=1.0, ge=0.0, description="Relative weight.")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum score to pass.")


class JudgeVerdict(BaseModelSerializable):
    criterion_name: str = Field(description="Name of the criterion evaluated.")
    score: float = Field(ge=0.0, le=1.0, description="Score for this criterion.")
    passed: bool = Field(description="Whether this criterion passed.")
    reasoning: str = Field(description="Reasoning behind the verdict.")
    specific_issues: list[SpecificIssue] = Field(
        default=[], description="Specific issues identified with reasoning and importance."
    )
    improvement_suggestions: list[ImprovementSuggestion] = Field(
        default=[], description="Improvement suggestions with reasoning and expected impact."
    )


class IssueAddressed(ChainOfThoughts):
    """An issue that was successfully addressed during refinement."""

    criterion_name: str = Field(description="Name of the criterion that was failed.")
    original_issue: str = Field(
        description="Description of the original issue from the judge verdict."
    )
    how_fixed: str = Field(
        description="Explanation of how the issue was addressed in the refined output."
    )
    changes_made: str = Field(description="Specific changes made to address this issue.")


class IssueNotAddressed(ChainOfThoughts):
    """An issue that could not be addressed during refinement."""

    criterion_name: str = Field(description="Name of the criterion that was failed.")
    original_issue: str = Field(
        description="Description of the original issue from the judge verdict."
    )
    why_not_addressed: str = Field(
        description="Explanation of why this issue could not be addressed "
        "(e.g., insufficient information, inherent limitation, trade-off with other criteria)."
    )
    suggested_alternative: str | None = Field(
        default=None,
        description="Alternative approach that might address this issue, if any.",
    )


class RefinementResult(ChainOfThoughts):
    """Structured result from the refinement agent."""

    refined_output: str = Field(description="The refined output text.")
    issues_addressed: list[IssueAddressed] = Field(
        default=[],
        description="Issues that were successfully addressed with details on how.",
    )
    issues_not_addressed: list[IssueNotAddressed] = Field(
        default=[],
        description="Issues that could not be addressed with explanations.",
    )
    additional_improvements: list[str] = Field(
        default=[],
        description="Any other improvements made beyond the failed criteria.",
    )


class RefinementAction(BaseModelSerializable):
    """Record of a refinement action taken on an issue."""

    criterion_name: str = Field(description="Name of the criterion.")
    issue_description: str = Field(description="Description of the issue from judge verdict.")
    was_addressed: bool = Field(description="Whether this issue was addressed.")
    importance: float = Field(ge=0.0, le=1.0, description="Importance of the issue (from verdict).")
    reasoning: str | None = Field(
        default=None,
        description="Explanation of how it was fixed (if addressed) or why not (if not addressed).",
    )
    changes_made: str | None = Field(
        default=None,
        description="Specific changes made to address the issue (if addressed).",
    )


class JudgeResult(ChainOfThoughts):
    iteration: int = Field(description="Refinement iteration number.")
    verdicts: list[JudgeVerdict] = Field(description="Verdicts for each criterion.")
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted average score.")
    passed: bool = Field(description="Whether overall evaluation passed.")
    refinement_actions: list[RefinementAction] = Field(
        default=[], description="Refinement actions taken."
    )
    refined_output: str | None = Field(default=None, description="Refined output if performed.")


class ModelContribution(BaseModelSerializable):
    model_name: str = Field(description="Name of the model.")
    importance: float = Field(ge=0.0, le=1.0, description="Model's importance weight.")
    perspective: str = Field(description="Model's unique perspective.")
    key_contributions: list[Contribution] = Field(
        default=[], description="Key contributions with reasoning and impact."
    )
    insights_incorporated: int = Field(default=0, description="Insights incorporated count.")
    insights_rejected: int = Field(default=0, description="Insights rejected count.")


class GenerationOutputSummary(BaseModelSerializable):
    """Full generation output from a single model for reporting."""

    model_name: str = Field(description="Name of the model that generated this output.")
    importance: float = Field(ge=0.0, le=1.0, description="Model's importance weight.")
    perspective: str = Field(default="", description="Model's unique perspective.")
    output: str = Field(description="The model's main output/answer.")
    assumptions: list[str] = Field(default=[], description="Assumptions made.")
    considered_alternatives: list[ConsideredAlternative] = Field(
        default=[], description="Alternative approaches considered."
    )
    confidence_breakdown: ConfidenceBreakdown | None = Field(
        default=None, description="Confidence scores by aspect."
    )
    key_insights: list[KeyInsight] = Field(
        default=[], description="Key insights from this model with reasoning and impact."
    )


class CritiqueSummary(BaseModelSerializable):
    """Summary of a critique between two models for reporting."""

    reviewer_name: str = Field(description="Name of the reviewing model.")
    target_name: str = Field(description="Name of the model being critiqued.")
    is_self_critique: bool = Field(description="True if critiquing own output.")
    strengths: list[StrengthWeakness] = Field(default=[], description="Identified strengths.")
    weaknesses: list[StrengthWeakness] = Field(default=[], description="Identified weaknesses.")
    missing_considerations: list[MissingConsideration] = Field(
        default=[], description="Important aspects missed with reasoning and impact."
    )
    flawed_assumptions: list[FlawedAssumption] = Field(
        default=[], description="Incorrect assumptions with reasoning and impact."
    )
    suggested_improvements: list[SuggestedImprovement] = Field(
        default=[], description="Suggested improvements."
    )
    agreement_level: float = Field(ge=0.0, le=1.0, description="Agreement with reviewed output.")
    overall_quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality score of reviewed output."
    )


class ConsensusResult(BaseModelSerializable):
    user_input: str = Field(description="The original user request/task.")
    final_output: str = Field(description="The final consensus output.")
    consensus_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in consensus.")
    judge_score: float = Field(ge=0.0, le=1.0, description="Final judge score.")
    refinement_iterations: int = Field(description="Number of refinement iterations.")
    model_contributions: list[ModelContribution] = Field(description="Model contributions summary.")
    judge_results: list[JudgeResult] = Field(
        default=[],
        description="Detailed judge results per iteration with verdicts and refinements.",
    )
    key_agreements: list[AgreementPoint] = Field(
        default=[], description="Key agreements with reasoning and confidence."
    )
    resolved_conflicts: list[ConflictResolution] = Field(
        default=[], description="Resolved conflicts with full details."
    )
    remaining_uncertainties: list[UncertaintyArea] = Field(
        default=[], description="Remaining uncertainties with reasoning and impact."
    )
    generation_summary: str = Field(description="Generation phase summary.")
    critique_summary: str = Field(description="Critique phase summary.")
    synthesis_summary: str = Field(description="Synthesis phase summary.")
    judge_summary: str = Field(description="Judge phase summary.")

    raw_generation_outputs: list[GenerationOutputSummary] = Field(
        default=[],
        description="Full generation outputs from each model.",
    )
    critique_matrix: list[CritiqueSummary] = Field(
        default=[],
        description="Complete critique data between all model pairs.",
    )
    incorporated_insights: list[IncorporatedInsight] = Field(
        default=[],
        description="All insights incorporated with weights and usage.",
    )
    conflict_resolutions: list[ConflictResolution] = Field(
        default=[],
        description="Full conflict resolution details with positions.",
    )
    rejected_approaches: list[RejectedApproach] = Field(
        default=[],
        description="Approaches that were rejected with reasons.",
    )
    synthesis_approach: str = Field(
        default="",
        description="How the outputs were combined.",
    )
    judge_criteria_used: list[JudgeCriteria] = Field(
        default=[],
        description="The criteria used for judging.",
    )

    def export_report_to_html(
        self,
        output_path: str | Path,
        title: str = "Consensus Decision Report",
    ) -> Path:
        """Export consensus result to an interactive HTML report with D3.js visualization."""
        from blockether_foundation.agents.hooks.consensus.presentation import (
            export_consensus_report_to_html,
        )

        return export_consensus_report_to_html(self, output_path, title)


class ModelConfig(BaseModelSerializable):
    model: Model = Field(description="The Agno model instance.")
    name: str = Field(description="Name/identifier for this model.")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight factor (0.0-1.0).")
    perspective: str = Field(default="", description="Unique perspective description.")
    tools: list[Any] = Field(default=[], description="Tools available during Generation phase.")
    verification_tools: list[Any] = Field(
        default=[],
        description="Tools available during CoVe Verification phase (e.g., search, RAG).",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("name cannot be empty")
        return v


class ConsensusHooksConfig:
    """Configuration for consensus hooks with multi-model agreement.

    The consensus process runs 5 rounds (after triage):
    0. Triage: Quick check if consensus is even needed (skips for simple requests)
    1. Generation: Each model generates a response independently
    1.5. Verification (CoVe): Extract claims, generate verification questions, answer independently
    2. Critique: Models critique each other's outputs with verification context
    3. Synthesis: Combine outputs into a weighted consensus
    4. Judge & Refine: Evaluate and iteratively refine until quality threshold met

    Args:
        models: List of ModelConfig with model instances, names, importance weights,
            and optional tools for the Generation phase.
        judge_criteria: List of JudgeCriteria for final evaluation
        triage_model: Model used for quick triage decision. If None, uses the first model
            from the models list. Use a fast model (like gpt-5-mini) for efficiency.
        skip_triage: If True, always run consensus without triage check (default: False).
        max_refinement_iterations: Max iterations for judge/refine loop (default: 3)
        judge_threshold: Minimum score to pass judgment (default: 0.7)
        async_hooks: Use async hooks if True, sync wrappers if False (default: True)
        auto_save_html: Auto-save HTML report after consensus (default: False)
        output_directory: Directory for HTML reports (required if auto_save_html=True)
        concurrent_processor: ConcurrentProcessor for parallel model execution.
            If None, a default processor with concurrency=5 is created.
    """

    DEFAULT_CONCURRENCY = 5

    def __init__(
        self,
        models: list[ModelConfig],
        judge_criteria: list[JudgeCriteria],
        triage_model: Model | None = None,
        skip_triage: bool = False,
        max_refinement_iterations: int = 3,
        judge_threshold: float = 0.7,
        async_hooks: bool = True,
        auto_save_html: bool = False,
        output_directory: str | Path | None = None,
        concurrent_processor: ConcurrentProcessor[Any, Any] | None = None,
        hitl: bool = False,
        hitl_max_questions: int = 5,
    ):
        if not models:
            raise ValueError("At least one model must be provided")
        if not judge_criteria:
            raise ValueError("At least one judge criterion must be provided")
        if auto_save_html and output_directory is None:
            raise ValueError("output_directory must be provided when auto_save_html is True")

        self.models = models
        self.judge_criteria = judge_criteria
        self.triage_model = triage_model or models[0].model
        self.skip_triage = skip_triage
        self.max_refinement_iterations = max_refinement_iterations
        self.judge_threshold = judge_threshold
        self.async_hooks = async_hooks
        self.auto_save_html = auto_save_html
        self.output_directory = Path(output_directory) if output_directory else None
        self.hitl = hitl
        self.hitl_max_questions = hitl_max_questions
        self.concurrent_processor = concurrent_processor or ConcurrentProcessor(
            concurrency=self.DEFAULT_CONCURRENCY,
        )

    def pre_hook(self) -> AgnoPreHook:
        return _create_consensus_hook(
            models=self.models,
            judge_criteria=self.judge_criteria,
            triage_model=self.triage_model,
            skip_triage=self.skip_triage,
            max_refinement_iterations=self.max_refinement_iterations,
            judge_threshold=self.judge_threshold,
            async_hooks=not self.async_hooks,
            auto_save_html=self.auto_save_html,
            output_directory=self.output_directory,
            concurrent_processor=self.concurrent_processor,
            hitl=self.hitl,
            hitl_max_questions=self.hitl_max_questions,
        )


async def _run_triage_check(
    triage_model: Model,
    user_input: str,
    debug_mode: DebugMode,
    parent_agent: Agent | Team,
) -> TriageDecision:
    """
    Quick check to determine if a request requires full multi-model consensus.

    Args:
        triage_model: A fast model to use for triage decision.
        user_input: The user's input/request.
        debug_mode: Debug mode settings.
        parent_agent: The main agent/team to extract context from.

    Returns:
        TriageDecision indicating whether consensus is needed.
    """
    log_debug("Running triage check to determine if consensus is needed")

    main_agent_context = format_main_agent_context(parent_agent)

    instructions_text = dedent(f"""
        {main_agent_context}

        <triage_instructions>
            <task>Quickly determine if this request requires multi-model consensus research.</task>
            <requirements>
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
        </triage_instructions>
    """)

    expected_output_text = dedent("""
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
    """)

    agent = create_agent_with_instructions(
        description="Triage Agent",
        instructions=instructions_text,
        expected_output=expected_output_text,
        model=triage_model,
        debug_mode=debug_mode,
        output_schema=TriageDecision,
    )

    response = await agent.arun(user_input)
    decision = cast(TriageDecision, response.content)

    assert decision is not None and isinstance(decision, TriageDecision), (
        "Triage agent did not return TriageDecision"
    )

    return decision


async def _run_round_1_generation(
    models: list[ModelConfig],
    user_input: str,
    debug_mode: DebugMode,
    processor: ConcurrentProcessor[Any, Any],
    parent_agent: Agent | Team | None = None,
) -> list[tuple[ModelConfig, GenerationOutput]]:
    log_debug(f"Round 1: Generation with {len(models)} models")

    main_agent_context = format_main_agent_context(parent_agent) if parent_agent else ""

    async def generate_for_model(
        model_config: ModelConfig,
    ) -> list[tuple[ModelConfig, GenerationOutput]]:
        perspective_xml = ""
        if model_config.perspective:
            perspective_xml = f"<model_perspective>{model_config.perspective}</model_perspective>"

        instructions_text = dedent(f"""
            {main_agent_context}

            <generation_instructions>
                <task>Generate a response to given task with step-by-step reasoning.</task>
                {perspective_xml}
                <requirements>
                    <requirement>DO NOT use markdown formatting - use plain text with proper spacing and newlines</requirement>
                    <requirement>Consider multiple approaches before settling on one</requirement>
                    <requirement>Document your assumptions explicitly with criticality scores</requirement>
                    <requirement>Break down your confidence by different aspects (factual, completeness, coherence, relevance)</requirement>
                    <requirement>Identify key insights you bring to this task</requirement>
                </requirements>
            </generation_instructions>
        """)

        expected_output_text = dedent("""
            <example task="What is the best programming language for beginners?">
                <output>Python is the best programming language for beginners because of its readable syntax, extensive learning resources, and versatile applications.</output>
                <assumptions>
                    <assumption criticality="0.7">Beginner means someone with no prior programming experience</assumption>
                    <assumption criticality="0.5">User wants a general-purpose language, not domain-specific</assumption>
                </assumptions>
                <considered_alternatives>
                    <alternative potential_value="0.6">
                        <approach>JavaScript - widely used for web development</approach>
                        <why_not_chosen>Async patterns and browser APIs add complexity for beginners</why_not_chosen>
                    </alternative>
                    <alternative potential_value="0.4">
                        <approach>Scratch - visual programming for absolute beginners</approach>
                        <why_not_chosen>Limited real-world applicability after learning</why_not_chosen>
                    </alternative>
                </considered_alternatives>
                <confidence_breakdown>
                    <factual_accuracy>0.85</factual_accuracy>
                    <completeness>0.75</completeness>
                    <logical_coherence>0.90</logical_coherence>
                    <relevance_to_task>0.95</relevance_to_task>
                </confidence_breakdown>
                <key_insights>
                    <insight importance="0.9">
                        <insight>Learning curve matters more than language power for beginners</insight>
                        <reasoning>Beginners are more likely to give up if they encounter too much complexity early on</reasoning>
                        <evidence>Python's popularity in educational settings like Code.org and university intro courses</evidence>
                    </insight>
                    <insight importance="0.7">
                        <insight>Community support accelerates learning</insight>
                        <reasoning>Having access to help and resources reduces frustration and speeds up problem-solving</reasoning>
                        <evidence>Stack Overflow shows Python has the most beginner questions with high-quality answers</evidence>
                    </insight>
                </key_insights>
            </example>
        """)

        agent = create_agent_with_instructions(
            description=f"Model {model_config.name}",
            instructions=instructions_text,
            expected_output=expected_output_text,
            model=model_config.model,
            debug_mode=debug_mode,
            output_schema=GenerationOutput,
            tools=model_config.tools,
        )

        response = await agent.arun(user_input)
        output = cast(GenerationOutput, response.content)
        assert output is not None and isinstance(output, GenerationOutput), (
            f"Generation for {model_config.name} did not return GenerationOutput"
        )
        return [(model_config, output)]

    results = await processor.process(models, generate_for_model)
    log_debug(f"Round 1 complete: {len(results)} outputs generated")
    return list(results)


CritiqueTask = tuple[ModelConfig, ModelConfig, GenerationOutput, bool]


async def _run_round_2_critique(
    models: list[ModelConfig],
    generation_outputs: list[tuple[ModelConfig, GenerationOutput]],
    debug_mode: DebugMode,
    processor: ConcurrentProcessor[Any, Any],
    parent_agent: Agent | Team | None = None,
) -> list[tuple[ModelConfig, str, CritiqueFeedback]]:
    log_debug("Round 2: Self + Peer Critique phase (with inline verification)")

    main_agent_context = format_main_agent_context(parent_agent) if parent_agent else ""

    async def generate_critique(
        task: CritiqueTask,
    ) -> list[tuple[ModelConfig, str, CritiqueFeedback]]:
        reviewer, target_config, target_output, is_self = task
        critique_type = "self-critique" if is_self else "peer critique"
        final_instruction = (
            "Be honest about your own shortcomings." if is_self else "Be constructive but thorough."
        )
        perspective_text = reviewer.perspective or "General expert"

        instructions_text = dedent(f"""
            {main_agent_context}

            <critique_instructions type="{critique_type}">
                <reviewer_context>
                    <perspective>{perspective_text}</perspective>
                    <importance_weight>{reviewer.importance}</importance_weight>
                </reviewer_context>
                <analysis_requirements>
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
            </critique_instructions>
        """)

        expected_output_text = dedent(f"""
            <example>
                <is_self_critique>{str(is_self).lower()}</is_self_critique>
                <target_model>{target_config.name}</target_model>
                <strengths>
                    <strength importance_or_value="0.8">
                        <description>Clear, well-structured response with logical flow</description>
                        <evidence>"The response starts with the main point and provides supporting details"</evidence>
                    </strength>
                </strengths>
                <weaknesses>
                    <weakness importance_or_value="0.6">
                        <description>Missing consideration of edge cases</description>
                        <evidence>"The response only covers the happy path scenario"</evidence>
                    </weakness>
                </weaknesses>
                <missing_considerations>
                    <consideration importance="0.7">
                        <consideration>Performance implications not addressed</consideration>
                        <reasoning>For production systems, performance can significantly affect user experience and cost</reasoning>
                        <suggested_action>Add benchmarks or complexity analysis</suggested_action>
                    </consideration>
                </missing_considerations>
                <flawed_assumptions>
                    <assumption importance="0.5">
                        <assumption>Assumes all users have admin privileges</assumption>
                        <reasoning>Most users will have limited permissions in real environments</reasoning>
                        <correct_understanding>Should consider role-based access control scenarios</correct_understanding>
                    </assumption>
                </flawed_assumptions>
                <suggested_improvements>
                    <improvement expected_importance="0.7">
                        <what_to_change>Add error handling section</what_to_change>
                        <why>Improves robustness and production readiness</why>
                        <proposed_fix>Include try-catch blocks and validation examples</proposed_fix>
                    </improvement>
                </suggested_improvements>
                <checked_claims>
                    <claim confidence="0.9">
                        <claim_text>Python was released in 1991</claim_text>
                        <is_accurate>true</is_accurate>
                    </claim>
                    <claim confidence="0.95">
                        <claim_text>JavaScript is the most popular language</claim_text>
                        <is_accurate>false</is_accurate>
                        <correction>JavaScript is one of the most popular languages, but rankings vary by metric</correction>
                    </claim>
                </checked_claims>
                <accurate_claims>
                    <claim>Python was released in 1991</claim>
                </accurate_claims>
                <inaccurate_claims>
                    <claim>JavaScript is the most popular language</claim>
                    <correction>JavaScript is one of the most popular languages, but rankings vary by metric</correction>
                </inaccurate_claims>
                <factual_accuracy_score>0.85</factual_accuracy_score>
                <agreement_level>0.75</agreement_level>
                <overall_quality_score>0.70</overall_quality_score>
            </example>
        """)

        agent = create_agent_with_instructions(
            description=f"Critic {reviewer.name} ({critique_type})",
            instructions=instructions_text,
            expected_output=expected_output_text,
            model=reviewer.model,
            debug_mode=debug_mode,
            output_schema=CritiqueFeedback,
        )

        assumptions_xml = (
            "\n".join(
                f"            <assumption>{html.escape(a)}</assumption>"
                for a in target_output.assumptions
            )
            if target_output.assumptions
            else "            <assumption>None documented</assumption>"
        )

        alternatives_xml = (
            "\n".join(
                f'            <alternative potential_value="{(f"{a.potential_value:.2f}" if a.potential_value is not None else "N/A")}">\n'
                f"                <approach>{a.approach}</approach>\n"
                f"                <why_not_chosen>{a.why_not_chosen or 'This was chosen'}</why_not_chosen>\n"
                f"            </alternative>"
                for a in target_output.considered_alternatives
            )
            if target_output.considered_alternatives
            else "            <alternative>None documented</alternative>"
        )

        insights_xml = (
            "\n".join(
                f'            <insight importance="{ki.importance:.2f}">\n'
                f"                <text>{html.escape(ki.insight)}</text>\n"
                + (
                    f"                <reasoning>{html.escape(ki.reasoning)}</reasoning>\n"
                    if ki.reasoning
                    else ""
                )
                + (
                    f"                <evidence>{html.escape(ki.evidence)}</evidence>\n"
                    if ki.evidence
                    else ""
                )
                + "            </insight>"
                for ki in target_output.key_insights
            )
            if target_output.key_insights
            else "            <insight>None documented</insight>"
        )

        target_perspective = target_config.perspective or "Not specified"

        critique_input = dedent(f"""
        <critique_target>
            <model name="{target_config.name}" perspective="{target_perspective}" importance="{target_config.importance}" />
            <output_to_critique>
{target_output.output}
            </output_to_critique>
            <assumptions>
{assumptions_xml}
            </assumptions>
            <alternatives_considered>
{alternatives_xml}
            </alternatives_considered>
            <key_insights>
{insights_xml}
            </key_insights>
            <confidence_breakdown>
                <factual_accuracy>{target_output.confidence_breakdown.factual_accuracy:.2f}</factual_accuracy>
                <completeness>{target_output.confidence_breakdown.completeness:.2f}</completeness>
                <logical_coherence>{target_output.confidence_breakdown.logical_coherence:.2f}</logical_coherence>
                <relevance_to_task>{target_output.confidence_breakdown.relevance_to_task:.2f}</relevance_to_task>
            </confidence_breakdown>
        </critique_target>
        """)

        response = await agent.arun(critique_input)
        feedback = cast(CritiqueFeedback, response.content)
        assert feedback is not None and isinstance(feedback, CritiqueFeedback), (
            "CritiqueFeedback parsing failed"
        )

        # Split checked_claims into accurate and inaccurate for clarity
        accurate_claims = [c for c in feedback.checked_claims if c.is_accurate]
        inaccurate_claims = [c for c in feedback.checked_claims if not c.is_accurate]

        # Calculate factual accuracy from actual verified claims
        if feedback.checked_claims:
            feedback.factual_accuracy_score = len(accurate_claims) / len(feedback.checked_claims)

        return [(reviewer, target_config.name, feedback)]

    critique_tasks: list[CritiqueTask] = []
    for reviewer in models:
        for target_config, target_output in generation_outputs:
            is_self = reviewer.name == target_config.name
            if is_self:
                continue  # Skip self-critique, only do peer critique with verification
            critique_tasks.append((reviewer, target_config, target_output, is_self))

    critiques = await processor.process(critique_tasks, generate_critique)
    log_debug(f"Round 2 complete: {len(critiques)} critiques generated")
    return list(critiques)


async def _run_round_3_synthesis(
    models: list[ModelConfig],
    generation_outputs: list[tuple[ModelConfig, GenerationOutput]],
    critiques: list[tuple[ModelConfig, str, CritiqueFeedback]],
    user_input: str,
    debug_mode: DebugMode,
    parent_agent: Agent | Team | None = None,
    hitl: bool = False,
    hitl_max_questions: int = 5,
) -> ConsensusSynthesis | ConsensusSynthesisWithHITL:
    log_debug(f"Round 3: Weighted Synthesis phase{' (with HITL)' if hitl else ''}")

    main_agent_context = format_main_agent_context(parent_agent) if parent_agent else ""

    synthesis_model = max(models, key=lambda m: m.importance)

    output_summaries: list[str] = []
    for config, output in generation_outputs:
        perspective = config.perspective or "Not specified"
        insights_xml = (
            "\n".join(
                f'                <insight importance="{ki.importance:.2f}">\n'
                f"                    <text>{html.escape(ki.insight)}</text>\n"
                + (
                    f"                    <reasoning>{html.escape(ki.reasoning)}</reasoning>\n"
                    if ki.reasoning
                    else ""
                )
                + (
                    f"                    <evidence>{html.escape(ki.evidence)}</evidence>\n"
                    if ki.evidence
                    else ""
                )
                + "                </insight>"
                for ki in output.key_insights
            )
            if output.key_insights
            else "                <insight>None</insight>"
        )
        summary = dedent(f"""
        <model_output name="{config.name}" importance="{config.importance}" perspective="{perspective}">
            <output>{html.escape(output.output)}</output>
            <key_insights>
{insights_xml}
            </key_insights>
        </model_output>""")
        output_summaries.append(summary)
    outputs_xml = "\n".join(output_summaries)

    critiques_by_target: dict[str, list[tuple[str, CritiqueFeedback]]] = {}
    for reviewer, target_name, feedback in critiques:
        if target_name not in critiques_by_target:
            critiques_by_target[target_name] = []
        critiques_by_target[target_name].append((reviewer.name, feedback))

    critiques_parts: list[str] = []
    for target_name, target_critiques in critiques_by_target.items():
        critique_items: list[str] = []
        for reviewer_name, fb in target_critiques:
            ctype = "self" if fb.is_self_critique else "peer"

            strengths_xml = (
                "\n".join(
                    f'                    <strength importance="{s.importance:.2f}">'
                    f"{html.escape(s.description)}</strength>"
                    for s in fb.strengths
                )
                if fb.strengths
                else "                    <strength>None identified</strength>"
            )

            weaknesses_xml = (
                "\n".join(
                    f'                    <weakness importance="{w.importance:.2f}">'
                    f"{html.escape(w.description)}</weakness>"
                    for w in fb.weaknesses
                )
                if fb.weaknesses
                else "                    <weakness>None identified</weakness>"
            )

            missing_xml = (
                "\n".join(
                    f'                    <missing importance="{m.importance:.2f}">'
                    f"{html.escape(m.consideration)}</missing>"
                    for m in fb.missing_considerations
                )
                if fb.missing_considerations
                else ""
            )

            flawed_xml = (
                "\n".join(
                    f'                    <flawed_assumption importance="{f.importance:.2f}">'
                    f"{html.escape(f.assumption)}</flawed_assumption>"
                    for f in fb.flawed_assumptions
                )
                if fb.flawed_assumptions
                else ""
            )

            improvements_xml = (
                "\n".join(
                    f'                    <improvement impact="{i.expected_impact:.2f}">'
                    f"{html.escape(i.what_to_change)}</improvement>"
                    for i in fb.suggested_improvements
                )
                if fb.suggested_improvements
                else ""
            )

            critique_content = f"""            <critique type="{ctype}" reviewer="{reviewer_name}" quality_score="{fb.overall_quality_score:.2f}" agreement="{fb.agreement_level:.2f}">
                <strengths>
{strengths_xml}
                </strengths>
                <weaknesses>
{weaknesses_xml}
                </weaknesses>"""

            if missing_xml:
                critique_content += f"""
                <missing_considerations>
{missing_xml}
                </missing_considerations>"""

            if flawed_xml:
                critique_content += f"""
                <flawed_assumptions>
{flawed_xml}
                </flawed_assumptions>"""

            if improvements_xml:
                critique_content += f"""
                <suggested_improvements>
{improvements_xml}
                </suggested_improvements>"""

            critique_content += "\n            </critique>"
            critique_items.append(critique_content)

        critiques_parts.append(
            f'        <critiques_for model="{target_name}">\n'
            + "\n".join(critique_items)
            + "\n        </critiques_for>"
        )
    critiques_xml = "\n".join(critiques_parts)

    model_weights = {config.name: config.importance for config in models}
    total_weight = sum(model_weights.values())
    normalized_weights = {name: w / total_weight for name, w in model_weights.items()}
    weights_xml = "\n".join(
        f'        <model name="{name}" normalized_weight="{weight:.2%}" />'
        for name, weight in normalized_weights.items()
    )

    # Build HITL instructions if enabled
    hitl_instructions = ""
    if hitl:
        hitl_instructions = dedent(f"""
            <hitl_questionnaire_instructions>
                <task>Additionally, generate a questionnaire for human-in-the-loop feedback.</task>
                <requirements>
                    <requirement>ONLY generate questions if there are genuine uncertainties that user input can resolve</requirement>
                    <requirement>If synthesis confidence is high (>90%) and no significant uncertainties, set hitl_questionnaire to null or leave questions empty</requirement>
                    <requirement>Maximum {hitl_max_questions} questions</requirement>
                    <requirement>Each question should have clear, actionable options derived from the uncertainties</requirement>
                    <requirement>DO NOT fabricate questions - only ask about real uncertainties</requirement>
                </requirements>
            </hitl_questionnaire_instructions>
        """)

    instructions_text = dedent(f"""
        {main_agent_context}

        <synthesis_instructions>
            <task>Synthesize multiple model outputs into a final consensus.</task>
            <model_weights>
{weights_xml}
            </model_weights>
            <requirements>
                <requirement>DO NOT use markdown formatting - use plain text with proper spacing and newlines</requirement>
                <requirement>Analyze all outputs and critiques thoroughly</requirement>
                <requirement>Weight contributions by model importance</requirement>
                <requirement>Resolve conflicts considering importance, critique feedback, and self-critique admissions</requirement>
                <requirement>Incorporate best insights from all models</requirement>
                <requirement>Document rejected approaches with reasons</requirement>
                <requirement>Identify areas of strong agreement vs remaining uncertainty</requirement>
            </requirements>
            <goal>Create a synthesis that is BETTER than any individual output.</goal>
        </synthesis_instructions>
        {hitl_instructions}
    """)

    # Build expected output with HITL example if enabled
    hitl_example = ""
    if hitl:
        hitl_example = """
            <hitl_questionnaire>
                <questions>
                    <question>
                        <question_id>framework_preference</question_id>
                        <question_text>Which framework consideration is most important for your project?</question_text>
                        <options>
                            <option option_id="a" label="Enterprise readiness" description="Large team, long-term maintenance"/>
                            <option option_id="b" label="Simplicity" description="Quick development, easy onboarding"/>
                            <option option_id="c" label="Performance" description="Speed and bundle size critical"/>
                        </options>
                        <category>uncertainty</category>
                        <allows_multiple>false</allows_multiple>
                        <related_uncertainty>Best framework choice (React vs Vue vs Svelte)</related_uncertainty>
                    </question>
                </questions>
                <context_summary>Models agree on JavaScript/TypeScript but differ on framework choice.</context_summary>
                <synthesis_confidence>0.85</synthesis_confidence>
                <should_skip_hitl>false</should_skip_hitl>
            </hitl_questionnaire>"""

    hitl_no_questions_example = ""
    if hitl:
        hitl_no_questions_example = """
        <example_no_uncertainties task="What is 2+2?">
            <synthesized_output>The answer is 4.</synthesized_output>
            <synthesis_approach>Direct mathematical calculation</synthesis_approach>
            <consensus_confidence>1.0</consensus_confidence>
            <areas_of_uncertainty></areas_of_uncertainty>
            <hitl_questionnaire>
                <questions></questions>
                <context_summary>Complete agreement, no uncertainties.</context_summary>
                <synthesis_confidence>1.0</synthesis_confidence>
                <should_skip_hitl>true</should_skip_hitl>
            </hitl_questionnaire>
        </example_no_uncertainties>"""

    expected_output_text = dedent(f"""
        <example task="Best programming language for web development">
            <synthesized_output>
                JavaScript/TypeScript is the recommended choice for web development due to its
                universal browser support, rich ecosystem, and full-stack capabilities with Node.js.
                For type safety and larger projects, TypeScript adds significant value.
            </synthesized_output>
            <synthesis_approach>Weighted combination prioritizing practical applicability</synthesis_approach>
            <incorporated_insights>
                <insight from_model="ModelA" weight_applied="0.4" how_used="Adopted TypeScript recommendation for type safety"/>
                <insight from_model="ModelB" weight_applied="0.35" how_used="Incorporated ecosystem analysis"/>
            </incorporated_insights>
            <conflict_resolutions>
                <resolution topic="TypeScript vs JavaScript">
                    <conflicting_positions>
                        <position model="ModelA">TypeScript always</position>
                        <position model="ModelB">JavaScript for simplicity</position>
                    </conflicting_positions>
                    <resolution>TypeScript for larger projects, JavaScript acceptable for prototypes</resolution>
                    <winning_model>ModelA</winning_model>
                </resolution>
            </conflict_resolutions>
            <rejected_approaches>
                <rejected from_model="ModelC" approach="Use WebAssembly exclusively" rejection_reason="Limited browser API access and steeper learning curve"/>
            </rejected_approaches>
            <consensus_confidence>0.85</consensus_confidence>
            <areas_of_strong_agreement>
                <agreement confidence="0.95">
                    <point>JavaScript ecosystem maturity</point>
                    <reasoning>All models cited npm package availability and community size as key factors</reasoning>
                    <supporting_models>ModelA, ModelB, ModelC</supporting_models>
                </agreement>
                <agreement confidence="0.90">
                    <point>Full-stack capability importance</point>
                    <reasoning>Models agreed that using one language across stack reduces complexity</reasoning>
                    <supporting_models>ModelA, ModelB</supporting_models>
                </agreement>
            </areas_of_strong_agreement>
            <areas_of_uncertainty>
                <uncertainty importance="0.6">
                    <area>Best framework choice (React vs Vue vs Svelte)</area>
                    <reasoning>Models provided conflicting recommendations based on different criteria</reasoning>
                    <potential_resolution>Context-dependent: React for enterprise, Vue for simplicity, Svelte for performance</potential_resolution>
                </uncertainty>
            </areas_of_uncertainty>{hitl_example}
        </example>
        {hitl_no_questions_example}
    """)

    # Use combined schema when HITL is enabled
    output_schema = ConsensusSynthesisWithHITL if hitl else ConsensusSynthesis

    agent = create_agent_with_instructions(
        description="Consensus Synthesizer",
        instructions=instructions_text,
        expected_output=expected_output_text,
        model=synthesis_model.model,
        debug_mode=debug_mode,
        output_schema=output_schema,
    )

    synthesis_input = dedent(f"""
    <synthesis_context>
        <original_task>{user_input}</original_task>
        <model_outputs>
{outputs_xml}
        </model_outputs>
        <critiques>
{critiques_xml}
        </critiques>
    </synthesis_context>
    """)

    response = await agent.arun(synthesis_input)

    if hitl:
        synthesis = cast(ConsensusSynthesisWithHITL, response.content)
    else:
        synthesis = cast(ConsensusSynthesis, response.content)

    synthesis.model_contribution_weights = normalized_weights

    assert synthesis is not None and isinstance(
        synthesis, (ConsensusSynthesis, ConsensusSynthesisWithHITL)
    ), "Synthesis parsing failed"
    return synthesis


async def _run_round_4_judge_and_refine(
    models: list[ModelConfig],
    synthesis: ConsensusSynthesis,
    user_input: str,
    judge_criteria: list[JudgeCriteria],
    max_iterations: int,
    judge_threshold: float,
    debug_mode: DebugMode,
    parent_agent: Agent | Team | None = None,
) -> tuple[str, float, int, list[JudgeResult]]:
    log_debug("Round 4: Judge & Refine phase")

    main_agent_context = format_main_agent_context(parent_agent) if parent_agent else ""

    judge_model = max(models, key=lambda m: m.importance)
    current_output = synthesis.synthesized_output
    judge_results: list[JudgeResult] = []
    overall_score: float = 0.5

    for iteration in range(1, max_iterations + 1):
        log_debug(f"Judge iteration {iteration}/{max_iterations}")

        verdicts: list[JudgeVerdict] = []
        total_weighted_score: float = 0.0
        total_weight: float = 0.0

        for criterion in judge_criteria:
            eval_instance = AgentAsJudgeEval(
                name=criterion.name,
                criteria=criterion.description,
                scoring_strategy="numeric",
                threshold=int(criterion.threshold * 10),
                model=judge_model.model,
            )

            result: AgentAsJudgeResult | None = eval_instance.run(
                input=user_input, output=current_output, print_results=False
            )

            assert result is not None and result.results, "AgentAsJudgeEval returned no results"

            first_result: AgentAsJudgeEvaluation = result.results[0]
            score = first_result.score / 10.0 if first_result.score is not None else 0.5
            reasoning = first_result.reason
            passed_criterion = first_result.passed

            verdicts.append(
                JudgeVerdict(
                    criterion_name=criterion.name,
                    score=score,
                    passed=passed_criterion,
                    reasoning=reasoning,
                    specific_issues=[]
                    if passed_criterion
                    else [
                        SpecificIssue(
                            issue=reasoning,
                            reasoning=f"Failed criterion: {criterion.name}",
                            importance=1.0 - score,
                        )
                    ],
                    improvement_suggestions=[]
                    if passed_criterion
                    else [
                        ImprovementSuggestion(
                            suggestion=f"Improve {criterion.name}",
                            reasoning=f"Score {score:.0%} is below threshold {criterion.threshold:.0%}",
                            importance=criterion.threshold - score,
                        )
                    ],
                )
            )
            total_weighted_score += score * criterion.weight
            total_weight += criterion.weight

        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.5
        passed = overall_score >= judge_threshold and all(v.passed for v in verdicts)

        judge_result = JudgeResult(
            iteration=iteration,
            verdicts=verdicts,
            overall_score=overall_score,
            passed=passed,
        )

        if passed:
            log_debug(f"Judge passed at iteration {iteration} with score {overall_score:.2f}")
            judge_results.append(judge_result)
            return current_output, overall_score, iteration, judge_results

        if iteration < max_iterations:
            log_debug(f"Refining output (iteration {iteration})")

            failed_criteria = [v for v in verdicts if not v.passed]
            issues_xml = "\n".join(
                f'            <failed_criterion name="{v.criterion_name}" score="{(f"{v.score:.2f}" if v.score is not None else "N/A")}">\n'
                f"                <reasoning>{v.reasoning}</reasoning>\n"
                f"            </failed_criterion>"
                for v in failed_criteria
            )

            instructions_text = dedent(f"""
                {main_agent_context}

                <refinement_instructions>
                    <task>Refine consensus output to address failed quality criteria.</task>
                    <failed_criteria>
{issues_xml}
                    </failed_criteria>
                    <requirements>
                        <requirement>DO NOT use markdown formatting - use plain text with proper spacing and newlines</requirement>
                        <requirement>Maintain all key insights from original output</requirement>
                        <requirement>Focus specifically on fixing the identified issues</requirement>
                        <requirement>Preserve overall structure and flow</requirement>
                        <requirement>Be honest about which issues you can and cannot address</requirement>
                    </requirements>
                    <output_format>
                        You must return a structured output that includes:
                        1. refined_output: The improved text
                        2. issues_addressed: List of issues you successfully fixed with explanation
                        3. issues_not_addressed: List of issues you could not fix with explanation
                        4. additional_improvements: Any other improvements you made
                    </output_format>
                </refinement_instructions>
            """)

            expected_output_text = dedent("""
                <description>
                    Return a structured refinement result that includes:
                    - The refined output text (improved version)
                    - For each issue that WAS addressed: explain how and what changes were made
                    - For each issue that was NOT addressed: explain why (e.g., insufficient info, trade-offs)
                    - Any other improvements made beyond the specific criteria

                    Be honest about limitations. Not all issues can be addressed in refinement.
                </description>
                <example>
                    <failed_criteria>
                        <criterion name="Completeness" score="0.5" reasoning="Lacks specific examples"/>
                        <criterion name="Accuracy" score="0.4" reasoning="Contains outdated information"/>
                    </failed_criteria>
                    <refinement_result>
                        <refined_output>Python is ideal for beginners due to its readable syntax that
                        resembles English, extensive documentation and tutorials, active community
                        support, and versatile applications from web development to data science.</refined_output>
                        <issues_addressed>
                            <issue>
                                <criterion_name>Completeness</criterion_name>
                                <original_issue>Lacks specific examples</original_issue>
                                <how_fixed>Added specific examples like documentation, community support, and applications</how_fixed>
                                <changes_made>Expanded from 6 words to 30+ words with concrete examples</changes_made>
                            </issue>
                        </issues_addressed>
                        <issues_not_addressed>
                            <issue>
                                <criterion_name>Accuracy</criterion_name>
                                <original_issue>Contains outdated information</original_issue>
                                <why_not_addressed>Cannot verify current information without external sources</why_not_addressed>
                                <suggested_alternative>Consult official Python documentation for latest version details</suggested_alternative>
                            </issue>
                        </issues_not_addressed>
                        <additional_improvements>
                            <improvement>Improved sentence flow and readability</improvement>
                        </additional_improvements>
                    </refinement_result>
                </example>
            """)

            refine_agent = create_agent_with_instructions(
                description="Consensus Refiner",
                instructions=instructions_text,
                expected_output=expected_output_text,
                model=judge_model.model,
                debug_mode=debug_mode,
                output_schema=RefinementResult,
            )

            refine_input = dedent(f"""
            <refinement_context>
                <original_output>
                    {current_output}
                </original_output>
            </refinement_context>
            """)
            refine_response = await refine_agent.arun(refine_input)
            refinement_result = cast(RefinementResult, refine_response.content)

            assert refinement_result is not None and isinstance(
                refinement_result, RefinementResult
            ), "RefinementResult parsing failed"
            current_output = refinement_result.refined_output
            judge_result.refined_output = refinement_result.refined_output

            # Build refinement_actions from the structured result
            refinement_actions: list[RefinementAction] = []

            # Add addressed issues
            for addressed in refinement_result.issues_addressed:
                # Find the original verdict to get importance
                original_verdict = next(
                    (v for v in verdicts if v.criterion_name == addressed.criterion_name), None
                )
                importance = (
                    float(original_verdict.specific_issues[0].importance)
                    if original_verdict
                    and original_verdict.specific_issues
                    and original_verdict.specific_issues[0].importance is not None
                    else 0.5
                )

                refinement_actions.append(
                    RefinementAction(
                        criterion_name=addressed.criterion_name,
                        issue_description=addressed.original_issue,
                        was_addressed=True,
                        importance=importance,
                        reasoning=addressed.how_fixed,
                        changes_made=addressed.changes_made,
                    )
                )

            # Add unaddressed issues
            for not_addressed in refinement_result.issues_not_addressed:
                # Find the original verdict to get importance
                original_verdict = next(
                    (v for v in verdicts if v.criterion_name == not_addressed.criterion_name), None
                )
                importance = (
                    original_verdict.specific_issues[0].importance
                    if original_verdict and original_verdict.specific_issues
                    else 0.5
                )

                refinement_actions.append(
                    RefinementAction(
                        criterion_name=not_addressed.criterion_name,
                        issue_description=not_addressed.original_issue,
                        was_addressed=False,
                        importance=importance or 0.5,
                        reasoning=not_addressed.why_not_addressed,
                        changes_made=None,
                    )
                )

            judge_result.refinement_actions = refinement_actions

        judge_results.append(judge_result)

    log_debug(f"Max iterations reached, final score: {overall_score:.2f}")
    return current_output, overall_score, max_iterations, judge_results


def _assemble_consensus_result(
    user_input: str,
    final_output: str,
    judge_score: float,
    refinement_iterations: int,
    generation_outputs: list[tuple[ModelConfig, GenerationOutput]],
    critiques: list[tuple[ModelConfig, str, CritiqueFeedback]],
    synthesis: ConsensusSynthesis,
    judge_results: list[JudgeResult],
    judge_criteria: list[JudgeCriteria] | None = None,
) -> ConsensusResult:
    model_contributions: list[ModelContribution] = []
    raw_generation_outputs: list[GenerationOutputSummary] = []

    for config, gen_output in generation_outputs:
        incorporated = len(
            [i for i in synthesis.incorporated_insights if i.from_model == config.name]
        )
        rejected = len([r for r in synthesis.rejected_approaches if r.from_model == config.name])
        # Convert KeyInsight to Contribution for model contributions
        contributions = [
            Contribution(
                contribution=ki.insight,
                reasoning=ki.reasoning,
                importance=ki.importance,
                uniqueness=ki.evidence,
            )
            for ki in gen_output.key_insights[:3]
        ]
        model_contributions.append(
            ModelContribution(
                model_name=config.name,
                importance=config.importance,
                perspective=config.perspective,
                key_contributions=contributions,
                insights_incorporated=incorporated,
                insights_rejected=rejected,
            )
        )
        raw_generation_outputs.append(
            GenerationOutputSummary(
                model_name=config.name,
                importance=config.importance,
                perspective=config.perspective,
                output=gen_output.output,
                assumptions=gen_output.assumptions,
                considered_alternatives=gen_output.considered_alternatives,
                confidence_breakdown=gen_output.confidence_breakdown,
                key_insights=gen_output.key_insights,
            )
        )

    critique_matrix: list[CritiqueSummary] = []
    for reviewer, target_name, feedback in critiques:
        critique_matrix.append(
            CritiqueSummary(
                reviewer_name=reviewer.name,
                target_name=target_name,
                is_self_critique=feedback.is_self_critique,
                strengths=feedback.strengths,
                weaknesses=feedback.weaknesses,
                missing_considerations=feedback.missing_considerations,
                flawed_assumptions=feedback.flawed_assumptions,
                suggested_improvements=feedback.suggested_improvements,
                agreement_level=feedback.agreement_level,
                overall_quality_score=feedback.overall_quality_score,
            )
        )

    model_names = ", ".join(c.name for c, _ in generation_outputs)
    generation_summary = f"Generated {len(generation_outputs)} outputs. Models: {model_names}."

    self_critiques = sum(1 for _, _, c in critiques if c.is_self_critique)
    peer_critiques = len(critiques) - self_critiques
    avg_quality = (
        sum(c.overall_quality_score for _, _, c in critiques) / len(critiques) if critiques else 0
    )
    critique_summary = f"{len(critiques)} critiques ({self_critiques} self, {peer_critiques} peer). Avg quality: {avg_quality:.2f}."

    synthesis_summary = f"Approach: {synthesis.synthesis_approach[:80]}. Insights: {len(synthesis.incorporated_insights)}, Conflicts: {len(synthesis.conflict_resolutions)}."

    judge_summary = f"{refinement_iterations} iteration(s). Score: {judge_score:.2f}."

    return ConsensusResult(
        user_input=user_input,
        final_output=final_output,
        consensus_confidence=synthesis.consensus_confidence,
        judge_score=judge_score,
        refinement_iterations=refinement_iterations,
        model_contributions=model_contributions,
        judge_results=judge_results,
        key_agreements=synthesis.areas_of_strong_agreement,
        resolved_conflicts=synthesis.conflict_resolutions,
        remaining_uncertainties=synthesis.areas_of_uncertainty,
        generation_summary=generation_summary,
        critique_summary=critique_summary,
        synthesis_summary=synthesis_summary,
        judge_summary=judge_summary,
        raw_generation_outputs=raw_generation_outputs,
        critique_matrix=critique_matrix,
        incorporated_insights=synthesis.incorporated_insights,
        conflict_resolutions=synthesis.conflict_resolutions,
        rejected_approaches=synthesis.rejected_approaches,
        synthesis_approach=synthesis.synthesis_approach,
        judge_criteria_used=judge_criteria or [],
    )


def _inject_consensus_result(run_input: RunInput, consensus_result: ConsensusResult) -> None:
    contributions_xml = "\n".join(
        f'        <model name="{mc.model_name}" importance="{(f"{mc.importance:.0%}" if mc.importance is not None else "N/A")}" insights_incorporated="{mc.insights_incorporated}" insights_rejected="{mc.insights_rejected}">\n'
        f"            <perspective>{mc.perspective or 'General'}</perspective>\n"
        f"            <key_contributions>{', '.join(c.contribution for c in mc.key_contributions) if mc.key_contributions else 'None'}</key_contributions>\n"
        f"        </model>"
        for mc in consensus_result.model_contributions
    )

    agreements_xml = (
        "\n".join(
            f'        <agreement confidence="{(f"{a.confidence:.0%}" if a.confidence is not None else "N/A")}">\n'
            f"            <point>{a.point}</point>\n"
            f"            <reasoning>{a.reasoning}</reasoning>\n"
            f"        </agreement>"
            for a in consensus_result.key_agreements
        )
        if consensus_result.key_agreements
        else "        <agreement>None</agreement>"
    )

    conflicts_xml = (
        "\n".join(
            f'        <conflict topic="{cr.topic}">\n'
            f"            <resolution>{cr.resolution}</resolution>\n"
            f"            <rationale>{cr.resolution_rationale}</rationale>\n"
            f"        </conflict>"
            for cr in consensus_result.resolved_conflicts
        )
        if consensus_result.resolved_conflicts
        else "        <conflict>None</conflict>"
    )

    uncertainties_xml = (
        "\n".join(
            f'        <uncertainty importance="{(f"{u.importance:.0%}" if u.importance is not None else "N/A")}">\n'
            f"            <area>{u.area}</area>\n"
            f"            <reasoning>{u.reasoning}</reasoning>\n"
            f"        </uncertainty>"
            for u in consensus_result.remaining_uncertainties
        )
        if consensus_result.remaining_uncertainties
        else "        <uncertainty>None</uncertainty>"
    )

    judge_iterations_parts: list[str] = []
    for jr in consensus_result.judge_results:
        verdicts_xml = "\n".join(
            f'                <verdict criterion="{v.criterion_name}" score="{(f"{v.score:.2f}" if v.score is not None else "N/A")}" passed="{v.passed}">\n'
            f"                    <reasoning>{v.reasoning}</reasoning>\n"
            + (
                f"                    <issues>{', '.join(si.issue for si in v.specific_issues)}</issues>\n"
                if v.specific_issues
                else ""
            )
            + "                </verdict>"
            for v in jr.verdicts
        )
        refinements_xml = (
            "\n".join(
                f'                <action criterion="{ra.criterion_name}" was_addressed="{str(ra.was_addressed).lower()}" importance="{(f"{ra.importance:.2f}" if ra.importance is not None else "N/A")}">\n'
                f"                    <issue>{ra.issue_description}</issue>\n"
                + (
                    f"                    <reasoning>{ra.reasoning}</reasoning>\n"
                    if ra.reasoning
                    else ""
                )
                + (
                    f"                    <changes_made>{ra.changes_made}</changes_made>\n"
                    if ra.changes_made
                    else ""
                )
                + "                </action>"
                for ra in jr.refinement_actions
            )
            if jr.refinement_actions
            else "                <action>None</action>"
        )
        judge_iterations_parts.append(
            f'            <iteration number="{jr.iteration}" overall_score="{(f"{jr.overall_score:.2f}" if jr.overall_score is not None else "N/A")}" passed="{jr.passed}">\n'
            f"                <verdicts>\n{verdicts_xml}\n                </verdicts>\n"
            f"                <refinement_actions>\n{refinements_xml}\n                </refinement_actions>\n"
            + (
                f"                <refined_output>{html.escape(jr.refined_output)}</refined_output>\n"
                if jr.refined_output
                else ""
            )
            + "            </iteration>"
        )
    judge_results_xml = (
        "\n".join(judge_iterations_parts)
        if judge_iterations_parts
        else "            <iteration>None</iteration>"
    )

    formatted_result = dedent(f"""
    <multi_model_consensus confidence="{consensus_result.consensus_confidence:.2%}" judge_score="{consensus_result.judge_score:.2%}" iterations="{consensus_result.refinement_iterations}">
        <consensus_explanation>
            This is a MULTI-MODEL CONSENSUS result. Before you received this task, multiple AI models
            with different perspectives independently analyzed it. Their outputs were:
            1. GENERATED independently by each model
            2. CRITIQUED by all models (self-critique + peer critique)
            3. SYNTHESIZED into a unified answer, weighting each model's importance
            4. JUDGED and refined until quality thresholds were met

            HOW TO USE THIS:
            - The final_output represents the AGREED-UPON answer from all models
            - Key agreements show where all models strongly aligned
            - Resolved conflicts show disagreements that were worked through
            - Remaining uncertainties are areas where models couldn't fully agree
            - Use this consensus as authoritative pre-analyzed context for your response
            - Build upon the consensus rather than contradicting it unless you have strong reason
        </consensus_explanation>
        <final_output>
{consensus_result.final_output}
        </final_output>
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
        <process_summary>
            <generation>{consensus_result.generation_summary}</generation>
            <critique>{consensus_result.critique_summary}</critique>
            <synthesis>{consensus_result.synthesis_summary}</synthesis>
            <judge>{consensus_result.judge_summary}</judge>
        </process_summary>
    </multi_model_consensus>
    """)

    inject_context_to_run_input(
        run_input=run_input, context_content=formatted_result, message_role="system", prepend=True
    )
    log_debug("Consensus result injected into run input")


def _create_consensus_hook(
    models: list[ModelConfig],
    judge_criteria: list[JudgeCriteria],
    triage_model: Model,
    skip_triage: bool = False,
    max_refinement_iterations: int = 3,
    judge_threshold: float = 0.7,
    async_hooks: bool = True,
    auto_save_html: bool = False,
    output_directory: Path | None = None,
    concurrent_processor: ConcurrentProcessor[Any, Any] | None = None,
    hitl: bool = False,
    hitl_max_questions: int = 5,
) -> AgnoPreHook:
    async def hook_logic(
        agent: Agent | Team,
        run_input: RunInput,
        session: AgentSession | TeamSession,
        user_id: UserId,
        debug_mode: DebugMode,
    ) -> None:
        user_input = run_input.input_content_string()

        # Triage: Quick check if consensus is needed
        if not skip_triage:
            triage_decision = await _run_triage_check(
                triage_model=triage_model,
                user_input=user_input,
                debug_mode=debug_mode,
                parent_agent=agent,
            )

            if not triage_decision.requires_consensus:
                log_debug(
                    f"Triage: Skipping consensus - {triage_decision.category}: "
                    f"{triage_decision.reason}"
                )
                # Inject a note that consensus was skipped
                skip_note = dedent(f"""
                <consensus_skipped>
                    <reason>{triage_decision.reason}</reason>
                    <category>{triage_decision.category}</category>
                    <note>This request does not require multi-model consensus research.
                    Respond directly without the overhead of full consensus analysis.</note>
                </consensus_skipped>
                """)
                inject_context_to_run_input(
                    run_input=run_input,
                    context_content=skip_note,
                    message_role="system",
                    prepend=True,
                )
                return

        log_debug(f"Starting consensus process with {len(models)} models")

        processor = concurrent_processor or ConcurrentProcessor(concurrency=5)

        generation_outputs = await _run_round_1_generation(
            models, user_input, debug_mode, processor, agent
        )

        if not generation_outputs:
            return

        critiques = await _run_round_2_critique(
            models,
            generation_outputs,
            debug_mode,
            processor,
            agent,
        )

        # Run synthesis - includes HITL questionnaire generation in single call if hitl=True
        synthesis = await _run_round_3_synthesis(
            models,
            generation_outputs,
            critiques,
            user_input,
            debug_mode,
            agent,
            hitl=hitl,
            hitl_max_questions=hitl_max_questions,
        )

        # HITL mode: check if questionnaire was generated and inject toolkit
        if hitl and isinstance(synthesis, ConsensusSynthesisWithHITL):
            questionnaire = synthesis.hitl_questionnaire

            # Skip HITL if no meaningful questions to ask
            if (
                questionnaire is None
                or questionnaire.should_skip_hitl
                or not questionnaire.questions
            ):
                log_debug(
                    "HITL skipped: no uncertainties or meaningful questions to ask. "
                    "Proceeding with standard Judge & Refine."
                )
                # Fall through to standard mode below
            else:
                log_debug(
                    f"HITL mode: injecting toolkit with {len(questionnaire.questions)} questions"
                )

                # Build refinement kwargs for the toolkit (passed as function args)
                refinement_kwargs: dict[str, Any] = {
                    "models": models,
                    "synthesis": synthesis,
                    "user_input": user_input,
                    "judge_criteria": judge_criteria,
                    "max_iterations": max_refinement_iterations,
                    "judge_threshold": judge_threshold,
                    "debug_mode": debug_mode,
                    "parent_agent": agent,
                }

                # Create and add the HITL toolkit to the agent
                # Pass refinement function to avoid cyclic import
                hitl_toolkit = ConsensusHITLToolkit(
                    questionnaire=questionnaire,
                    synthesis=synthesis,
                    refinement_func=_run_round_4_judge_and_refine,
                    refinement_kwargs=refinement_kwargs,
                )

                # Dynamically add the toolkit to the agent
                if isinstance(agent, Agent):
                    agent.add_tool(hitl_toolkit)
                    log_debug(
                        f"Added ConsensusHITLToolkit with {len(questionnaire.questions)} questions"
                    )

                # Inject consensus result without running Judge & Refine
                # The toolkit's run_refinement_with_feedback() will handle that
                consensus_result = _assemble_consensus_result(
                    user_input,
                    synthesis.synthesized_output,  # Use synthesis output directly
                    judge_score=0.0,
                    refinement_iterations=0,
                    generation_outputs=generation_outputs,
                    critiques=critiques,
                    synthesis=synthesis,
                    judge_results=[],
                    judge_criteria=judge_criteria,
                )

                log_debug(
                    f"HITL Consensus pre-synthesis complete: confidence={consensus_result.consensus_confidence:.2%}"
                )
                _inject_consensus_result(run_input, consensus_result)
                return

        # Standard mode: run Judge & Refine phase
        (
            final_output,
            judge_score,
            iterations,
            judge_results,
        ) = await _run_round_4_judge_and_refine(
            models,
            synthesis,
            user_input,
            judge_criteria,
            max_refinement_iterations,
            judge_threshold,
            debug_mode,
            agent,
        )

        consensus_result = _assemble_consensus_result(
            user_input,
            final_output,
            judge_score,
            iterations,
            generation_outputs,
            critiques,
            synthesis,
            judge_results,
            judge_criteria,
        )

        log_debug(
            f"Consensus complete: confidence={consensus_result.consensus_confidence:.2%}, judge_score={consensus_result.judge_score:.2%}"
        )
        _inject_consensus_result(run_input, consensus_result)

        if auto_save_html and output_directory:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consensus_report_{timestamp}.html"
            output_path = output_directory / filename
            consensus_result.export_report_to_html(output_path)
            log_debug(f"Consensus HTML report saved to {output_path}")

    if async_hooks:

        def sync_hook(
            agent: Agent | Team,
            run_input: RunInput,
            session: AgentSession | TeamSession,
            user_id: UserId,
            debug_mode: DebugMode,
        ) -> None:
            asyncio.run(hook_logic(agent, run_input, session, user_id, debug_mode))

        return sync_hook
    else:
        return hook_logic
