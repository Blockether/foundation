"""Consensus hooks for Agno agents - Multi-model agreement through structured rounds."""

from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ....concurrency import ConcurrentProcessor as ConcurrentProcessorType

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
from agno.utils.log import log_debug, log_warning  # type: ignore
from pydantic import Field, field_validator

from ....concurrency import ConcurrentProcessor
from ....models import BaseModelSerializable, ChainOfThoughts
from ....utils import (
    AgnoPreHook,
    DebugMode,
    UserId,
    create_agent_with_instructions,
    inject_context_to_run_input,
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


class Assumption(BaseModelSerializable):
    assumption: str = Field(description="The assumption that was made.")
    criticality: float = Field(
        ge=0.0, le=1.0, description="How critical this assumption is - impact if wrong (0.0-1.0)."
    )


class ConfidenceBreakdown(BaseModelSerializable):
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
    assumptions: list[Assumption] = Field(
        default=[], description="Assumptions made while generating."
    )
    considered_alternatives: list[ConsideredAlternative] = Field(
        default=[], description="Alternative approaches considered."
    )
    confidence_breakdown: ConfidenceBreakdown = Field(...)
    key_insights: list[str] = Field(default=[], description="Main insights this model brings.")


class StrengthWeakness(BaseModelSerializable):
    description: str = Field(description="Description of the strength or weakness.")
    severity_or_value: float = Field(
        ge=0.0, le=1.0, description="Severity or value score (0.0-1.0)."
    )
    evidence: str = Field(description="Quote or reference from the output.")


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
    missing_considerations: list[str] = Field(default=[], description="Important aspects missed.")
    flawed_assumptions: list[str] = Field(default=[], description="Incorrect assumptions.")
    suggested_improvements: list[SuggestedImprovement] = Field(
        default=[], description="Improvements."
    )
    alternative_approaches: list[ConsideredAlternative] = Field(
        default=[], description="Better approaches."
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
    areas_of_strong_agreement: list[str] = Field(
        default=[], description="Areas of strong agreement."
    )
    areas_of_uncertainty: list[str] = Field(default=[], description="Areas that remain uncertain.")


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
    specific_issues: list[str] = Field(default=[], description="Specific issues identified.")
    improvement_suggestions: list[str] = Field(default=[], description="Improvement suggestions.")


class RefinementAction(BaseModelSerializable):
    issue_addressed: str = Field(description="The issue addressed.")
    action_taken: str = Field(description="Action taken to address it.")
    models_consulted: list[str] = Field(default=[], description="Models re-consulted.")


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
    key_contributions: list[str] = Field(default=[], description="Key contributions.")
    insights_incorporated: int = Field(default=0, description="Insights incorporated count.")
    insights_rejected: int = Field(default=0, description="Insights rejected count.")


class ConsensusResult(BaseModelSerializable):
    final_output: str = Field(description="The final consensus output.")
    consensus_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in consensus.")
    judge_score: float = Field(ge=0.0, le=1.0, description="Final judge score.")
    refinement_iterations: int = Field(description="Number of refinement iterations.")
    model_contributions: list[ModelContribution] = Field(description="Model contributions summary.")
    judge_results: list[JudgeResult] = Field(
        default=[],
        description="Detailed judge results per iteration with verdicts and refinements.",
    )
    key_agreements: list[str] = Field(default=[], description="Key agreements.")
    resolved_conflicts: list[str] = Field(default=[], description="Resolved conflicts.")
    remaining_uncertainties: list[str] = Field(default=[], description="Remaining uncertainties.")
    generation_summary: str = Field(description="Generation phase summary.")
    critique_summary: str = Field(description="Critique phase summary.")
    synthesis_summary: str = Field(description="Synthesis phase summary.")
    judge_summary: str = Field(description="Judge phase summary.")

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

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("name cannot be empty")
        return v


class ConsensusHooksConfig:
    """Configuration for consensus hooks with multi-model agreement.

    The consensus process runs 4 rounds:
    1. Generation: Each model generates a response independently
    2. Critique: Models critique each other's outputs (self + peer)
    3. Synthesis: Combine outputs into a weighted consensus
    4. Judge & Refine: Evaluate and iteratively refine until quality threshold met

    Args:
        models: List of ModelConfig with model instances, names, importance weights
        judge_criteria: List of JudgeCriteria for final evaluation
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
        max_refinement_iterations: int = 3,
        judge_threshold: float = 0.7,
        async_hooks: bool = True,
        auto_save_html: bool = False,
        output_directory: str | Path | None = None,
        concurrent_processor: "ConcurrentProcessor[Any, Any] | None" = None,
    ):
        if not models:
            raise ValueError("At least one model must be provided")
        if not judge_criteria:
            raise ValueError("At least one judge criterion must be provided")
        if auto_save_html and output_directory is None:
            raise ValueError("output_directory must be provided when auto_save_html is True")

        self.models = models
        self.judge_criteria = judge_criteria
        self.max_refinement_iterations = max_refinement_iterations
        self.judge_threshold = judge_threshold
        self.async_hooks = async_hooks
        self.auto_save_html = auto_save_html
        self.output_directory = Path(output_directory) if output_directory else None
        self.concurrent_processor = concurrent_processor or ConcurrentProcessor(
            concurrency=self.DEFAULT_CONCURRENCY
        )

    def pre_hook(self) -> AgnoPreHook:
        return _create_consensus_hook(
            models=self.models,
            judge_criteria=self.judge_criteria,
            max_refinement_iterations=self.max_refinement_iterations,
            judge_threshold=self.judge_threshold,
            async_hooks=not self.async_hooks,
            auto_save_html=self.auto_save_html,
            output_directory=self.output_directory,
            concurrent_processor=self.concurrent_processor,
        )


async def _run_round_1_generation(
    models: list[ModelConfig],
    user_input: str,
    debug_mode: DebugMode,
    processor: "ConcurrentProcessor[Any, Any]",
) -> list[tuple[ModelConfig, GenerationOutput]]:
    log_debug(f"Round 1: Generation with {len(models)} models")

    async def generate_for_model(
        model_config: ModelConfig,
    ) -> tuple[ModelConfig, GenerationOutput]:
        perspective_xml = ""
        if model_config.perspective:
            perspective_xml = f"<model_perspective>{model_config.perspective}</model_perspective>"

        instructions_text = dedent(f"""
            <generation_instructions>
                <task>Generate a response to the given task with step-by-step reasoning.</task>
                {perspective_xml}
                <requirements>
                    <requirement>Consider multiple approaches before settling on one</requirement>
                    <requirement>Document your assumptions explicitly with criticality scores</requirement>
                    <requirement>Break down your confidence by different aspects (factual, completeness, coherence, relevance)</requirement>
                    <requirement>Identify key insights you bring to this task</requirement>
                </requirements>
            </generation_instructions>
        """)

        expected_output_text = dedent("""
            <expected_output>
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
                        <insight>Learning curve matters more than language power for beginners</insight>
                        <insight>Community support accelerates learning</insight>
                    </key_insights>
                </example>
            </expected_output>
        """)

        agent = create_agent_with_instructions(
            description=f"Model {model_config.name}",
            instructions=instructions_text,
            expected_output=expected_output_text,
            model=model_config.model,
            debug_mode=debug_mode,
            output_schema=GenerationOutput,
        )

        response = await agent.arun(user_input)
        output = cast(GenerationOutput, response.content)
        return (model_config, output)

    results = await processor.process(models, generate_for_model)
    log_debug(f"Round 1 complete: {len(results)} outputs generated")
    return list(results)


CritiqueTask = tuple[ModelConfig, ModelConfig, GenerationOutput, bool]


async def _run_round_2_critique(
    models: list[ModelConfig],
    generation_outputs: list[tuple[ModelConfig, GenerationOutput]],
    debug_mode: DebugMode,
    processor: "ConcurrentProcessor[Any, Any]",
) -> list[tuple[ModelConfig, str, CritiqueFeedback]]:
    log_debug("Round 2: Self + Peer Critique phase")

    async def generate_critique(
        task: CritiqueTask,
    ) -> tuple[ModelConfig, str, CritiqueFeedback]:
        reviewer, target_config, target_output, is_self = task
        critique_type = "self-critique" if is_self else "peer critique"
        final_instruction = (
            "Be honest about your own shortcomings." if is_self else "Be constructive but thorough."
        )
        perspective_text = reviewer.perspective or "General expert"

        instructions_text = dedent(f"""
            <critique_instructions type="{critique_type}">
                <reviewer_context>
                    <perspective>{perspective_text}</perspective>
                    <importance_weight>{reviewer.importance}</importance_weight>
                </reviewer_context>
                <analysis_requirements>
                    <requirement>Identify specific strengths with evidence from the output</requirement>
                    <requirement>Identify specific weaknesses with evidence from the output</requirement>
                    <requirement>Note any missing considerations or blind spots</requirement>
                    <requirement>Evaluate the validity of stated assumptions</requirement>
                    <requirement>Suggest concrete, actionable improvements</requirement>
                    <requirement>Consider alternative approaches that might be better</requirement>
                </analysis_requirements>
                <guidance>{final_instruction}</guidance>
            </critique_instructions>
        """)

        expected_output_text = dedent(f"""
            <expected_output>
                <example>
                    <is_self_critique>{str(is_self).lower()}</is_self_critique>
                    <target_model>{target_config.name}</target_model>
                    <strengths>
                        <strength severity_or_value="0.8">
                            <description>Clear, well-structured response with logical flow</description>
                            <evidence>"The response starts with the main point and provides supporting details"</evidence>
                        </strength>
                    </strengths>
                    <weaknesses>
                        <weakness severity_or_value="0.6">
                            <description>Missing consideration of edge cases</description>
                            <evidence>"The response only covers the happy path scenario"</evidence>
                        </weakness>
                    </weaknesses>
                    <missing_considerations>
                        <item>Performance implications not addressed</item>
                        <item>Security considerations overlooked</item>
                    </missing_considerations>
                    <suggested_improvements>
                        <improvement expected_impact="0.7">
                            <what_to_change>Add error handling section</what_to_change>
                            <why>Improves robustness and production readiness</why>
                            <proposed_fix>Include try-catch blocks and validation examples</proposed_fix>
                        </improvement>
                    </suggested_improvements>
                    <agreement_level>0.75</agreement_level>
                    <overall_quality_score>0.70</overall_quality_score>
                </example>
            </expected_output>
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
                f'            <assumption criticality="{a.criticality:.2f}">{a.assumption}</assumption>'
                for a in target_output.assumptions
            )
            if target_output.assumptions
            else "            <assumption>None documented</assumption>"
        )

        alternatives_xml = (
            "\n".join(
                f'            <alternative potential_value="{a.potential_value:.2f}">\n'
                f"                <approach>{a.approach}</approach>\n"
                f"                <why_not_chosen>{a.why_not_chosen or 'This was chosen'}</why_not_chosen>\n"
                f"            </alternative>"
                for a in target_output.considered_alternatives
            )
            if target_output.considered_alternatives
            else "            <alternative>None documented</alternative>"
        )

        insights_xml = (
            "\n".join(f"            <insight>{i}</insight>" for i in target_output.key_insights)
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
        return (reviewer, target_config.name, feedback)

    critique_tasks: list[CritiqueTask] = []
    for reviewer in models:
        for target_config, target_output in generation_outputs:
            is_self = reviewer.name == target_config.name
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
) -> ConsensusSynthesis:
    log_debug("Round 3: Weighted Synthesis phase")

    synthesis_model = max(models, key=lambda m: m.importance)

    output_summaries: list[str] = []
    for config, output in generation_outputs:
        perspective = config.perspective or "Not specified"
        insights_xml = (
            "\n".join(f"                <insight>{i}</insight>" for i in output.key_insights)
            if output.key_insights
            else "                <insight>None</insight>"
        )
        truncated_output = output.output[:1500]
        summary = dedent(f"""
        <model_output name="{config.name}" importance="{config.importance}" perspective="{perspective}">
            <output>{truncated_output}...</output>
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
            critique_items.append(
                f'            <critique type="{ctype}" reviewer="{reviewer_name}" '
                f'quality_score="{fb.overall_quality_score:.2f}" agreement="{fb.agreement_level:.2f}" />'
            )
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

    instructions_text = dedent(f"""
        <synthesis_instructions>
            <task>Synthesize multiple model outputs into a final consensus.</task>
            <model_weights>
{weights_xml}
            </model_weights>
            <requirements>
                <requirement>Analyze all outputs and critiques thoroughly</requirement>
                <requirement>Weight contributions by model importance</requirement>
                <requirement>Resolve conflicts considering importance, critique feedback, and self-critique admissions</requirement>
                <requirement>Incorporate best insights from all models</requirement>
                <requirement>Document rejected approaches with reasons</requirement>
                <requirement>Identify areas of strong agreement vs remaining uncertainty</requirement>
            </requirements>
            <goal>Create a synthesis that is BETTER than any individual output.</goal>
        </synthesis_instructions>
    """)

    expected_output_text = dedent("""
        <expected_output>
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
                    <area>JavaScript ecosystem maturity</area>
                    <area>Full-stack capability importance</area>
                </areas_of_strong_agreement>
                <areas_of_uncertainty>
                    <area>Best framework choice (React vs Vue vs Svelte)</area>
                </areas_of_uncertainty>
            </example>
        </expected_output>
    """)

    agent = create_agent_with_instructions(
        description="Consensus Synthesizer",
        instructions=instructions_text,
        expected_output=expected_output_text,
        model=synthesis_model.model,
        debug_mode=debug_mode,
        output_schema=ConsensusSynthesis,
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
    synthesis = cast(ConsensusSynthesis, response.content)

    synthesis.model_contribution_weights = normalized_weights
    return synthesis


async def _run_round_4_judge_and_refine(
    models: list[ModelConfig],
    synthesis: ConsensusSynthesis,
    user_input: str,
    judge_criteria: list[JudgeCriteria],
    max_iterations: int,
    judge_threshold: float,
    debug_mode: DebugMode,
) -> tuple[str, float, int, list[JudgeResult]]:
    log_debug("Round 4: Judge & Refine phase")

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
                    specific_issues=[] if passed_criterion else [reasoning],
                    improvement_suggestions=[]
                    if passed_criterion
                    else [f"Improve {criterion.name}"],
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
            issues = [f"{v.criterion_name}: {v.reasoning}" for v in failed_criteria]
            issues_xml = "\n".join(
                f'            <failed_criterion name="{v.criterion_name}" score="{v.score:.2f}">\n'
                f"                <reasoning>{v.reasoning}</reasoning>\n"
                f"            </failed_criterion>"
                for v in failed_criteria
            )

            instructions_text = dedent(f"""
                <refinement_instructions>
                    <task>Refine the consensus output to address failed quality criteria.</task>
                    <failed_criteria>
{issues_xml}
                    </failed_criteria>
                    <requirements>
                        <requirement>Maintain all key insights from the original output</requirement>
                        <requirement>Focus specifically on fixing the identified issues</requirement>
                        <requirement>Preserve the overall structure and flow</requirement>
                    </requirements>
                </refinement_instructions>
            """)

            expected_output_text = dedent("""
                <expected_output>
                    <description>
                        Return the refined output as plain text. The refined version should:
                        - Address each failed criterion specifically
                        - Keep the same structure and key points from the original
                        - Improve clarity, accuracy, or completeness as needed
                    </description>
                    <example>
                        <original>Python is good for beginners.</original>
                        <failed_criterion>Completeness - lacks supporting details</failed_criterion>
                        <refined>Python is ideal for beginners due to its readable syntax that
                        resembles English, extensive documentation and tutorials, active community
                        support, and versatile applications from web development to data science.</refined>
                    </example>
                </expected_output>
            """)

            refine_agent = create_agent_with_instructions(
                description="Consensus Refiner",
                instructions=instructions_text,
                expected_output=expected_output_text,
                model=judge_model.model,
                debug_mode=debug_mode,
            )

            refine_input = dedent(f"""
            <refinement_context>
                <original_output>
                    {current_output}
                </original_output>
            </refinement_context>
            """)
            refine_response = await refine_agent.arun(refine_input)
            refined = cast(str, refine_response.content)
            current_output = refined
            judge_result.refined_output = refined
            judge_result.refinement_actions = [
                RefinementAction(issue_addressed=i, action_taken="Refined") for i in issues
            ]

        judge_results.append(judge_result)

    log_debug(f"Max iterations reached, final score: {overall_score:.2f}")
    return current_output, overall_score, max_iterations, judge_results


def _assemble_consensus_result(
    final_output: str,
    judge_score: float,
    refinement_iterations: int,
    generation_outputs: list[tuple[ModelConfig, GenerationOutput]],
    critiques: list[tuple[ModelConfig, str, CritiqueFeedback]],
    synthesis: ConsensusSynthesis,
    judge_results: list[JudgeResult],
) -> ConsensusResult:
    model_contributions: list[ModelContribution] = []
    for config, gen_output in generation_outputs:
        incorporated = len(
            [i for i in synthesis.incorporated_insights if i.from_model == config.name]
        )
        rejected = len([r for r in synthesis.rejected_approaches if r.from_model == config.name])
        model_contributions.append(
            ModelContribution(
                model_name=config.name,
                importance=config.importance,
                perspective=config.perspective,
                key_contributions=gen_output.key_insights[:3],
                insights_incorporated=incorporated,
                insights_rejected=rejected,
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
        final_output=final_output,
        consensus_confidence=synthesis.consensus_confidence,
        judge_score=judge_score,
        refinement_iterations=refinement_iterations,
        model_contributions=model_contributions,
        judge_results=judge_results,
        key_agreements=synthesis.areas_of_strong_agreement,
        resolved_conflicts=[cr.topic for cr in synthesis.conflict_resolutions],
        remaining_uncertainties=synthesis.areas_of_uncertainty,
        generation_summary=generation_summary,
        critique_summary=critique_summary,
        synthesis_summary=synthesis_summary,
        judge_summary=judge_summary,
    )


def _inject_consensus_result(run_input: RunInput, consensus_result: ConsensusResult) -> None:
    contributions_xml = "\n".join(
        f'        <model name="{mc.model_name}" importance="{mc.importance:.0%}" insights_incorporated="{mc.insights_incorporated}" insights_rejected="{mc.insights_rejected}">\n'
        f"            <perspective>{mc.perspective or 'General'}</perspective>\n"
        f"            <key_contributions>{', '.join(mc.key_contributions) if mc.key_contributions else 'None'}</key_contributions>\n"
        f"        </model>"
        for mc in consensus_result.model_contributions
    )

    agreements_xml = (
        "\n".join(f"        <agreement>{a}</agreement>" for a in consensus_result.key_agreements)
        if consensus_result.key_agreements
        else "        <agreement>None</agreement>"
    )

    conflicts_xml = (
        "\n".join(f"        <conflict>{c}</conflict>" for c in consensus_result.resolved_conflicts)
        if consensus_result.resolved_conflicts
        else "        <conflict>None</conflict>"
    )

    uncertainties_xml = (
        "\n".join(
            f"        <uncertainty>{u}</uncertainty>"
            for u in consensus_result.remaining_uncertainties
        )
        if consensus_result.remaining_uncertainties
        else "        <uncertainty>None</uncertainty>"
    )

    judge_iterations_parts: list[str] = []
    for jr in consensus_result.judge_results:
        verdicts_xml = "\n".join(
            f'                <verdict criterion="{v.criterion_name}" score="{v.score:.2f}" passed="{v.passed}">\n'
            f"                    <reasoning>{v.reasoning}</reasoning>\n"
            + (
                f"                    <issues>{', '.join(v.specific_issues)}</issues>\n"
                if v.specific_issues
                else ""
            )
            + f"                </verdict>"
            for v in jr.verdicts
        )
        refinements_xml = (
            "\n".join(
                f'                <action issue="{ra.issue_addressed}">{ra.action_taken}</action>'
                for ra in jr.refinement_actions
            )
            if jr.refinement_actions
            else "                <action>None</action>"
        )
        judge_iterations_parts.append(
            f'            <iteration number="{jr.iteration}" overall_score="{jr.overall_score:.2f}" passed="{jr.passed}">\n'
            f"                <verdicts>\n{verdicts_xml}\n                </verdicts>\n"
            f"                <refinement_actions>\n{refinements_xml}\n                </refinement_actions>\n"
            + (
                f"                <refined_output>{jr.refined_output[:200]}...</refined_output>\n"
                if jr.refined_output
                else ""
            )
            + f"            </iteration>"
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
    max_refinement_iterations: int,
    judge_threshold: float,
    async_hooks: bool = True,
    auto_save_html: bool = False,
    output_directory: Path | None = None,
    concurrent_processor: "ConcurrentProcessor[Any, Any] | None" = None,
) -> AgnoPreHook:
    async def hook_logic(
        agent: Agent | Team,
        run_input: RunInput,
        session: AgentSession | TeamSession,
        user_id: UserId,
        debug_mode: DebugMode,
    ) -> None:
        user_input = run_input.input_content_string()
        log_debug(f"Starting consensus process with {len(models)} models")

        processor = concurrent_processor or ConcurrentProcessor(concurrency=5)

        generation_outputs = await _run_round_1_generation(
            models, user_input, debug_mode, processor
        )
        if not generation_outputs:
            return

        critiques = await _run_round_2_critique(models, generation_outputs, debug_mode, processor)

        synthesis = await _run_round_3_synthesis(
            models, generation_outputs, critiques, user_input, debug_mode
        )

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
        )

        consensus_result = _assemble_consensus_result(
            final_output,
            judge_score,
            iterations,
            generation_outputs,
            critiques,
            synthesis,
            judge_results,
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
