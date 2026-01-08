"""Consensus hooks for Agno agents - Multi-model agreement through structured rounds."""

from __future__ import annotations

import asyncio
import datetime
import html
from pathlib import Path
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
from agno.utils.log import log_debug  # type: ignore

from ....concurrency import ConcurrentProcessor
from ....utils import (
    AgnoPreHook,
    DebugMode,
    UserId,
    create_agent_with_instructions,
    format_main_agent_context,
    inject_context_to_run_input,
)
from . import prompts
from .hitl import ConsensusHITLToolkit
from .models import (
    ConsensusResult,
    ConsensusSynthesis,
    ConsensusSynthesisWithHITL,
    Contribution,
    CritiqueFeedback,
    CritiqueSummary,
    CritiqueTask,
    GenerationOutput,
    GenerationOutputSummary,
    HITLIteration,
    HITLQuestionnaire,
    ImprovementSuggestion,
    JudgeCriteria,
    JudgeResult,
    JudgeVerdict,
    ModelConfig,
    ModelContribution,
    RefinementAction,
    RefinementResult,
    SpecificIssue,
    TriageDecision,
)


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

    instructions_text = prompts.build_triage_instructions(main_agent_context)
    expected_output_text = prompts.TRIAGE_EXAMPLES

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
        instructions_text = prompts.build_generation_instructions(
            main_agent_context, model_config.perspective
        )
        expected_output_text = prompts.build_generation_example()

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

        instructions_text = prompts.build_critique_instructions(
            main_agent_context=main_agent_context,
            critique_type=critique_type,
            perspective_text=perspective_text,
            importance_weight=reviewer.importance,
            final_instruction=final_instruction,
        )
        expected_output_text = prompts.build_critique_example(is_self, target_config.name)

        agent = create_agent_with_instructions(
            description=f"Critic {reviewer.name} ({critique_type})",
            instructions=instructions_text,
            expected_output=expected_output_text,
            model=reviewer.model,
            debug_mode=debug_mode,
            output_schema=CritiqueFeedback,
        )

        critique_input = prompts.build_critique_target_xml(
            target_config_name=target_config.name,
            target_perspective=target_config.perspective or "Not specified",
            target_importance=target_config.importance,
            target_output=target_output.output,
            assumptions=target_output.assumptions,
            considered_alternatives=target_output.considered_alternatives,
            key_insights=target_output.key_insights,
            confidence_breakdown=target_output.confidence_breakdown,
            html_escape_func=html.escape,
        )

        response = await agent.arun(critique_input)
        feedback = cast(CritiqueFeedback, response.content)
        assert feedback is not None and isinstance(feedback, CritiqueFeedback), (
            "CritiqueFeedback parsing failed"
        )

        # Split checked_claims into accurate and inaccurate for clarity
        accurate_claims = [c for c in feedback.checked_claims if c.is_accurate]

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
) -> tuple[ConsensusSynthesis, list[HITLIteration]]:
    log_debug("Round 3: Weighted Synthesis phase" + (" with HITL" if hitl else ""))

    main_agent_context = format_main_agent_context(parent_agent) if parent_agent else ""
    synthesis_model = max(models, key=lambda m: m.importance)

    output_summaries: list[str] = []
    for config, output in generation_outputs:
        summary = prompts.build_model_output_xml(
            config_name=config.name,
            config_importance=config.importance,
            perspective=config.perspective or "Not specified",
            output=output.output,
            key_insights=output.key_insights,
            html_escape_func=html.escape,
        )
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
            critique_content = prompts.build_critique_feedback_xml(
                reviewer_name=reviewer_name,
                is_self_critique=fb.is_self_critique,
                feedback=fb,
                html_escape_func=html.escape,
            )
            critique_items.append(critique_content)

        critiques_parts.append(prompts.build_critiques_for_wrapper_xml(target_name, critique_items))
    critiques_xml = "\n".join(critiques_parts)

    model_weights = {config.name: config.importance for config in models}
    total_weight = sum(model_weights.values())
    normalized_weights = {name: w / total_weight for name, w in model_weights.items()}
    weights_xml = prompts.build_model_weights_xml(normalized_weights)

    synthesis = await _run_single_synthesis(
        synthesis_model=synthesis_model,
        main_agent_context=main_agent_context,
        weights_xml=weights_xml,
        user_input=user_input,
        outputs_xml=outputs_xml,
        critiques_xml=critiques_xml,
        debug_mode=debug_mode,
    )
    synthesis.model_contribution_weights = normalized_weights

    hitl_iterations: list[HITLIteration] = []
    if hitl:
        synthesis, hitl_iterations = await _run_synthesis_hitl_loop(
            models=models,
            synthesis=synthesis,
            synthesis_model=synthesis_model,
            main_agent_context=main_agent_context,
            weights_xml=weights_xml,
            user_input=user_input,
            outputs_xml=outputs_xml,
            critiques_xml=critiques_xml,
            debug_mode=debug_mode,
            hitl_max_questions=hitl_max_questions,
            normalized_weights=normalized_weights,
        )

    return synthesis, hitl_iterations


async def _run_single_synthesis(
    synthesis_model: ModelConfig,
    main_agent_context: str,
    weights_xml: str,
    user_input: str,
    outputs_xml: str,
    critiques_xml: str,
    debug_mode: DebugMode,
    hitl_feedback: str | None = None,
    hitl_toolkit: ConsensusHITLToolkit | None = None,
) -> ConsensusSynthesis:
    instructions_text = prompts.build_synthesis_instructions(
        main_agent_context=main_agent_context,
        weights_xml=weights_xml,
        hitl_feedback=hitl_feedback,
    )
    expected_output_text = prompts.build_synthesis_example()

    tools: list[Any] = []
    if hitl_toolkit is not None:
        tools.append(hitl_toolkit)

    agent = create_agent_with_instructions(
        description="Consensus Synthesizer",
        instructions=instructions_text,
        expected_output=expected_output_text,
        model=synthesis_model.model,
        debug_mode=debug_mode,
        output_schema=ConsensusSynthesis,
        tools=tools if tools else None,
    )

    synthesis_input = prompts.build_synthesis_context_xml(user_input, outputs_xml, critiques_xml)

    response = await agent.arun(synthesis_input)

    synthesis = cast(ConsensusSynthesis, response.content)
    assert synthesis is not None and isinstance(synthesis, ConsensusSynthesis), (
        "Synthesis parsing failed"
    )
    return synthesis


async def _run_synthesis_hitl_loop(
    models: list[ModelConfig],
    synthesis: ConsensusSynthesis,
    synthesis_model: ModelConfig,
    main_agent_context: str,
    weights_xml: str,
    user_input: str,
    outputs_xml: str,
    critiques_xml: str,
    debug_mode: DebugMode,
    hitl_max_questions: int,
    normalized_weights: dict[str, float],
    max_hitl_iterations: int = 3,
) -> tuple[ConsensusSynthesis, list[HITLIteration]]:
    log_debug("Starting HITL loop for synthesis refinement")

    hitl_toolkit = ConsensusHITLToolkit()
    current_synthesis = synthesis
    hitl_iterations: list[HITLIteration] = []

    for iteration in range(1, max_hitl_iterations + 1):
        log_debug(f"HITL iteration {iteration}/{max_hitl_iterations}")

        confidence_before = current_synthesis.consensus_confidence

        questionnaire = await _generate_hitl_questionnaire_from_synthesis(
            models=models,
            synthesis=current_synthesis,
            user_input=user_input,
            debug_mode=debug_mode,
            hitl_max_questions=hitl_max_questions,
        )

        if questionnaire.should_skip_hitl or not questionnaire.questions:
            log_debug("HITL: no meaningful questions, exiting loop")
            break

        log_debug(f"HITL: generated {len(questionnaire.questions)} questions")

        hitl_feedback = prompts.build_hitl_feedback_context_xml(questionnaire)

        previous_synthesis = current_synthesis
        current_synthesis = await _run_single_synthesis(
            synthesis_model=synthesis_model,
            main_agent_context=main_agent_context,
            weights_xml=weights_xml,
            user_input=user_input,
            outputs_xml=outputs_xml,
            critiques_xml=critiques_xml,
            debug_mode=debug_mode,
            hitl_feedback=hitl_feedback,
            hitl_toolkit=hitl_toolkit,
        )
        current_synthesis.model_contribution_weights = normalized_weights

        hitl_iteration = HITLIteration(
            iteration=iteration,
            questionnaire=questionnaire,
            user_answers=[],  # TODO: Capture actual user answers from toolkit
            synthesis_before_confidence=confidence_before,
            synthesis_after_confidence=current_synthesis.consensus_confidence,
            synthesis_output_after=current_synthesis.synthesized_output,
        )
        hitl_iterations.append(hitl_iteration)

        if current_synthesis.consensus_confidence >= 0.9:
            log_debug(
                f"HITL: confidence {current_synthesis.consensus_confidence:.0%} >= 90%, exiting loop"
            )
            break

    return current_synthesis, hitl_iterations


async def _generate_hitl_questionnaire_from_synthesis(
    models: list[ModelConfig],
    synthesis: ConsensusSynthesis,
    user_input: str,
    debug_mode: DebugMode,
    hitl_max_questions: int = 5,
) -> HITLQuestionnaire:
    """Generate HITL questionnaire from synthesis uncertainties.

    This function is called during Judge & Refine phase when HITL is enabled.
    It generates questions based on the areas_of_uncertainty in the synthesis.
    """
    log_debug("Generating HITL questionnaire from synthesis uncertainties")

    synthesis_model_config = max(models, key=lambda m: m.importance)

    # Build XML context for questionnaire generation
    uncertainties_xml = (
        "\n".join(
            [
                f'        <uncertainty importance="{u.importance:.2f}" area="{u.area}" reasoning="{u.reasoning}" />'
                for u in synthesis.areas_of_uncertainty
            ]
        )
        if synthesis.areas_of_uncertainty
        else "        <uncertainty>No uncertainties identified</uncertainty>"
    )

    instructions_text = f"""
<hitl_questionnaire_generation_instructions>
    <task>Generate a human-in-the-loop questionnaire based on consensus uncertainties.</task>
    <synthesis_context>
        <synthesis_output>{html.escape(synthesis.synthesized_output)}</synthesis_output>
        <synthesis_confidence>{synthesis.consensus_confidence:.2f}</synthesis_confidence>
        <areas_of_uncertainty>
{uncertainties_xml}
        </areas_of_uncertainty>
    </synthesis_context>
    <thinking_process>
        Before generating questions, you MUST use <thinking> tags to analyze:
        1. Review the synthesis confidence level - is it high enough to skip HITL?
        2. Examine each area of uncertainty - can user input genuinely resolve it?
        3. Prioritize uncertainties by importance - which matter most for the output quality?
        4. Consider what specific information from the user would be most valuable
        5. Plan clear, actionable questions with well-defined options
    </thinking_process>
    <requirements>
        <requirement>ALWAYS use <thinking> tags before generating questions to show your analysis</requirement>
        <requirement>ONLY generate questions if there are genuine uncertainties that user input can resolve</requirement>
        <requirement>If synthesis confidence is high (>90%) and no significant uncertainties, set should_skip_hitl to true</requirement>
        <requirement>Maximum {hitl_max_questions} questions</requirement>
        <requirement>Each question should have clear, actionable options derived from the uncertainties</requirement>
        <requirement>DO NOT fabricate questions - only ask about real uncertainties</requirement>
    </requirements>
    <output_format>
        <thinking>
        [Your analysis of whether HITL is needed and what questions would be valuable]
        </thinking>

        [Your questionnaire output - or set should_skip_hitl to true if not needed]
    </output_format>
</hitl_questionnaire_generation_instructions>
"""

    expected_output_text = prompts.build_synthesis_example()

    agent = create_agent_with_instructions(
        description="HITL Questionnaire Generator",
        instructions=instructions_text,
        expected_output=expected_output_text,
        model=synthesis_model_config.model,
        debug_mode=debug_mode,
        output_schema=HITLQuestionnaire,
    )

    response = await agent.arun(user_input)
    questionnaire = cast(HITLQuestionnaire, response.content)

    assert questionnaire is not None and isinstance(questionnaire, HITLQuestionnaire), (
        "Questionnaire generation failed"
    )
    return questionnaire


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
                input=user_input, output=current_output, print_results=debug_mode or False
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
            issues_xml = prompts.build_failed_criteria_xml(failed_criteria)

            instructions_text = prompts.build_refinement_instructions(
                main_agent_context=main_agent_context,
                failed_criteria_xml=issues_xml,
            )
            expected_output_text = prompts.build_refinement_example()

            refine_agent = create_agent_with_instructions(
                description="Consensus Refiner",
                instructions=instructions_text,
                expected_output=expected_output_text,
                model=judge_model.model,
                debug_mode=debug_mode,
                output_schema=RefinementResult,
            )

            refine_input = prompts.build_refinement_context_xml(current_output)
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
    hitl_iterations: list[HITLIteration] | None = None,
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

    synthesis_summary = f"Approach: {synthesis.synthesis_approach}. Insights: {len(synthesis.incorporated_insights)}, Conflicts: {len(synthesis.conflict_resolutions)}."

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
        hitl_iterations=hitl_iterations or [],
    )


def _inject_consensus_result(run_input: RunInput, consensus_result: ConsensusResult) -> None:
    formatted_result = prompts.build_consensus_result_xml(consensus_result, html.escape)

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
                skip_note = prompts.build_consensus_skipped_xml(
                    triage_decision.reason, triage_decision.category
                )
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

        synthesis, hitl_iterations = await _run_round_3_synthesis(
            models,
            generation_outputs,
            critiques,
            user_input,
            debug_mode,
            agent,
            hitl,
            hitl_max_questions,
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
            hitl_iterations,
        )

        log_debug(
            f"Consensus complete: confidence={consensus_result.consensus_confidence:.2%}, judge_score={consensus_result.judge_score:.2%}"
        )
        _inject_consensus_result(run_input, consensus_result)

        if auto_save_html and output_directory:
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
