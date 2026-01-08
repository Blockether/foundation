"""V-RACEF Framework Orchestrator.

Evaluates agent responses against V-RACEF (Verification, Reasoning, Assessment, Context, Execution, Feedback)
framework using specialized coordinated agents.

The V (Verification) phase implements Chain of Verification (CoVe) methodology from Meta AI research
(arXiv:2309.11495) to fact-check claims and reduce hallucinations before reasoning begins.
"""

from __future__ import annotations

import asyncio
import dataclasses as dc
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agno.agent import Agent
from agno.eval.accuracy import AccuracyEval, AccuracyResult
from agno.models.base import Model
from agno.utils.log import (
    log_debug,  # type: ignore
    log_error,  # type: ignore
    log_exception,  # type: ignore
    log_info,  # type: ignore
    log_warning,  # type: ignore
)
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from ..models import ChainOfThoughts
from ..utils import dataclass_copy

if TYPE_CHECKING:
    from .hooks.consensus.core import ConsensusHooksConfig


class DeltaType(str, Enum):
    ADD_INSTRUCTION = "add_instruction"
    UPDATE_INSTRUCTION = "update_instruction"
    REMOVE_INSTRUCTION = "remove_instruction"
    UPDATE_DESCRIPTION = "update_description"
    UPDATE_EXPECTED_OUTPUT = "update_expected_output"


class VRACEFPhase(str, Enum):
    VERIFICATION = "V"
    REASONING = "R"
    ASSESSMENT = "A"
    CONTEXT = "C"
    EXECUTION = "E"
    FEEDBACK = "F"


class PromptDelta(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    delta_type: DeltaType
    target_phase: VRACEFPhase | None = None
    content: str
    reasoning: str
    expected_impact: float = Field(ge=0.0, le=1.0)
    source_weakness: str | None = None


class TestCase(BaseModel):
    input: str
    expected_output: str
    observations: str | None = None
    id: str = Field(default_factory=lambda: str(uuid4())[:8])


class TestCaseResult(BaseModel):
    test_case_id: str
    passed: bool
    score: float
    actual_output: str | None = None
    error: str | None = None


class RegressionResult(BaseModel):
    passed: bool
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_score: float
    failures: list[TestCaseResult] = Field(default_factory=list[TestCaseResult])


class VRACEFScores(BaseModel):
    verification: float = 0.0
    reasoning: float = 0.0
    assessment: float = 0.0
    context: float = 0.0
    execution: float = 0.0
    feedback: float = 0.0
    overall: float = 0.0

    def weakest_phase(self) -> VRACEFPhase:
        phase_scores = {
            VRACEFPhase.VERIFICATION: self.verification,
            VRACEFPhase.REASONING: self.reasoning,
            VRACEFPhase.ASSESSMENT: self.assessment,
            VRACEFPhase.CONTEXT: self.context,
            VRACEFPhase.EXECUTION: self.execution,
            VRACEFPhase.FEEDBACK: self.feedback,
        }
        return min(phase_scores, key=lambda p: phase_scores[p])


class AgentSnapshot(BaseModel):
    id: str | None
    name: str | None
    description: str | None
    instructions: str
    expected_output: str | None


class OptimizationIteration(BaseModel):
    iteration: int
    timestamp: str
    delta_applied: PromptDelta | None
    before_score: float
    after_score: float
    regression_result: RegressionResult
    accepted: bool
    agent_snapshot: AgentSnapshot


class OptimizationHistory(BaseModel):
    agent_id: str
    agent_name: str
    started_at: str
    iterations: list[OptimizationIteration] = Field(default_factory=list[OptimizationIteration])
    final_score: float = 0.0
    total_deltas_applied: int = 0
    total_deltas_rejected: int = 0


class DeltaGeneratorResponse(BaseModel):
    chain_of_thoughts: ChainOfThoughts
    deltas: list[PromptDelta]


# ============================================================================
# Structured Output Models for VRACEF Agent System
# ============================================================================


class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class AgentType(str, Enum):
    FRONTEND = "Frontend"
    BACKEND = "Backend"
    GENERAL = "General"
    CODEBASE_ANALYSIS = "Codebase Analysis"
    OTHER = "Other"


class TaskAnalysis(BaseModel):
    """Output from VRACEFTaskAnalyzer - determines task complexity and agent type."""

    complexity: TaskComplexity
    agent_type: AgentType
    threshold: float = Field(ge=0.0, le=1.0)
    reasoning: str
    v_phase_importance: float = Field(
        description="How important V phase is for this task (0.0-1.0)"
    )


class Strength(BaseModel):
    description: str
    importance: float = Field(ge=0.0, le=1.0)


class Weakness(BaseModel):
    description: str
    importance: float = Field(ge=0.0, le=1.0)


class PhaseEvaluation(BaseModel):
    """Output from VRACEFPhaseEvaluator - evaluates a single V-RACEF phase."""

    phase: VRACEFPhase
    score: float = Field(ge=0.0, le=1.0)
    summary: str
    strengths: list[Strength] = Field(default_factory=list[Strength])
    weaknesses: list[Weakness] = Field(default_factory=list[Weakness])

    # V-phase specific fields
    claims_checked: int | None = Field(default=None)
    corrections_made: int | None = Field(default=None)


class OverallAssessment(BaseModel):
    """Output from VRACEFScorer - computes weighted overall score."""

    weighted_score: float = Field(ge=0.0, le=1.0)
    summary: str
    passes_threshold: bool
    threshold_used: float
    phase_scores: VRACEFScores


class CriticalFailure(BaseModel):
    """Represents a critical failure that must be addressed."""

    phase: VRACEFPhase
    description: str
    impact: str
    must_fix: str
    criticality: float = Field(ge=0.8, le=1.0)


class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImprovementPriority(BaseModel):
    """Represents an improvement recommendation with priority."""

    level: PriorityLevel
    phase: VRACEFPhase
    description: str
    suggested_fix: str
    expected_impact: float = Field(ge=0.0, le=1.0)


class PositiveReinforcement(BaseModel):
    """Positive feedback to reinforce good behavior."""

    praise: str = Field(description="Specific praise for excellent work")
    phase: VRACEFPhase
    insight: str = Field(description="Broader insight about agent's strengths")
    suggestion: str = Field(description="How to leverage these strengths")


class AgentRecommendation(BaseModel):
    """Specific recommendation for the agent."""

    priority: float = Field(ge=0.0, le=1.0)
    recommendation: str
    explanation: str
    example: str | None = Field(default=None)


class PromptEngineerRecommendation(BaseModel):
    """Suggestion for improving the agent's system prompt."""

    priority: float = Field(ge=0.0, le=1.0)
    recommendation: str
    explanation: str


class Recommendations(BaseModel):
    """Output from VRACEFRecommendationGenerator."""

    for_agent: list[AgentRecommendation] = Field(default_factory=list[AgentRecommendation])
    for_prompt_engineer: list[PromptEngineerRecommendation] = Field(
        default_factory=list[PromptEngineerRecommendation]
    )


class VRACEFEvaluation(BaseModel):
    """Complete V-RACEF evaluation result - orchestrates all agent outputs."""

    agent_name: str
    agent_type: AgentType
    task_description: str
    task_analysis: TaskAnalysis

    phase_evaluations: dict[VRACEFPhase, PhaseEvaluation]
    overall_assessment: OverallAssessment

    critical_failures: list[CriticalFailure] = Field(default_factory=list[CriticalFailure])
    improvement_priorities: list[ImprovementPriority] = Field(
        default_factory=list[ImprovementPriority]
    )
    positive_reinforcement: list[PositiveReinforcement] = Field(
        default_factory=list[PositiveReinforcement]
    )
    recommendations: Recommendations = Field(default_factory=Recommendations)

    def to_xml(self) -> str:
        """Convert evaluation to XML format for compatibility."""
        lines = ["<vracef_evaluation>", "  <agent_identity>"]
        lines.append(f"    <name>{self.agent_name}</name>")
        lines.append(f"    <type>{self.agent_type.value}</type>")
        lines.append(f"    <task_description>{self.task_description}</task_description>")
        lines.append("  </agent_identity>")
        lines.append("  <phase_scores>")

        for phase, eval in self.phase_evaluations.items():
            phase_tag = phase.value.lower()
            lines.append(f"    <{phase_tag}>")
            lines.append(f"      <score>{eval.score:.2f}</score>")
            lines.append(f"      <summary>{eval.summary}</summary>")
            lines.append("      <strengths>")
            for s in eval.strengths:
                lines.append(
                    f'        <strength importance="{s.importance:.2f}">{s.description}</strength>'
                )
            lines.append("      </strengths>")
            lines.append("      <weaknesses>")
            for w in eval.weaknesses:
                lines.append(
                    f'        <weakness importance="{w.importance:.2f}">{w.description}</weakness>'
                )
            lines.append("      </weaknesses>")
            lines.append(f"    </{phase_tag}>")

        lines.append("  </phase_scores>")
        lines.append("  <overall_assessment>")
        lines.append(
            f"    <weighted_score>{self.overall_assessment.weighted_score:.2f}</weighted_score>"
        )
        lines.append(f"    <summary>{self.overall_assessment.summary}</summary>")
        lines.append(
            f"    <passes_threshold>{str(self.overall_assessment.passes_threshold).lower()}</passes_threshold>"
        )
        lines.append(
            f"    <threshold_used>{self.overall_assessment.threshold_used:.2f}</threshold_used>"
        )
        lines.append("  </overall_assessment>")

        if self.critical_failures:
            lines.append("  <critical_failures>")
            for cf in self.critical_failures:
                lines.append(f'    <failure criticality="{cf.criticality:.2f}">')
                lines.append(f"      <phase>{cf.phase.value}</phase>")
                lines.append(f"      <description>{cf.description}</description>")
                lines.append(f"      <impact>{cf.impact}</impact>")
                lines.append(f"      <must_fix>{cf.must_fix}</must_fix>")
                lines.append("    </failure>")
            lines.append("  </critical_failures>")

        lines.append("</vracef_evaluation>")
        return "\n".join(lines)


VRACEF_TASK_ANALYZER = Agent(
    id="vracef-task-analyzer",
    name="V-RACEF Task Analyzer",
    output_schema=TaskAnalysis,
    instructions=dedent("""
        Analyze the task to determine complexity level and agent type.

        <thinking>
          1. ANALYZE TASK DESCRIPTION
             - What is the agent trying to accomplish?
             - How many steps are involved?
             - What dependencies or constraints exist?

          2. DETERMINE COMPLEXITY
             - Simple: Single-file changes, obvious bugs, straightforward questions
             - Moderate: Feature implementation, multi-file changes, moderate complexity
             - Complex: Architecture decisions, large refactors, novel problems

          3. IDENTIFY AGENT TYPE
             - Frontend: Visual design, UI/UX structure
             - Backend: API design, database interactions, business logic
             - General: Research, analysis, writing
             - Codebase Analysis: Understanding and navigating existing code

          4. SET THRESHOLD
             - Simple: 0.65 (focus on correctness)
             - Moderate: 0.75 (balanced RACEF compliance)
             - Complex: 0.80 (demand deep reasoning)

          5. ASSESS V-PHASE IMPORTANCE
             - High (0.8-1.0): Factual claims, technical specs, code behavior
             - Medium (0.4-0.7): Analysis with some factual elements
             - Low (0.0-0.3): Pure reasoning, creative tasks, opinions
        </thinking>

        <output_format>
        <thinking>
        [Your analysis of task complexity, agent type, threshold, and V-phase importance]
        </thinking>

        [Your TaskAnalysis structured output: complexity, agent_type, threshold, reasoning, v_phase_importance]
        </output_format>
    """),
)


VRACEF_PHASE_EVALUATOR = Agent(
    id="vracef-phase-evaluator",
    name="V-RACEF Phase Evaluator",
    output_schema=PhaseEvaluation,
    instructions=dedent("""
        Evaluate a single V-RACEF phase for the agent's response.

        <thinking>
          1. IDENTIFY THE PHASE
             - Which phase are you evaluating? (V, R, A, C, E, F)
             - What are the key criteria for this phase?

          2. ANALYZE THE RESPONSE
             - What did the agent do well in this phase?
             - What is missing or weak?
             - Quote specific evidence from the response

          3. ASSIGN SCORE (0.0-1.0)
             - Consider the criteria for this specific phase
             - Be objective and evidence-based
             - 0.9-1.0: Excellent, exceeds expectations
             - 0.7-0.9: Good, meets expectations
             - 0.5-0.7: Acceptable, has room for improvement
             - 0.3-0.5: Weak, needs significant improvement
             - 0.0-0.3: Critical failure

          4. IDENTIFY STRENGTHS
             - What specific behaviors or outputs were good?
             - Rate importance of each strength (0.0-1.0)

          5. IDENTIFY WEAKNESSES
             - What specific behaviors or outputs were lacking?
             - Rate importance of each weakness (0.0-1.0)

          6. V-PHASE SPECIFIC (if evaluating V phase)
             - Count factual claims extracted
             - Count corrections made after verification
        </thinking>

        <output_format>
        <thinking>
        [Your phase evaluation analysis - identify phase, analyze response, assign score with evidence, identify strengths and weaknesses]
        </thinking>

        [Your PhaseEvaluation structured output: phase, score, summary, strengths, weaknesses, claims_checked, corrections_made]
        </output_format>
    """),
)


VRACEF_SCORER = Agent(
    id="vracef-scorer",
    name="V-RACEF Scorer",
    output_schema=OverallAssessment,
    instructions=dedent("""
        Compute the overall V-RACEF score based on phase scores and task complexity.

        <thinking>
          1. REVIEW PHASE SCORES
             - Check all 6 phase scores
             - Identify which phases are strong/weak

          2. APPLY WEIGHTS BASED ON COMPLEXITY
             - Simple: V=0.10, R=0.18, A=0.18, C=0.18, E=0.18, F=0.18
             - Moderate: V=0.15, R=0.17, A=0.17, C=0.17, E=0.17, F=0.17
             - Complex: V=0.20, R=0.20, A=0.15, C=0.20, E=0.13, F=0.12

          3. COMPUTE WEIGHTED SCORE
             - Multiply each phase score by its weight
             - Sum the weighted scores

          4. CHECK THRESHOLD
             - Compare weighted score to threshold
             - Determine if passes

          5. WRITE SUMMARY
             - Overall assessment of V-RACEF compliance
             - Note any critical areas
        </thinking>

        <output_format>
        <thinking>
        [Your scoring analysis - review phase scores, apply weights, compute weighted score, check threshold, write summary]
        </thinking>

        [Your OverallAssessment structured output: weighted_score, summary, passes_threshold, threshold_used, phase_scores]
        </output_format>
    """),
)


VRACEF_IMPROVEMENT_PRIORITIZER = Agent(
    id="vracef-improvement-prioritizer",
    name="V-RACEF Improvement Prioritizer",
    output_schema=list[ImprovementPriority],  # type: ignore
    instructions=dedent("""
        Prioritize improvements based on phase evaluations and overall assessment.

        <thinking>
          1. IDENTIFY CRITICAL FAILURES
             - Any phase scoring below 0.30?
             - Any missing phases entirely?
             - Mark as CRITICAL priority

          2. FIND HIGH-IMPACT IMPROVEMENTS
             - Look for expected_impact > 0.7
             - These are HIGH priority

          3. FIND MEDIUM-IMPACT IMPROVEMENTS
             - Look for expected_impact 0.5-0.7
             - These are MEDIUM priority

          4. FOR EACH IMPROVEMENT
             - Which phase does it address?
             - What specifically needs improvement?
             - What is the suggested fix?
             - What is the expected impact?

          5. PRIORITIZE
             - Critical failures first
             - Then by expected impact (descending)
        </thinking>

        <output_format>
        <thinking>
        [Your prioritization analysis - identify critical failures, find high/medium/low impact improvements, prioritize by impact]
        </thinking>

        [Your list of ImprovementPriority structured outputs: level, phase, description, suggested_fix, expected_impact]
        </output_format>
    """),
)


VRACEF_REINFORCEMENT_GENERATOR = Agent(
    id="vracef-reinforcement-generator",
    name="V-RACEF Positive Reinforcement Generator",
    output_schema=list[PositiveReinforcement],  # type: ignore
    instructions=dedent("""
        Generate positive reinforcement to recognize and encourage good behavior.

        <thinking>
          1. IDENTIFY STRENGTHS
             - Which phases scored well (0.7+)?
             - What specific behaviors were good?

          2. ANALYZE IMPACT
             - Why do these strengths matter?
             - How do they contribute to overall quality?

          3. PROVIDE PRAISE
             - Specific praise for excellent work
             - Which phase does this relate to?

          4. OFFER INSIGHT
             - Broader insight about agent's capabilities
             - What does this say about the agent?

          5. SUGGEST LEVERAGE
             - How can these strengths be leveraged?
             - What should the agent continue doing?
        </thinking>

        <output_format>
        <thinking>
        [Your reinforcement analysis - identify strengths, analyze impact, provide praise, offer insight, suggest leverage]
        </thinking>

        [Your list of PositiveReinforcement structured outputs: praise, phase, insight, suggestion]
        </output_format>
    """),
)


VRACEF_RECOMMENDATION_GENERATOR = Agent(
    id="vracef-recommendation-generator",
    name="V-RACEF Recommendation Generator",
    output_schema=Recommendations,
    instructions=dedent("""
        Generate specific recommendations for the agent and prompt engineer.

        <thinking>
          FOR AGENT RECOMMENDATIONS:
          1. What specific changes should the agent make?
          2. Why would this help?
          3. Can you provide an example?
          4. Prioritize by impact (0.0-1.0)

          FOR PROMPT ENGINEER RECOMMENDATIONS:
          1. What system prompt changes would help?
          2. Why would these improve the agent?
          3. Prioritize by impact (0.0-1.0)

          Focus on:
          - Addressing the weakest phase
          - Leveraging the strongest phase
          - Practical, actionable suggestions
        </thinking>

        <output_format>
        <thinking>
        [Your recommendation analysis - for agent: specific changes with examples, for prompt engineer: system prompt changes, all prioritized by impact]
        </thinking>

        [Your Recommendations structured output: for_agent (list of AgentRecommendation), for_prompt_engineer (list of PromptEngineerRecommendation)]
        </output_format>
    """),
)


DELTA_GENERATOR_AGENT = Agent(
    id="vracef-delta-generator",
    name="V-RACEF Delta Generator",
    output_schema=DeltaGeneratorResponse,
    instructions=dedent("""
        You generate atomic prompt deltas to improve an agent's instructions based on V-RACEF evaluation feedback.

        <thinking_process>
          CRITICAL: Before generating deltas, you MUST think through the analysis using the <thinking> section.

          <thinking>
            1. ANALYZE V-RACEF EVALUATION
               - Which phase has the LOWEST score? (weakest_phase)
               - Are there any critical failures (scores < 0.30)?
               - What are the highest-priority improvement recommendations?

            2. UNDERSTAND THE AGENT
               - Read the current instructions
               - Read the description
               - Read the expected_output (if any)
               - Identify patterns and structure

            3. IDENTIFY ROOT CAUSE
               - What specific weakness is causing the low score?
               - Is it a missing instruction? Unclear instruction? Conflicting instruction?
               - Would the issue be better addressed in description or expected_output?

            4. GENERATE ATOMIC DELTA
               - Choose the BEST delta type for this issue
               - Write specific, actionable content (no vague "be better" statements)
               - Ensure it's reversible and doesn't break existing behavior
               - Explain WHY this delta will help (reasoning)
               - Estimate expected impact (0.0-1.0)

            5. VALIDATE DELTA
               - Is it atomic? (one change only)
               - Is it specific? (clear content)
               - Does it target the weakest phase?
               - Will it preserve existing working behavior?

            6. CONSIDER ADDITIONAL DELTAS
               - Can we address 2-3 related issues with multiple deltas?
               - Ensure each delta is independent
               - Prioritize by impact
          </thinking>

          ONLY AFTER completing this analysis, generate the structured output with your deltas.
        </thinking_process>

        INPUT:
        - Current agent instructions, description, and expected_output
        - V-RACEF evaluation with phase scores and improvement priorities
        - Failed test cases (if any)

        OUTPUT:
        Generate 1-3 atomic PromptDelta operations that address the HIGHEST PRIORITY weakness.

        DELTA TYPES:
        - add_instruction: Add a new instruction line to address a gap
        - update_instruction: Modify an existing instruction to be clearer/better
        - remove_instruction: Remove a conflicting or harmful instruction
        - update_description: Improve the agent's description
        - update_expected_output: Clarify what output format is expected

        RULES:
        1. ONE CHANGE AT A TIME - each delta should be atomic and reversible
        2. TARGET WEAKEST PHASE - focus on the phase with lowest score
        3. BE SPECIFIC - don't add vague instructions like "be better"
        4. PRESERVE EXISTING BEHAVIOR - don't break what's already working
        5. INCLUDE REASONING - explain why this delta will help

        PRIORITY ORDER:
        1. Fix critical failures (scores < 0.30)
        2. Address high-impact improvements (expected_impact > 0.7)
        3. Improve medium-impact areas (expected_impact 0.5-0.7)

        <output_format>
        <thinking>
        [Your analysis following the 6-step thinking process: analyze V-RACEF evaluation, understand agent, identify root cause, generate atomic delta, validate delta, consider additional deltas]
        </thinking>
        [Your DeltaGeneratorResponse structured output:
        - deltas: list of PromptDelta with operation_type, target_section, content, reasoning, expected_impact
        - weakest_phase: the phase with lowest score
        - expected_improvement: how much scores should improve]
        </output_format>
    """),
)


# ============================================================================
# VRACEF Orchestrator - Coordinates all specialized agents
# ============================================================================


class VRACEFOrchestrator:
    """Orchestrates multiple specialized VRACEF agents for comprehensive evaluation.

    This replaces the monolithic VRACEF_ENFORCER with a coordinated team of
    specialized agents, each with their own structured output schema.

    Usage:
        orchestrator = VRACEFOrchestrator(model=your_model)
        evaluation = await orchestrator.evaluate_agent(
            agent=agent_to_test,
            test_input=input_text,
            expected_output=expected_text
        )
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        self.task_analyzer = VRACEF_TASK_ANALYZER
        self.phase_evaluator = VRACEF_PHASE_EVALUATOR
        self.scorer = VRACEF_SCORER
        self.improvement_prioritizer = VRACEF_IMPROVEMENT_PRIORITIZER
        self.reinforcement_generator = VRACEF_REINFORCEMENT_GENERATOR
        self.recommendation_generator = VRACEF_RECOMMENDATION_GENERATOR

    async def evaluate_agent(
        self,
        agent: Agent,
        test_input: str,
        expected_output: str,
        observations: str | None = None,
    ) -> VRACEFEvaluation:
        """Run complete V-RACEF evaluation using specialized agents.

        Args:
            agent: The agent to evaluate
            test_input: Input to test the agent with
            expected_output: Expected output from the agent
            observations: Optional notes about the test case

        Returns:
            Complete V-RACEF evaluation with all phase scores and recommendations
        """
        # Get agent's response
        agent_response = agent.run(test_input)
        agent_output = str(agent_response.content) if agent_response.content else ""

        # Step 1: Analyze task complexity and agent type
        task_input = f"""
        <task_context>
            <agent_name>{agent.name or "Unknown"}</agent_name>
            <agent_instructions>{agent.instructions or ""}</agent_instructions>
            <task_input>{test_input}</task_input>
            <expected_output>{expected_output}</expected_output>
            <agent_output_preview>{agent_output[:500]}...</agent_output_preview>
        </task_context>
        """

        task_response = await self.task_analyzer.arun(task_input, model=self.model)
        task_analysis: TaskAnalysis = task_response.content  # type: ignore

        # Step 2: Evaluate each phase
        phase_evaluations: dict[VRACEFPhase, PhaseEvaluation] = {}

        for phase in VRACEFPhase:
            phase_input = f"""
            <phase_evaluation_request>
                <phase>{phase.value}</phase>
                <agent_name>{agent.name or "Unknown"}</agent_name>
                <agent_instructions>{agent.instructions or ""}</agent_instructions>
                <task_input>{test_input}</task_input>
                <expected_output>{expected_output}</expected_output>
                <actual_output>{agent_output}</actual_output>
                <observations>{observations or "None"}</observations>
                <v_phase_importance>{task_analysis.v_phase_importance}</v_phase_importance>

                <phase_criteria>
                    {
                "V": "Extract claims, generate verification questions, answer independently, correct errors",
                        "R": "Explicit reasoning chain, avoid fallacies, state assumptions, consider alternatives",
                        "A": "Confidence breakdown, identify uncertainty, self-critique, recognize limitations",
                        "C": "Understand codebase patterns, acknowledge constraints, ask clarifying questions",
                        "E": "Match planned approach, follow conventions, include error handling, provide testing",
                        "F": "Explain what was done, identify issues, provide confidence, suggest next steps"
                    }.get(phase.value)
                </phase_criteria>
            </phase_evaluation_request>
            """

            phase_response = await self.phase_evaluator.arun(phase_input, model=self.model)
            phase_eval: PhaseEvaluation = phase_response.content  # type: ignore
            phase_evaluations[phase] = phase_eval

        # Step 3: Compute overall score
        scorer_input = f"""
        <scoring_request>
            <complexity>{task_analysis.complexity.value}</complexity>
            <threshold>{task_analysis.threshold}</threshold>
            <phase_scores>
                <V>{phase_evaluations[VRACEFPhase.VERIFICATION].score}</V>
                <R>{phase_evaluations[VRACEFPhase.REASONING].score}</R>
                <A>{phase_evaluations[VRACEFPhase.ASSESSMENT].score}</A>
                <C>{phase_evaluations[VRACEFPhase.CONTEXT].score}</C>
                <E>{phase_evaluations[VRACEFPhase.EXECUTION].score}</E>
                <F>{phase_evaluations[VRACEFPhase.FEEDBACK].score}</F>
            </phase_scores>
        </scoring_request>
        """

        scorer_response = await self.scorer.arun(scorer_input, model=self.model)
        overall_assessment: OverallAssessment = scorer_response.content  # type: ignore

        # Step 4: Prioritize improvements
        priority_input = f"""
        <priority_request>
            <phase_evaluations>
                {
            json.dumps(
                {phase.value: eval.model_dump() for phase, eval in phase_evaluations.items()},
                indent=2,
            )
        }
            </phase_evaluations>
            <overall_assessment>
                {json.dumps(overall_assessment.model_dump(), indent=2)}
            </overall_assessment>
            <threshold>{task_analysis.threshold}</threshold>
        </priority_request>
        """

        priority_response = await self.improvement_prioritizer.arun(
            priority_input, model=self.model
        )
        improvement_priorities: list[ImprovementPriority] = priority_response.content  # type: ignore

        # Step 5: Generate positive reinforcement
        reinforcement_input = f"""
        <reinforcement_request>
            <phase_evaluations>
                {
            json.dumps(
                {phase.value: eval.model_dump() for phase, eval in phase_evaluations.items()},
                indent=2,
            )
        }
            </phase_evaluations>
        </reinforcement_request>
        """

        reinforcement_response = await self.reinforcement_generator.arun(
            reinforcement_input, model=self.model
        )
        positive_reinforcement: list[PositiveReinforcement] = reinforcement_response.content  # type: ignore

        # Step 6: Generate recommendations
        recommendation_input = f"""
        <recommendation_request>
            <agent_name>{agent.name or "Unknown"}</agent_name>
            <agent_type>{task_analysis.agent_type.value}</agent_type>
            <weakest_phase>{overall_assessment.phase_scores.weakest_phase().value}</weakest_phase>
            <phase_evaluations>
                {
            json.dumps(
                {phase.value: eval.model_dump() for phase, eval in phase_evaluations.items()},
                indent=2,
            )
        }
            </phase_evaluations>
            <improvement_priorities>
                {json.dumps([p.model_dump() for p in improvement_priorities], indent=2)}
            </improvement_priorities>
            <positive_reinforcement>
                {json.dumps([r.model_dump() for r in positive_reinforcement], indent=2)}
            </positive_reinforcement>
        </recommendation_request>
        """

        recommendation_response = await self.recommendation_generator.arun(
            recommendation_input, model=self.model
        )
        recommendations: Recommendations = recommendation_response.content  # type: ignore

        # Identify critical failures
        critical_failures = [
            CriticalFailure(
                phase=phase,
                description=f"Critical failure: score {eval.score:.2f} below 0.30",
                impact=f"Phase {phase.value} performance is critically low",
                must_fix=f"Immediate attention required for {phase.value} phase",
                criticality=1.0 - eval.score,
            )
            for phase, eval in phase_evaluations.items()
            if eval.score < 0.30
        ]

        # Assemble complete evaluation
        return VRACEFEvaluation(
            agent_name=agent.name or "Unknown",
            agent_type=task_analysis.agent_type,
            task_description=test_input[:200],
            task_analysis=task_analysis,
            phase_evaluations=phase_evaluations,
            overall_assessment=overall_assessment,
            critical_failures=critical_failures,
            improvement_priorities=improvement_priorities,
            positive_reinforcement=positive_reinforcement,
            recommendations=recommendations,
        )


@dataclass
class VRacefPromptOptimizer:
    evaluator_model: Model
    target_agent: Agent
    test_cases: list[TestCase] = field(default_factory=list[TestCase])
    history: OptimizationHistory | None = None
    min_accuracy_threshold: float = 7.0
    max_iterations: int = 10
    convergence_threshold: float = 0.02
    history_path: Path | None = None
    use_consensus_for_final_validation: bool = True
    consensus_config: ConsensusHooksConfig | None = None

    def __post_init__(self) -> None:
        self.history = OptimizationHistory(
            agent_id=self.target_agent.id or "unknown",
            agent_name=self.target_agent.name or "unknown",
            started_at=datetime.now(UTC).isoformat(),
        )

    def load_dataset(self, jsonl_path: str | Path) -> list[TestCase]:
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

        test_cases: list[TestCase] = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    test_cases.append(TestCase(**data))

        self.test_cases = test_cases
        log_info(f"Loaded {len(test_cases)} test cases from {jsonl_path}")
        return test_cases

    def _run_accuracy_eval(
        self, agent: Agent, test_case: TestCase, num_iterations: int = 1
    ) -> tuple[bool, float, str | None]:
        try:
            evaluation = AccuracyEval(
                model=self.evaluator_model,
                agent=agent,
                input=test_case.input,
                expected_output=test_case.expected_output,
                num_iterations=num_iterations,
            )
            result: AccuracyResult | None = evaluation.run(print_results=False)
            if result is None:
                return False, 0.0, "Evaluation returned None"

            passed = result.avg_score >= self.min_accuracy_threshold
            actual_output = None
            if result.results:
                actual_output = result.results[0].output if result.results[0].output else None

            return passed, result.avg_score, actual_output
        except Exception as e:
            log_exception(f"Error running accuracy eval: {e}")
            return False, 0.0, str(e)

    def run_regression_tests(self, agent: Agent) -> RegressionResult:
        if not self.test_cases:
            return RegressionResult(
                passed=True,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                avg_score=0.0,
            )

        results: list[TestCaseResult] = []
        total_score = 0.0

        for test_case in self.test_cases:
            passed, score, actual_output = self._run_accuracy_eval(agent, test_case)
            total_score += score

            result = TestCaseResult(
                test_case_id=test_case.id,
                passed=passed,
                score=score,
                actual_output=actual_output,
                error=None
                if passed
                else f"Score {score} below threshold {self.min_accuracy_threshold}",
            )
            results.append(result)

        passed_count = sum(1 for r in results if r.passed)
        failed_results = [r for r in results if not r.passed]

        return RegressionResult(
            passed=len(failed_results) == 0,
            total_cases=len(results),
            passed_cases=passed_count,
            failed_cases=len(failed_results),
            avg_score=total_score / len(results) if results else 0.0,
            failures=failed_results,
        )

    async def _run_vracef_evaluation(
        self, agent: Agent, test_case: TestCase
    ) -> tuple[str, VRACEFScores]:
        """Run V-RACEF evaluation using the new orchestrator with specialized agents.

        Returns:
            Tuple of (XML output string for compatibility, VRACEFScores object)
        """
        orchestrator = VRACEFOrchestrator(model=self.evaluator_model)
        evaluation = await orchestrator.evaluate_agent(
            agent=agent,
            test_input=test_case.input,
            expected_output=test_case.expected_output,
            observations=test_case.observations,
        )

        # Convert to XML for backward compatibility with existing code
        vracef_output = evaluation.to_xml()

        # Extract scores from the evaluation
        scores = VRACEFScores(
            verification=evaluation.phase_evaluations[VRACEFPhase.VERIFICATION].score,
            reasoning=evaluation.phase_evaluations[VRACEFPhase.REASONING].score,
            assessment=evaluation.phase_evaluations[VRACEFPhase.ASSESSMENT].score,
            context=evaluation.phase_evaluations[VRACEFPhase.CONTEXT].score,
            execution=evaluation.phase_evaluations[VRACEFPhase.EXECUTION].score,
            feedback=evaluation.phase_evaluations[VRACEFPhase.FEEDBACK].score,
            overall=evaluation.overall_assessment.weighted_score,
        )

        return vracef_output, scores

    def _parse_vracef_scores(self, vracef_output: str) -> VRACEFScores:
        import re

        scores = VRACEFScores()

        def extract_score(phase: str) -> float:
            pattern = rf"<{phase}>.*?<score>([\d.]+)</score>.*?</{phase}>"
            match = re.search(pattern, vracef_output, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return 0.0
            return 0.0

        scores.verification = extract_score("verification")
        scores.reasoning = extract_score("reasoning")
        scores.assessment = extract_score("assessment")
        scores.context = extract_score("context")
        scores.execution = extract_score("execution")
        scores.feedback = extract_score("feedback")

        overall_match = re.search(r"<weighted_score>([\d.]+)</weighted_score>", vracef_output)
        if overall_match:
            try:
                scores.overall = float(overall_match.group(1))
            except ValueError:
                scores.overall = 0.0

        if scores.overall == 0.0:
            phase_scores = [
                scores.verification,
                scores.reasoning,
                scores.assessment,
                scores.context,
                scores.execution,
                scores.feedback,
            ]
            non_zero = [s for s in phase_scores if s > 0]
            scores.overall = sum(non_zero) / len(non_zero) if non_zero else 0.0

        return scores

    async def _generate_deltas(
        self,
        agent: Agent,
        vracef_output: str,
        scores: VRACEFScores,
        failed_cases: list[TestCaseResult],
    ) -> list[PromptDelta]:
        delta_input = f"""
        <delta_generation_request>
            <current_agent>
                <name>{agent.name or "Unknown"}</name>
                <description>{agent.description or "None"}</description>
                <instructions>{agent.instructions or "None"}</instructions>
                <expected_output>{agent.expected_output or "None"}</expected_output>
            </current_agent>

            <vracef_evaluation>
                {vracef_output}
            </vracef_evaluation>

            <scores>
                <verification>{scores.verification}</verification>
                <reasoning>{scores.reasoning}</reasoning>
                <assessment>{scores.assessment}</assessment>
                <context>{scores.context}</context>
                <execution>{scores.execution}</execution>
                <feedback>{scores.feedback}</feedback>
                <overall>{scores.overall}</overall>
                <weakest_phase>{scores.weakest_phase().value}</weakest_phase>
            </scores>

            <failed_test_cases>
                {json.dumps([f.model_dump() for f in failed_cases], indent=2) if failed_cases else "None"}
            </failed_test_cases>
        </delta_generation_request>
        """

        response = await DELTA_GENERATOR_AGENT.arun(delta_input, model=self.evaluator_model)

        if response.content and isinstance(response.content, DeltaGeneratorResponse):
            return response.content.deltas

        return []

    def _apply_delta(self, agent: Agent, delta: PromptDelta) -> Agent:
        updates: dict[str, str] = {}

        raw_instructions = agent.instructions  # type: ignore
        if callable(raw_instructions):  # type: ignore
            current_instructions: str | list[str] = raw_instructions()  # type: ignore
            if isinstance(current_instructions, list):
                current_instructions = "\n".join(current_instructions)  # type: ignore
        elif isinstance(raw_instructions, list):
            current_instructions = "\n".join(raw_instructions)
        elif isinstance(raw_instructions, str):
            current_instructions = raw_instructions
        else:
            current_instructions = ""

        if delta.delta_type == DeltaType.ADD_INSTRUCTION:
            if current_instructions:
                updates["instructions"] = f"{current_instructions}\n{delta.content}"
            else:
                updates["instructions"] = delta.content

        elif delta.delta_type == DeltaType.UPDATE_INSTRUCTION:
            updates["instructions"] = delta.content

        elif delta.delta_type == DeltaType.REMOVE_INSTRUCTION:
            lines = current_instructions.split("\n")  # type: ignore
            updated_lines = [line for line in lines if delta.content.lower() not in line.lower()]  # type: ignore
            updates["instructions"] = "\n".join(updated_lines)  # type: ignore

        elif delta.delta_type == DeltaType.UPDATE_DESCRIPTION:
            updates["description"] = delta.content

        elif delta.delta_type == DeltaType.UPDATE_EXPECTED_OUTPUT:
            updates["expected_output"] = delta.content

        return dataclass_copy(agent, **updates)

    def _get_agent_snapshot(self, agent: Agent) -> AgentSnapshot:
        instructions_str = self._normalize_instructions(agent)

        return AgentSnapshot(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            instructions=instructions_str,
            expected_output=agent.expected_output,
        )

    def _normalize_instructions(self, agent: Agent) -> str:
        raw: str | list[str] | None = agent.instructions  # type: ignore[assignment]
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return "\n".join(raw)
        return ""

    async def _run_consensus_validation(
        self, agent: Agent, test_cases: list[TestCase]
    ) -> tuple[bool, float]:
        if not self.use_consensus_for_final_validation:
            return True, 1.0

        if self.consensus_config is None:
            log_info("No consensus config provided, skipping consensus validation")
            return True, 1.0

        total_score = 0.0
        passed_count = 0

        for test_case in test_cases:
            try:
                consensus_hook = self.consensus_config.pre_hook()
                response = await agent.arun(test_case.input, pre_hooks=[consensus_hook])

                if response.content:
                    passed_count += 1
                    total_score += 1.0
            except Exception as e:
                log_warning(f"Consensus validation failed for test case {test_case.id}: {e}")

        avg_score = total_score / len(test_cases) if test_cases else 0.0
        all_passed = passed_count == len(test_cases)

        log_info(
            f"Consensus validation: {passed_count}/{len(test_cases)} passed, avg score: {avg_score:.2f}"
        )
        return all_passed, avg_score

    def _save_history(self) -> None:
        if self.history_path and self.history:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("w") as f:
                f.write(self.history.model_dump_json(indent=2))
            log_info(f"Saved optimization history to {self.history_path}")

    async def aoptimize(
        self,
        jsonl_path: str | Path | None = None,
        sample_test_case: TestCase | None = None,
    ) -> Agent:
        """Async version of optimize - uses orchestrator with specialized agents."""
        if jsonl_path:
            self.load_dataset(jsonl_path)

        if not self.test_cases and not sample_test_case:
            raise ValueError(
                "No test cases provided. Either load from JSONL or provide sample_test_case."
            )

        if sample_test_case and sample_test_case not in self.test_cases:
            self.test_cases.append(sample_test_case)

        current_agent = self.target_agent
        baseline_regression = self.run_regression_tests(current_agent)
        current_score = baseline_regression.avg_score

        log_info(
            f"Baseline score: {current_score:.2f} ({baseline_regression.passed_cases}/{baseline_regression.total_cases} passed)"
        )

        eval_test_case = sample_test_case or self.test_cases[0]

        for iteration in range(self.max_iterations):
            log_info(f"Optimization iteration {iteration + 1}/{self.max_iterations}")

            vracef_output, scores = await self._run_vracef_evaluation(current_agent, eval_test_case)
            log_info(
                f"V-RACEF scores - V:{scores.verification:.2f} R:{scores.reasoning:.2f} A:{scores.assessment:.2f} C:{scores.context:.2f} E:{scores.execution:.2f} F:{scores.feedback:.2f} Overall:{scores.overall:.2f}"
            )

            if scores.overall >= 0.85 and baseline_regression.passed:
                log_info("Agent already has high V-RACEF compliance. Stopping optimization.")
                break

            deltas = await self._generate_deltas(
                current_agent, vracef_output, scores, baseline_regression.failures
            )

            if not deltas:
                log_info("No deltas generated. Stopping optimization.")
                break

            delta = deltas[0]
            log_info(f"Applying delta: {delta.delta_type.value} - {delta.reasoning[:100]}...")

            candidate_agent = self._apply_delta(current_agent, delta)
            regression_result = self.run_regression_tests(candidate_agent)

            iteration_record = OptimizationIteration(
                iteration=iteration + 1,
                timestamp=datetime.now(UTC).isoformat(),
                delta_applied=delta,
                before_score=current_score,
                after_score=regression_result.avg_score,
                regression_result=regression_result,
                accepted=False,
                agent_snapshot=self._get_agent_snapshot(candidate_agent),
            )

            should_accept = (
                regression_result.passed and regression_result.avg_score >= current_score
            )

            if should_accept and self.use_consensus_for_final_validation and self.consensus_config:
                log_info("Running consensus validation before accepting delta...")
                consensus_passed, _ = await self._run_consensus_validation(
                    candidate_agent, self.test_cases[:3]
                )
                if not consensus_passed:
                    log_info("Consensus validation failed. Rejecting delta.")
                    should_accept = False

            if should_accept:
                log_info(
                    f"Delta accepted! Score: {current_score:.2f} -> {regression_result.avg_score:.2f}"
                )
                current_agent = candidate_agent
                current_score = regression_result.avg_score
                iteration_record.accepted = True
                if self.history:
                    self.history.total_deltas_applied += 1
            else:
                log_info(
                    f"Delta rejected. Regression: {regression_result.passed}, Score: {regression_result.avg_score:.2f} (was {current_score:.2f})"
                )
                if self.history:
                    self.history.total_deltas_rejected += 1

            if self.history:
                self.history.iterations.append(iteration_record)

            self._save_history()

            if iteration > 0 and self.history and len(self.history.iterations) >= 2:
                prev_score = self.history.iterations[-2].after_score
                if abs(current_score - prev_score) < self.convergence_threshold:
                    log_info(
                        f"Convergence reached (delta < {self.convergence_threshold}). Stopping."
                    )
                    break

        if self.history:
            self.history.final_score = current_score

        self._save_history()

        log_info(f"Optimization complete. Final score: {current_score:.2f}")
        if self.history:
            log_info(
                f"Deltas applied: {self.history.total_deltas_applied}, rejected: {self.history.total_deltas_rejected}"
            )

        return current_agent

    def optimize(
        self,
        jsonl_path: str | Path | None = None,
        sample_test_case: TestCase | None = None,
    ) -> Agent:
        """Sync version of optimize - wraps async version."""

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.aoptimize(jsonl_path=jsonl_path, sample_test_case=sample_test_case)
        )


# ============================================================================
# Interactive Dataset Generator
# ============================================================================


class InteractiveDatasetGenerator:
    """Reusable interactive dataset generator for creating test cases with AI assistance.

    Provides a CLI for iteratively building test case datasets with AI-generated
    expected outputs, allowing users to accept, manually correct, or retry with
    specific instructions.

    Usage:
        generator = InteractiveDatasetGenerator(
            agent=YOUR_AGENT,
            model=YOUR_MODEL,
            dataset_path=Path("data/my_dataset.jsonl"),
            dataset_name="My Dataset"
        )
        await generator.run()
    """

    def __init__(
        self,
        agent: Agent,
        model: Model,
        dataset_path: Path,
        dataset_name: str = "Dataset",
        prompt_for_input: str = "Enter input text:",
        prompt_prefix: str = "",
    ):
        self.agent = agent
        self.model = model
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.prompt_for_input = prompt_for_input
        self.prompt_prefix = prompt_prefix

    def load_existing_entries(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        with self.dataset_path.open() as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(data)
        return entries

    def save_entry(
        self,
        input_text: str,
        expected_output: BaseModel | str,
        observations: str | None = None,
    ) -> None:
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        entry: dict[str, str | None] = {"input": input_text}

        if isinstance(expected_output, BaseModel):
            entry["expected_output"] = expected_output.model_dump_json()
        else:
            entry["expected_output"] = expected_output

        if observations is not None:
            entry["observations"] = observations

        with self.dataset_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def _format_output_for_display(self, output: BaseModel) -> str:
        return output.model_dump_json(indent=2)

    async def run_agent_proposal(
        self,
        input_text: str,
        correction_instructions: str | None = None,
        model: Model | None = None,
    ) -> BaseModel | None:
        agent = dataclass_copy(self.agent, model=model or self.model)

        try:
            if correction_instructions:
                full_input = (
                    f"{input_text}\n\n[correction]\n{correction_instructions}\n[/correction]"
                )
            else:
                full_input = input_text

            response = await agent.arun(full_input)
            content = response.content

            assert content is not None, "Agent returned None content"
            assert isinstance(content, BaseModel), f"Expected BaseModel, got {type(content)}"

            return content
        except AssertionError:
            return None
        except Exception as e:
            log_error(f"Agent error: {e}")
            return None

    async def run(self, console: Console | None = None) -> None:
        if console is None:
            console = Console()

        entries = self.load_existing_entries()

        console.print(
            Panel.fit(
                f"[bold cyan]{self.dataset_name}[/bold cyan]\n"
                "Create test cases with AI assistance.\n\n"
                f"Dataset: {self.dataset_path}\n"
                f"Existing entries: {len(entries)}\n\n"
                "Commands:\n"
                "  [green]c[/green] - Create new entry\n"
                "  [green]l[/green] - List existing entries\n"
                "  [green]x[/green] - Exit",
                title="Interactive Dataset Generator",
            )
        )

        from rich.prompt import Prompt as RichPrompt

        while True:
            action = RichPrompt.ask(
                "\n[bold]Action[/bold]",
                choices=["c", "l", "x"],
                default="c",
            )

            if action == "x":
                console.print("[dim]Goodbye![/dim]")
                break

            if action == "l":
                if not entries:
                    console.print("[dim]No entries yet.[/dim]")
                else:
                    console.print(f"\n[bold]Dataset entries: {len(entries)}[/bold]")
                    for idx, entry in enumerate(entries, 1):
                        preview = entry.get("input", "")[:80]
                        if len(entry.get("input", "")) > 80:
                            preview += "..."
                        obs = entry.get("observations", "none")
                        console.print(f"  {idx}. [dim]{obs}[/dim] | {preview}")
                continue

            # Create new entry
            console.print("\n[bold cyan]Creating new entry[/bold cyan]")
            console.print(f"[dim]{self.prompt_prefix}[/dim]")
            console.print(f"[dim]{self.prompt_for_input}[/dim]")
            console.print("[dim]Paste content, then press Enter to submit:[/dim]")

            lines: list[str] = []
            empty_count = 0
            while empty_count < 1:
                line = input()
                if not line:
                    empty_count += 1
                else:
                    empty_count = 0
                    lines.append(line)

            input_text = "\n".join(lines).strip()
            if not input_text:
                console.print("[yellow]Empty input, skipping.[/yellow]")
                continue

            console.print("\n[dim]Running agent...[/dim]")
            proposed = await self.run_agent_proposal(input_text)

            if proposed is None:
                console.print("[red]Agent failed to generate output.[/red]")
                continue

            console.print("\n[bold]Proposed Output:[/bold]")
            console.print(self._format_output_for_display(proposed))

            while True:
                action = RichPrompt.ask(
                    "[bold]Accept, Correct manually, Retry with instructions, or Skip?[/bold]",
                    choices=["a", "c", "r", "s"],
                    default="a",
                )

                if action == "s":
                    console.print("[dim]Skipped.[/dim]")
                    break

                if action == "a":
                    assert proposed is not None
                    observations = RichPrompt.ask(
                        "[dim]Optional observations (press Enter to skip)[/dim]",
                        default="",
                    )

                    self.save_entry(input_text, proposed, observations if observations else None)
                    console.print(f"[green]Saved to {self.dataset_path}[/green]")

                    entries = self.load_existing_entries()
                    break

                if action == "c":
                    assert proposed is not None
                    console.print(
                        "\n[yellow]Enter corrected output (or press Enter to accept proposed):[/yellow]"
                    )
                    console.print("[dim]Paste output, then press Enter to submit:[/dim]")

                    lines: list[str] = []
                    empty_count = 0
                    while empty_count < 1:
                        line = input()
                        if not line:
                            empty_count += 1
                        else:
                            empty_count = 0
                            lines.append(line)

                    corrected: BaseModel | str = "\n".join(lines) if lines else proposed

                    observations = RichPrompt.ask(
                        "[dim]Optional observations (press Enter to skip)[/dim]",
                        default="",
                    )

                    self.save_entry(input_text, corrected, observations if observations else None)
                    console.print(f"[green]Saved to {self.dataset_path}[/green]")

                    entries = self.load_existing_entries()
                    break

                if action == "r":
                    correction_instructions = RichPrompt.ask(
                        "[bold]Enter correction instructions for AI:[/bold]",
                        default="",
                    )

                    if not correction_instructions:
                        console.print("[yellow]No instructions provided, skipping retry.[/yellow]")
                        continue

                    console.print("\n[dim]Regenerating with instructions...[/dim]")
                    proposed = await self.run_agent_proposal(input_text, correction_instructions)

                    if proposed is None:
                        console.print("[red]Failed to regenerate, keeping previous output.[/red]")
                        continue

                    console.print("\n[bold]Proposed Output:[/bold]")
                    console.print(self._format_output_for_display(proposed))
                    continue
