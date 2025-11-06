from typing import Literal

from pydantic import BaseModel, Field

from ..base import ChainOfThoughts

type ProgramMode = Literal["hybrid", "learn", "answer"]


class ProgramModeRequest(ChainOfThoughts):
    requested_mode: ProgramMode | None = Field(
        description=(
            "The specific ACE program mode requested by the user. "
            "Leave `None` if the request is ambiguous or no mode was specified.\n\n"
            "**Available modes**:\n"
            "- `hybrid`: learning + answering\n"
            "- `learn`: information gathering only\n"
            "- `answer`: direct responses"
        ),
    )


class UserRequestComplexityAnalysis(ChainOfThoughts):
    iterations: int = Field(
        description=(
            "Estimated number of iterations the ACE program will need to "
            "complete the <USER_REQUEST> successfully taking into account the content of the <PLAYBOOK>. Use `1` for simple requests, "
            "`2-3` for moderate complexity, and `4+` for complex multi-step involving multiple tools requests."
        ),
        default=1,
    )

    reasoning_level: Literal["minimal", "low", "medium", "high"] = Field(
        description=(
            "**Depth of reasoning required**:\n"
            "- `minimal`: very little reasoning required\n"
            "- `low`: some reasoning required, but mostly straightforward\n"
            "- `medium`: multi-step synthesis\n"
            "- `high`: deep reasoning with hypotheses and cross-checks"
        )
    )


class RelevantHistoryItem(ChainOfThoughts):
    relevance_score: float = Field(
        description="Relevance score (0 no relevance, 1 - directly applicable).",
        ge=0.0,
        le=1.0,
    )

    summary: str = Field(
        description="Brief summary of the past interaction (Markdown preferred)."
    )

    key_insights: list[str] = Field(
        description=(
            "Key insights or takeaways from the past interaction that are relevant to "
            "the current request. Leave empty if there are none."
        ),
    )


class ProgramExpectedUserSpecifiedOutcome(ChainOfThoughts):
    outcome: str = Field(
        description="Description of the expected outcome specified in the prompt via system or user in markdown format"
    )


class MathPythonEquations(ChainOfThoughts):
    python_snippet: str = Field(
        description="Python snippet which will evaluate to the solution of the given equation. Prefer using SymPy library for symbolic mathematics."
    )
    equation: str = Field(
        description="The equation described in natural mathematical language (symbols)"
    )


class ConsensusModelRequirement(ChainOfThoughts):
    name: str = Field(
        description="Name of the model to involve in consensus for this sub-problem."
    )
    alias: str = Field(description="Short log/display name for this participant.")
    persona: str = Field(description="Behavior/style profile the model should adopt.")
    perspective: str = Field(
        description="Lens or position used when assessing this sub-problem."
    )


class SubProblemPlaybookEntry(ChainOfThoughts):
    identifier: int = Field(
        description="ID of the playbook entry to use for this sub-problem."
    )
    success_criteria: str = Field(
        description="How to verify this playbook entry was applied correctly. What observable outcomes indicate success?"
    )

    problem_type: Literal["math", "coding", "analysis", "writing"] = Field(
        description=(
            "Type of problem this sub-problem represents."
            "Choose one of the following: math, coding, analysis, writing."
            "- math - numerical calculations or equations"
            "- coding - programming tasks or code generation which is far more sophisticated than simple math and might require algorithmic thinking"
            "- analysis - data interpretation or logical reasoning"
            "- writing - content creation or text generation"
        )
    )

    consensus_models_requirements: list[ConsensusModelRequirement] | None = Field(
        default=None,
        description=(
            "Consensus models requirements for this sub-problem if applicable. Rule of thumb: If the user <USER_REQUEST> explicitly asks for consensus or mentions words like: 'ultrathink', 'collaborate', or 'groupthink', "
            "include relevant models here according to user specification or your best judgment. Leave `None` if consensus is not needed or not directly required by the sub-problem, playbook entry, or <USER_REQUEST>."
        ),
    )

    computable_math_solutions: list[MathPythonEquations] = Field(
        default_factory=list,
        description=(
            "Python-executable solutions for mathematical equations in this sub-problem. For any mathematical equations, provide Python snippets that use SymPy which is a python library for symbolic mathematics."
            "Empty if no mathematical computations are required."
        ),
    )


class DecomposedSubProblem(ChainOfThoughts):
    name: str = Field(description="Name or title of the sub-problem.")
    description: str | None = Field(
        default=None, description="Detailed description of the sub-problem."
    )
    suggested_entries_to_use: list[SubProblemPlaybookEntry] = Field(
        default_factory=list,
        description="List of playbook entries to use for this sub-problem. Should be empty if no playbook entries are relevant.",
    )


class AnalysisOutput(BaseModel):
    mode_request: ProgramModeRequest | None = Field(
        description=(
            "Analysis of the user's requested program mode. "
            "Leave `None` if no specific mode was requested or the request is ambiguous."
        ),
    )

    complexity: UserRequestComplexityAnalysis | None = Field(
        description=(
            "Analysis of <USER_REQUEST> complexity, including estimated iterations, "
            "complexity level, and reasoning depth. Leave `None` if it cannot be determined."
        ),
    )

    required_user_clarification: str | None = Field(
        description=(
            "Critical questions needed to fulfill the user's request. "
            "None if: user message is a greeting/pleasantry, acknowledgment, or self-explanatory. "
            "Only ask if the ambiguity genuinely prevents task completion."
        )
    )

    relevant_history: list[RelevantHistoryItem] = Field(
        default_factory=list,
        description=(
            "List of relevant past interactions that may inform the current request. "
            "Each item includes a summary and key insights. Leave empty if no relevant history exists."
        ),
    )

    problem_decomposition: list[SubProblemPlaybookEntry] = Field(
        description=(
            "Decomposition of the user's request into smaller, manageable sub-tasks or steps. "
            "Leave empty if the request is straightforward and does not require decomposition."
        ),
    )

    target_objective: ProgramExpectedUserSpecifiedOutcome | None = Field(
        description=(
            "Analysis of the expected high-level outcome/target objective as specified by the user in the prompt. "
            "Leave `None` if no specific outcome was mentioned or if you are unsure."
        )
    )

    user_request_language_used: str | None = Field(
        description=(
            "The natural language in which the user made their request e.g., 'English', 'Spanish', etc. "
            "Leave `None` if the language cannot be determined."
        ),
    )
