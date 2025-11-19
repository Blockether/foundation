"""ACE Program - Orchestrates the complete ACE learning loop.

The Program ties together Generator, Reflector, and Curator modules
to create a self-improving agent that learns from experience.
"""

from __future__ import annotations

import copy
import inspect
import logging
import os
from collections.abc import Callable
from re import split
from textwrap import dedent
from typing import Any, cast

from agno.agent import Agent, AgentSession
from agno.db.in_memory import InMemoryDb
from agno.guardrails import PromptInjectionGuardrail
from agno.models.base import Model
from agno.models.message import Message
from agno.models.openai import OpenAIChat
from agno.models.utils import get_model
from agno.run.agent import RunInput
from agno.session import TeamSession
from agno.team import Team
from agno.utils.log import log_debug
from agno.workflow import Step, StepInput, StepOutput, Workflow
from openai.types import ReasoningEffort
from pydantic import BaseModel, Field

from .models.base import BaseModelFilePersistable
from .models.playbook import EntryMetadataStatistic
from .models.program.analysis import AnalysisOutput, ProgramMode
from .models.program.curator import CuratorOutput
from .models.program.generator import GeneratorOutput
from .models.program.reflector import ReflectorOutput
from .playbook import Playbook

logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
)

type UserId = str | None
type DebugMode = bool | None

model = OpenAIChat(
    id="gpt-4o",
    base_url=os.getenv("BLOCKETHER_LLM_API_BASE_URL"),
    api_key=os.getenv("BLOCKETHER_LLM_API_KEY"),
)

playbook = Playbook()


class AceProgram(BaseModelFilePersistable):
    generator_model: Model = Field(
        description="Model used for generation in the ACE program"
    )
    premade_playbook: Playbook | None = Field(
        default=None, description="Optional premade playbook to use in the ACE program"
    )
    last_history_messages: int = Field(
        default=5, description="Number of past messages to use from session history"
    )
    enable_consensus: bool = Field(
        default=True,
        description="Whether to enable consensus-building across multiple models or agents",
    )
    consensus_models: list[Model] = Field(
        default_factory=list,
        description=("List of models to use for consensus-building if enabled."),
    )
    mode: ProgramMode = Field(
        default="answer",
        description="Operational mode of the ACE program."
        "   - 'hybrid': Combines learning and answering capabilities - fewer iterations than 'learn' mode."
        "   - 'learn': Focuses solely on learning from interactions - typically many iterations."
        "   - 'answer': Dedicated to providing answers based on existing knowledge - typically just one iteration.",
    )
    playbook_file_path: str | None = Field(
        default="playbook.json",
        description="Path to the JSON file for loading/saving the playbook. If None, file persistence is disabled.",
    )

    def __model_post_init__(self, context: Any) -> None:
        if self.enable_consensus:
            if (
                not self.generator_model.supports_json_schema_outputs
                and not self.generator_model.supports_native_structured_outputs
            ):
                raise ValueError(
                    "Generator model must support JSON Schema outputs or native structured outputs to coerce to agno input schema."
                )

            for model in self.consensus_models:
                if (
                    not model.supports_json_schema_outputs
                    and not model.supports_native_structured_outputs
                ):
                    raise ValueError(
                        "All consensus models must support JSON Schema outputs or native structured outputs."
                    )

                if not model.name:
                    raise ValueError("All consensus models must have a name.")

    def _consensus_markdown(self) -> str:
        if not self.enable_consensus or not self.consensus_models:
            return ""

        models_list = (
            ",  ".join(model.name for model in self.consensus_models if model.name)
            if self.enable_consensus
            else "N/A"
        )
        return dedent(
            f"""<CONSENSUS>
            <CONSENSUS_AVAILABLE_MODELS>{models_list}</CONSENSUS_AVAILABLE_MODELS>
        </CONSENSUS>"""
        )

    def change_mode(self, new_mode: ProgramMode) -> AceProgram:
        """
        Change the operational mode of the ACE program.

        Args:
            new_mode: The new mode to set ('hybrid', 'learn', or 'answer')

        Returns:
            Updated ProgramFactory with modified mode
        """
        self.mode = new_mode
        return self

    def _predict_run_analysis(
        self, executor: Agent | Team, input_as_str: str | None, playbook: Playbook
    ) -> AnalysisOutput:
        user_and_playbook_content = f"{input_as_str}{playbook.to_markdown()}"
        full_phase_input = dedent(
            f"""{user_and_playbook_content}

<PHASE>
    <PHASE_NAME>Analysis</PHASE_NAME>

    <METADATA>
        {self._consensus_markdown()}
    </METADATA>
</PHASE>"""
        )
        response = executor.run(stream=False, input=full_phase_input)
        content = cast(AnalysisOutput, response.content)
        log_debug(f"Analysis step response: {content}")

        return content

    def _stateless_agno_executor(
        self,
        executor: Agent | Team,
        output_schema: type[BaseModel] | None,
        model: Model | None | str = None,
    ) -> Agent | Team:
        changes = {
            "executor": executor,
            "pre_hooks": [],
            "post_hooks": [],
            "db": None,
            "memory_manager": None,
            "model": self.generator_model,
            "update_knowledge": False,
            "update_cultural_knowledge": False,
            "add_location_to_context": False,
            "input_schema": None,
        }

        if output_schema:
            changes["output_schema"] = output_schema  # type: ignore

        if model:
            changes["model"] = model  # type: ignore

        valid_params = inspect.signature(executor.__class__).parameters
        valid_changes = {k: v for k, v in changes.items() if k in valid_params}
        stateless_executor = copy.deepcopy(executor)

        for key, value in valid_changes.items():
            setattr(stateless_executor, key, value)

        if isinstance(stateless_executor, Team) and isinstance(executor, Team):
            stateless_executor.members = [
                self._stateless_agno_executor(member, output_schema)
                for member in executor.members
            ]

        return stateless_executor

    def _analysis_step(self, executor: Agent | Team, playbook: Playbook) -> Step:
        def step_executor(input: StepInput) -> StepOutput:
            input_as_str = input.get_input_as_string()
            stateless_executor = self._stateless_agno_executor(executor, AnalysisOutput)

            response = self._predict_run_analysis(
                executor=stateless_executor,
                input_as_str=input_as_str,
                playbook=playbook,
            )
            return StepOutput(
                content=response,
                images=input.images,
                audio=input.audio,
                videos=input.videos,
                files=input.files,
            )

        return Step(
            name="Analysis Step",
            description="Perform interactive analysis of the user's request",
            executor=step_executor,
        )

    def _model_with_reasoning(
        self, model: Model | str, effort: ReasoningEffort | None
    ) -> Model:
        model_with_reasoning = self._resolve_model(model)

        if not effort:
            return model_with_reasoning

        if not hasattr(model_with_reasoning, "reasoning"):
            log_debug(
                f"Model '{model_with_reasoning.name}' does not support reasoning effort settings."
            )
            return model_with_reasoning

        log_debug(
            f"Setting reasoning effort '{effort}' for model '{model_with_reasoning.name}'"
        )

        setattr(model_with_reasoning, "reasoning", {"effort": effort})  # noqa: B010

        return model_with_reasoning

    def _resolve_model(self, model: str | Model) -> Model:
        resolved_model = None

        if isinstance(model, str):
            provider, model_name = split(":", model)
            if not model_name and provider:
                provider, model_name = split("/", model)
                if not model_name and not provider:
                    raise ValueError("Invalid model string provided.")
                resolved_model = get_model(model_name, provider)  # type: ignore
            else:
                resolved_model = get_model(model_name, provider)  # type: ignore
        else:
            resolved_model = model
        return resolved_model

    def _reasoning_effort_from_analysis(
        self, analysis_step: AnalysisOutput
    ) -> ReasoningEffort | None:
        return (
            analysis_step.complexity.reasoning_level
            if analysis_step.complexity
            else None
        )

    def _generator_step(self, executor: Agent | Team, playbook: Playbook) -> Step:
        def step_executor(input: StepInput) -> StepOutput:
            stateless_executor = self._stateless_agno_executor(
                executor=executor,
                output_schema=GeneratorOutput,
            )

            full_phase_input = dedent(
                f"""
<PHASE>
    <PHASE_NAME>Generation</PHASE_NAME>

    <PHASE_TASK>
        You are an advanced problem-solving agent that applies accumulated strategic knowledge from the <PLAYBOOK> to generate accurate, well-reasoned answers for <USER_REQUEST>. Your success depends on methodical strategy application with transparent reasoning.

        1. Review the playbook carefully and identify relevant strategies
        2. Apply relevant insights from the playbook and show your step-by-step reasoning
        3. Provide your final answer
    </PHASE_TASK>
</PHASE>
{playbook.to_markdown()}
{input.get_input_as_string()}
                """
            )

            response = stateless_executor.run(full_phase_input, stream=False)

            return StepOutput(
                content=response.content,
                images=input.images,
                audio=input.audio,
                videos=input.videos,
                files=input.files,
            )

        return Step(
            name="Generation Step",
            description="Generate response based on user's request and playbook",
            executor=step_executor,
        )

    def _reflector_step(self, executor: Agent | Team, playbook: Playbook) -> Step:
        def step_executor(input: StepInput) -> StepOutput:
            previous_step: GeneratorOutput = cast(
                GeneratorOutput, input.previous_step_content
            )

            full_phase_input = dedent(
                f"""
<PHASE>
    <PHASE_NAME>Reflection</PHASE_NAME>

    <PHASE_TASK>
        You are a senior reviewer who diagnoses GENERATION_RESPONSE performance through systematic analysis, extracting concrete, actionable learnings from actual execution experiences to improve future performance.

        1. Analyze what went right or wrong
        2. Identify the root cause of any errors
        3. Provide actionable insights for improvement
        4. Tag each ground truth used as helpful/harmful/neutral
    </PHASE_TASK>
</PHASE>
{input.get_input_as_string()}

<GENERATION_RESPONSE>
    <REASONING>
        {previous_step.reasoning}
    </REASONING>

    <GROUND_TRUTHS_USED>
        {previous_step.ground_truths_used if previous_step.ground_truths_used else ''}
    </GROUND_TRUTHS_USED>

    <ANSWER>
        {previous_step.answer}
    </ANSWER>
</GENERATION_RESPONSE>
                """
            )

            stateless_executor = self._stateless_agno_executor(
                executor, ReflectorOutput
            )

            response = stateless_executor.run(full_phase_input, stream=False)

            # Update ground truth metadata based on reflector feedback
            reflector_output: ReflectorOutput = cast(ReflectorOutput, response.content)
            if (
                hasattr(reflector_output, "ground_truths")
                and reflector_output.ground_truths
            ):
                updated_count = 0
                for ground_truth_feedback in reflector_output.ground_truths:
                    if hasattr(ground_truth_feedback, "id") and hasattr(
                        ground_truth_feedback, "tag"
                    ):
                        # Cast the tag to EntryMetadataStatistic to ensure type safety
                        tag = cast(EntryMetadataStatistic, ground_truth_feedback.tag)
                        updated = playbook.update_ground_truth_metadata(
                            ground_truth_feedback.id, tag
                        )
                        if updated:
                            updated_count += 1
                log_debug(
                    f"Updated metadata for {updated_count} ground truth entries based on reflector feedback"
                )

            return StepOutput(
                content=response.content,
                images=input.images,
                audio=input.audio,
                videos=input.videos,
                files=input.files,
            )

        return Step(
            name="Reflection Step",
            description="Reflect on the generated response and user input",
            executor=step_executor,
        )

    def _curator_step(self, executor: Agent | Team, playbook: Playbook) -> Step:
        def step_executor(input: StepInput) -> StepOutput:
            previous_step: ReflectorOutput = cast(
                ReflectorOutput, input.previous_step_content
            )

            full_phase_input = dedent(
                f"""
<PHASE>
    <PHASE_NAME>Curation</PHASE_NAME>

    <PHASE_TASK>
        You are the playbook architect who transforms execution experiences into high-quality, atomic strategic updates based on REFLECTION_PHASE. Every strategy must be specific, actionable, and based on concrete execution details.

        ## WHEN TO UPDATE PLAYBOOK

        MANDATORY - Update when:
        ✓ Reflection reveals new error pattern
        ✓ Missing capability identified
        ✓ Strategy needs refinement based on evidence
        ✓ Contradiction between strategies detected
        ✓ Success pattern worth preserving

        FORBIDDEN - Skip updates when:
        ✗ Reflection too vague or theoretical
        ✗ Strategy already exists (>70% similar)
        ✗ Learning lacks concrete evidence
        ✗ Atomicity score below 40%

        ## NEGATIVE LEARNING SYSTEM

        ### HARMFUL STRATEGY MARKING (NEW SECTION)
        When Reflector identifies consistently problematic choices:
        - MARK strategies as harmful rather than removing them
        - CREATE new avoidance strategies: "Avoid [specific option/tool/service] - causes [specific problem type]"
        - TRACK patterns of failure for specific approaches across contexts

        ### CHOICE-CONSEQUENCE ANALYSIS (NEW SECTION)
        When processing Reflector feedback about option/tool/service selection:
        - IDENTIFY: Which specific choice contributed to success/failure?
        - PATTERN: Are there repeated outcomes with the same approach?
        - ACTION: Create evidence-based recommendations or mark harmful patterns

        ### STRATEGY SPECIFICITY RULES (NEW SECTION)
        - GENERIC strategies: "Use [category] approaches" (when no clear evidence)
        - SPECIFIC strategies: "Use [specific option] for [task type]" (only when evidence shows superiority)
        - AVOIDANCE strategies: "Avoid [specific option] - causes [problem type]" (when evidence shows consistent issues)
        - HARMFUL marking: Tag existing strategies that evidence shows are problematic
    </PHASE_TASK>
</PHASE>
{input.get_input_as_string()}

<REFLECTION_PHASE>
    <REASONING>
        {previous_step.reasoning}
    </REASONING>

    <ERROR_IDENTIFICATION>
        {previous_step.error_identification or ''}
    </ERROR_IDENTIFICATION>

    <ROOT_CAUSE_ANALYSIS>
        {previous_step.root_cause_analysis or ''}
    </ROOT_CAUSE_ANALYSIS>

    <CORRECT_APPROACH>
        {previous_step.correct_approach or ''}
    </CORRECT_APPROACH>

    <KEY_INSIGHTS>
        {previous_step.key_insights or ''}
    </KEY_INSIGHTS>
</REFLECTION_PHASE>

{playbook.to_markdown()}
"""
            )

            stateless_executor = self._stateless_agno_executor(executor, CuratorOutput)

            response = stateless_executor.run(full_phase_input, stream=False)

            return StepOutput(
                content=response.content,
                images=input.images,
                audio=input.audio,
                videos=input.videos,
                files=input.files,
            )

        return Step(
            name="Reflection Step",
            description="Reflect on the generated response and user input",
            executor=step_executor,
        )

    def _pre_hook_workflow(
        self, playbook: Playbook, executor: Agent | Team, debug_mode: bool
    ) -> Workflow:
        return Workflow(
            steps=[
                # self._analysis_step(executor, playbook=playbook),
                self._generator_step(executor, playbook=playbook),
                self._reflector_step(executor, playbook=playbook),
                self._curator_step(executor, playbook=playbook),
            ],
            debug_mode=debug_mode,
        )

    def _coerce_to_agent_input_schema(
        self,
        result_content: str,
        agent_input_schema: type[BaseModel],
    ) -> Any:
        try:
            response = self.generator_model.invoke(
                messages=[
                    Message(
                        role="system",
                        content="Generate output matching the requested response format",
                    ),
                    Message(role="user", content=result_content),
                ],
                response_format=agent_input_schema,
            )

            if not response.content:
                raise ValueError(
                    "Coercer model returned empty content when trying to coerce to agno input schema."
                )

            return response.content
        except Exception as e:
            raise ValueError(
                f"Failed to create response format from agent input schema: {e}"
            ) from e

    def pre_hook(
        self,
    ) -> Callable[
        [Agent | Team, RunInput, AgentSession | TeamSession, UserId, DebugMode], None
    ]:
        def hook(
            agent: Agent | Team,
            run_input: RunInput,
            session: AgentSession | TeamSession,
            user_id: UserId,
            debug_mode: DebugMode,
        ) -> None:
            history_content = self._get_conversation_history(session)
            session_data = session.session_data or {}
            playbook = self._get_playbook(session_data)

            self._set_session_playbook(session, playbook)

            user_request_content = (
                f"\n<USER_REQUEST>{run_input.input_content}</USER_REQUEST>"
            )
            input_content = f"{user_request_content}{history_content}"
            workflow = self._pre_hook_workflow(
                playbook=playbook, executor=agent, debug_mode=debug_mode or False
            )

            result = workflow.run(
                input=input_content,
                stream=False,
                images=list(run_input.images) if run_input.images else None,
                audio=list(run_input.audios) if run_input.audios else None,
                videos=list(run_input.videos) if run_input.videos else None,
            )

            generator_output: GeneratorOutput = cast(
                GeneratorOutput, result.step_results[0].content  # type: ignore
            )
            reflector_output: ReflectorOutput = cast(
                ReflectorOutput, result.step_results[1].content  # type: ignore
            )
            curator_output: CuratorOutput = cast(
                CuratorOutput, result.step_results[-1].content  # type: ignore
            )

            run_input.input_content = dedent(
                f"""Based on the following analysis and reflections, generate the final answer.
{input_content}

<GENERATION_PHASE>
    {generator_output.model_dump_json(indent=4)}
</GENERATION_PHASE>
<REFLECTION_PHASE>
    {reflector_output.model_dump_json(indent=4)}
</REFLECTION_PHASE>
<CURATION_PHASE>
    {curator_output.model_dump_json(indent=4)}
</CURATION_PHASE>
                """
            )

            # Apply curator deltas to playbook
            if (
                curator_output
                and hasattr(curator_output, "operations")
                and curator_output.operations is not None
                and curator_output
            ):
                deltas = playbook.curator_operations_to_deltas(
                    curator_output.operations
                )
                playbook = playbook.apply_deltas(deltas)
                log_debug(
                    f"Applied {len(deltas)} deltas to playbook from curator output"
                )

            self._set_session_playbook(session, playbook)

            playbook = self._get_playbook(session_data)
            if self.playbook_file_path:
                playbook.to_json_file(self.playbook_file_path)
                log_debug(f"Saved playbook to file: {self.playbook_file_path}")

            return None

        return hook

    def _get_conversation_history(self, session):
        history_messages: list[Message] = session.get_messages_for_session()[
            : self.last_history_messages
        ]

        plain_history_str = "\n".join(
            f"[{message.created_at}] {message.role}] {message.content}"
            for message in history_messages
        )

        history_str = (
            f"\n\n<CONVERSATION_HISTORY>{plain_history_str}</CONVERSATION_HISTORY>"
            if self.last_history_messages > 0
            else ""
        )

        log_debug(
            f"Session history for analysis: {plain_history_str if history_str else 'No history used.'}"
        )

        return history_str

    def _get_playbook(self, session_data: dict[str, Any]) -> Playbook:
        playbook = session_data.get("playbook")

        if not playbook:
            log_debug(
                "No playbook found in session data. Using premade, file, or default playbook."
            )

            # Try to load from file first
            try:
                playbook_file = session_data.get(
                    "playbook_file", self.playbook_file_path or "playbook.json"
                )
                if playbook_file:
                    playbook = cast(Playbook, Playbook.from_json_file(playbook_file))
                    log_debug(f"Loaded playbook from file: {playbook_file}")
                else:
                    raise FileNotFoundError("No playbook file path configured")
            except (FileNotFoundError, ValueError, KeyError):
                log_debug("No playbook file found or invalid file format.")

                if not self.premade_playbook:
                    log_debug(
                        "No premade playbook provided. Using default empty playbook."
                    )
                    playbook = Playbook()
                else:
                    log_debug("Using premade playbook provided to the ACE program.")
                    playbook = self.premade_playbook

        if isinstance(playbook, dict):
            playbook = Playbook.model_validate(playbook)

        return playbook

    def set_playbook_file(self, file_path: str) -> None:
        """
        Set the playbook file path to be used for loading/saving the playbook.

        Args:
            file_path: Path to the JSON file containing the playbook
        """
        self.playbook_file_path = file_path

    def load_playbook_from_file(self, file_path: str) -> Playbook:
        """
        Load a playbook from a JSON file.

        Args:
            file_path: Path to the JSON file containing the playbook

        Returns:
            Loaded Playbook object
        """
        return cast(Playbook, Playbook.from_json_file(file_path))

    def _set_session_playbook(
        self, session: AgentSession | TeamSession, playbook: Playbook
    ) -> None:
        session_data = session.session_data or {}
        session_data["playbook"] = playbook.model_dump()

        # if not session.session_data:
        # session.session_data["playbook"] = playbook
        # log_debug("Session playbook set.")

        #             name="ConditionallyPerformAnalysis",
        #             description="Check if interactive analysis is needed based on flag.",
        #             evaluator=lambda _input: self.enable_conversation_analysis,
        #             steps=[self._analysis_step()],
        #         )
        #     ],
        #     **kwargs,
        # )


ace_program = AceProgram(
    generator_model=model,
    premade_playbook=playbook,
    last_history_messages=5,
    enable_consensus=True,
    consensus_models=[model],
)
agent = Agent(
    model=model,
    pre_hooks=[PromptInjectionGuardrail(), ace_program.pre_hook()],
    debug_mode=True,
    db=InMemoryDb(),
)

if __name__ == "__main__":
    agent.cli_app(
        session_id="ace_program_session",
        stream=True,
    )
