"""ACE Program - Orchestrates the complete ACE learning loop.

The Program ties together Generator, Reflector, and Curator modules
to create a self-improving agent that learns from experience.
"""

from __future__ import annotations

import copy
import inspect
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
from .models.program.analysis import AnalysisStepOutput, ProgramAnalysis, ProgramMode
from .models.program.generator import GeneratorOutput
from .playbook import Playbook

type UserId = str | None
type DebugMode = bool | None

model = OpenAIChat(
    id="gpt-4o",
    base_url=os.getenv("BLOCKETHER_LLM_API_BASE_URL"),
    api_key=os.getenv("BLOCKETHER_LLM_API_KEY"),
)

playbook = Playbook()

ACEGenerator = Agent(
    name="ACE Generator",
    description="A generator part of the ACE self-improving agent framework, used to generate answers based on a playbook of strategies.",
    instructions=playbook.to_markdown(),
)


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
    ) -> AnalysisStepOutput:
        user_and_playbook_content = f"{input_as_str}{playbook.to_markdown()}"
        full_phase_input = dedent(
            f"""{user_and_playbook_content}

<PHASE>
    <PHASE_NAME>Analysis</PHASE_NAME>
    <PHASE_CONTEXT>
        - `mode_request`: What specific mode did user request? (None if unclear)
        - `complexity`: How many steps/reasoning depth needed?
        - `required_user_clarification`: What additional context you need from user to fulfill the <USER_REQUEST>?
            ONLY request clarifications if the user's intent is genuinely ambiguous and affects the response.
            DO NOT request clarifications for:
                * Greetings, pleasantries, or phatic expressions (e.g., "hello", "hi", "siema", "how are you")
                * Simple acknowledgments (e.g., "ok", "thanks", "got it")
                * Clear, self-contained questions or statements
                * Casual conversation that doesn't require task execution
            If the user's message is clear OR is a social pleasantry, return None.
        - `relevant_history`: What past interactions are relevant to this <USER_REQUEST>?
        - `problem_decomposition`: If the <USER_REQUEST> is complex, break it down into smaller sub-problems or steps. Leave empty if straightforward.
        - `target_objective`: What outcome did the user specify in the prompt? Should be provided only if unambiguous.
        - `user_request_language_used`: Should be the language of the POLICY if the POLICY specifies a language, otherwise default to the user's language.
    </PHASE_CONTEXT>

    <METADATA>
        {self._consensus_markdown()}
    </METADATA>
</PHASE>"""
        )
        response = executor.run(stream=False, input=full_phase_input)
        log_debug(f"Analysis step response: {response.content}")

        return AnalysisStepOutput(
            next_context=(
                f"{user_and_playbook_content}\n\n<PREVIOUS_PHASE_RESPONSE>{response.content}</PREVIOUS_PHASE_RESPONSE>"
                if response.content
                else user_and_playbook_content
            ),
            analysis=response.content,
        )

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
            stateless_executor = self._stateless_agno_executor(
                executor, ProgramAnalysis
            )

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
                resolved_model = get_model(model_name, provider)
            else:
                resolved_model = get_model(model_name, provider)
        else:
            resolved_model = model
        return resolved_model

    def _reasoning_effort_from_analysis(
        self, analysis_step: AnalysisStepOutput
    ) -> ReasoningEffort | None:
        return (
            analysis_step.analysis.complexity.reasoning_level
            if analysis_step.analysis and analysis_step.analysis.complexity
            else None
        )

    def _generator_step(self, executor: Agent | Team, playbook: Playbook) -> Step:
        def step_executor(input: StepInput) -> StepOutput:
            previous_step: AnalysisStepOutput = cast(
                AnalysisStepOutput, input.previous_step_content
            )
            model_with_reasoning = self._model_with_reasoning(
                self.generator_model,
                effort=self._reasoning_effort_from_analysis(previous_step),
            )
            stateless_executor = self._stateless_agno_executor(
                executor=executor,
                output_schema=GeneratorOutput,
                model=model_with_reasoning,
            )
            response = stateless_executor.run(previous_step.next_context, stream=False)

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

    def _pre_hook_workflow(
        self, playbook: Playbook, executor: Agent | Team, debug_mode: bool
    ) -> Workflow:
        return Workflow(
            steps=[
                self._analysis_step(executor, playbook=playbook),
                self._generator_step(executor, playbook=playbook),
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
            agent_input_schema = agent.input_schema
            history_content = self._get_conversation_history(session)
            session_data = session.session_data or {}
            playbook = self._get_playbook(session_data)

            self._set_session_playbook(session, playbook)

            user_request_content = (
                f"\n\n<USER_REQUEST>{run_input.input_content}</USER_REQUEST>"
            )
            input_content = (
                "<PLAYBOOK> defines the execution guidelines and context. <PHASE> if present specifies which decomposed sub-problem or workflow step you are currently addressing."
                "Using the <PLAYBOOK> and considering the <PHASE>, fulfill the <USER_REQUEST>:"
                "\n\n------------------------------------------------------------------"
                f"{user_request_content}{history_content}"
                "\n\n------------------------------------------------------------------\n"
            )
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

            run_input.input_content = (
                self._coerce_to_agent_input_schema(
                    result_content=result.get_content_as_string(),
                    agent_input_schema=agent_input_schema,
                )
                if agent_input_schema is not None
                else result.get_content_as_string()
            )
            # new_play# playbook.apply_deltas(result.get_playbook_deltas())

            self._set_session_playbook(session, playbook)
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
            f"\n\nTake into account the following conversation history:\n\n<CONVERSATION_HISTORY>{plain_history_str}</CONVERSATION_HISTORY>"
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
                "No playbook found in session data. Using premade or default playbook."
            )

            if not self.premade_playbook:
                log_debug("No premade playbook provided. Using default empty playbook.")
                playbook = Playbook()
            else:
                log_debug("Using premade playbook provided to the ACE program.")
                playbook = self.premade_playbook

        if isinstance(playbook, dict):
            playbook = Playbook.model_validate(playbook)

        return playbook

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


# ACEGenerator = Agent(
#     model=
#     name="Taking into account the provided playbook perform the task as best as you can.",
#     description="A generator part of the ACE self-improving agent framework, used to generate answers based on a playbook",
#     instructions=playbook.to_markdown(),
#     debug_mode=True,
# )

# ACEReflector = Agent(
#     model=OpenAIChat(
#         id="gpt-4o",
#         base_url=os.getenv("BLOCKETHER_LLM_API_BASE_URL"),
#         api_key=os.getenv("BLOCKETHER_LLM_API_KEY"),
#     ),
#     name="ACE Reflector",
#     description="Analyzes generator outputs to identify issues and improvement opportunities based on the provided playbook.",
#     instructions=dedent(f"""
#         Taking into account the
#         {playbook.to_markdown()}
#     """),
#     debug_mode=True,
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

agent.cli_app(
    session_id="ace_program_session",
    stream=True,
)
