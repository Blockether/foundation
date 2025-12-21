"""Integration test utilities for Agno evals."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agno.agent.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.openai import OpenAIChat
from pydantic import BaseModel

if TYPE_CHECKING:
    from agno.run.agent import RunOutput
    from agno.run.team import TeamRunOutput
    from agno.session import AgentSession, TeamSession
    from agno.team import Team

# Global registry to store agent references for validators
AGENT_REGISTRY: dict[str, Agent] = {}

# Define proper hook type signatures to match Agno's expectations
PreHookType = Callable[
    [
        Union[Agent, "Team"],
        Any,  # RunInput
        Union["AgentSession", "TeamSession"],
        str,  # user_id
        bool,  # debug_mode
    ],
    Any,
]

PostHookType = Callable[
    [
        Union[Agent, "Team"],
        Union["RunOutput", "TeamRunOutput"],
        Union["AgentSession", "TeamSession"],
        str,  # user_id
        bool,  # debug_mode
    ],
    Any,
]


def create_agent_with_adapter(
    name: str,
    instructions: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model_id: str | None = None,
    output_schema: type[BaseModel] | None = None,
    pre_hooks: list[PreHookType] | None = None,
    post_hooks: list[PostHookType] | None = None,
) -> AgentWrapper:
    """Create an agno agent wrapped in our custom adapter."""
    # Use environment variables if not provided
    if api_key is None:
        api_key = os.getenv("BLOCKETHER_LLM_API_KEY")
    if base_url is None:
        base_url = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
    if model_id is None:
        model_id = os.getenv("BLOCKETHER_LLM_DEFAULT_MODEL", "gpt-4o")

    # Create LLM and agent
    llm = OpenAIChat(api_key=api_key, base_url=base_url, id=model_id)

    # Use InMemoryDb that supports both sync and async operations
    db = InMemoryDb()

    agent = Agent(
        model=llm,
        name=name,
        instructions=instructions,
        output_schema=output_schema,
        db=db,
        debug_mode=True,
        debug_level=2,
        pre_hooks=pre_hooks or [],  # type: ignore
        post_hooks=post_hooks or [],  # type: ignore
    )

    return AgentWrapper(agent)


def create_judge_agent(
    criteria: list[str] | None = None,
    instructions: str | None = None,
    name: str = "JudgeAgent",
    use_async: bool = False,
) -> Agent:
    """Create a judge agent for evaluation."""
    # Use environment variables if not provided
    api_key = os.getenv("BLOCKETHER_LLM_API_KEY")
    base_url = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
    model_id = os.getenv("BLOCKETHER_LLM_DEFAULT_MODEL", "gpt-4o")

    if criteria and not instructions:
        instructions = f"""You are a judge agent evaluating responses based on these criteria:

        {chr(10).join(f"- {criterion}" for criterion in criteria)}

        Score responses from 0-10 where:
        10 = Perfect, meets all criteria
        8-9 = Excellent, meets most criteria with minor issues
        6-7 = Good, meets some criteria but has noticeable issues
        0-5 = Poor, fails most or all criteria

        Be fair but strict in your evaluation."""

    # Create LLM and agent
    llm = OpenAIChat(api_key=api_key, base_url=base_url, id=model_id)

    # Use InMemoryDb that supports both sync and async operations
    db = InMemoryDb()

    return Agent(
        model=llm,
        name=name,
        instructions=instructions or "You are a judge agent evaluating responses.",
        db=db,
        debug_mode=True,
        debug_level=2,
    )


class AgentWrapper:
    """Wrapper that provides both agent and adapter-like interface."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.name = agent.name

    def get_agent(self) -> Agent:
        """Get the underlying Agno agent."""
        return self.agent
