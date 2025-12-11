"""Integration test utilities."""

from __future__ import annotations

import os

import scenario
from agno.agent.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.message import Message
from agno.models.openai import OpenAIChat
from pydantic import BaseModel


class CustomAgentAdapter(scenario.AgentAdapter):
    """Custom adapter that wraps agno agents for scenario framework."""

    def __init__(self, agent: Agent):
        """Initialize the adapter with an agno agent."""
        self.agent = agent
        self._responses = []

    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        # Get the current message count to identify new messages
        take_from = None

        try:
            take_from = len(self.agent.get_messages_for_session(input.thread_id)) + 1
        except Exception:
            take_from = 0
            pass

        # Run the Agno agent with the latest user message
        result = self.agent.run(input.last_new_user_message_str(), session_id=input.thread_id)

        # Extract only the new messages that were added during this call
        new_messages: list[Message] = (result.messages or [])[take_from:]

        # Format messages for Scenario (OpenAI format)
        openai_formatted_messages = [
            OpenAIChat()._format_message(message) for message in new_messages
        ]

        for msg in openai_formatted_messages:
            if msg["role"] == "developer":
                msg["role"] = "system"

        return openai_formatted_messages  # type: ignore

    def get_last_response(self) -> str | None:
        """Get the last response content."""
        if self._responses:
            return self._responses[-1].content
        return None

    def get_all_responses(self) -> list[str]:
        """Get all response contents."""
        return [resp.content for resp in self._responses]

    def reset(self) -> None:
        """Reset the adapter state."""
        self._responses.clear()


def create_agent_with_adapter(
    name: str,
    instructions: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model_id: str | None = None,
    output_schema: type[BaseModel] | None = None,
) -> CustomAgentAdapter:
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

    agent = Agent(
        model=llm,
        name=name,
        instructions=instructions,
        output_schema=output_schema,
        db=InMemoryDb(),
        debug_mode=True,
        debug_level=2,
    )

    return CustomAgentAdapter(agent)


def create_judge_agent(criteria: list[str] | None = None) -> scenario.JudgeAgent:
    """Create a judge agent for scenario validation."""
    return scenario.JudgeAgent(
        model=os.getenv("BLOCKETHER_LLM_DEFAULT_MODEL", "openai/gpt-4o"),
        api_key=os.getenv("BLOCKETHER_LLM_API_KEY"),
        base_url=os.getenv("BLOCKETHER_LLM_API_BASE_URL"),
        criteria=criteria,
    )
