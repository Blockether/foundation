"""Graph Agent Dataset Generator CLI for VRacef optimization."""

import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from agno.agent import Agent
from agno.models.base import Model
from agno.models.openai import OpenAIChat
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from blockether_foundation.agents.hooks.graph.agents import (
    INGESTION_AGENT,
    QUERY_AGENT,
)
from blockether_foundation.agents.models.zai import Zhipu
from blockether_foundation.agents.vracef import InteractiveDatasetGenerator, TestCase
from blockether_foundation.graph.models import LLMGraphOperations, LLMGraphQueryOperations
from blockether_foundation.utils import dataclass_copy

console = Console()

DATASET_PATH = Path(__file__).parent.parent.parent / "resources" / "optimizers" / "graph.jsonl"

MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")


class AgentType(str, Enum):
    INGESTION = "ingestion"
    QUERY = "query"


def get_agent_for_type(agent_type: AgentType) -> Agent:
    if agent_type == AgentType.INGESTION:
        return INGESTION_AGENT
    return QUERY_AGENT


def get_output_schema_for_type(agent_type: AgentType) -> type[BaseModel]:
    if agent_type == AgentType.INGESTION:
        return LLMGraphOperations
    return LLMGraphQueryOperations


async def run_graph_dataset_generator(model: Model) -> None:
    while True:
        console.print(
            Panel.fit(
                "[bold cyan]Graph Agent Dataset Generator[/bold cyan]\n"
                "Create training data for INGESTION and QUERY agents.\n\n"
                "Commands:\n"
                "  [green]i[/green] - Create INGESTION example (text -> entities/relationships)\n"
                "  [green]q[/green] - Create QUERY example (user request -> graph queries)\n"
                "  [green]x[/green] - Exit",
                title="VRacef Dataset Generator",
            )
        )

        action = Prompt.ask(
            "\n[bold]Action[/bold]",
            choices=["i", "q", "x"],
            default="i",
        )

        if action == "x":
            console.print("[dim]Goodbye![/dim]")
            break

        agent_type = AgentType.INGESTION if action == "i" else AgentType.QUERY
        agent = get_agent_for_type(agent_type)
        agent = dataclass_copy(agent, model=model)

        prompt_text = (
            "Enter text/story to extract entities from:"
            if agent_type == AgentType.INGESTION
            else "Enter user message/request for query generation:"
        )

        prompt_prefix = f"[bold cyan]Creating {agent_type.value.upper()} example[/bold cyan]\n"

        generator = InteractiveDatasetGenerator(
            agent=agent,
            model=model,
            dataset_path=DATASET_PATH,
            dataset_name=f"Graph {agent_type.value.upper()} Dataset",
            prompt_for_input=prompt_text,
            prompt_prefix=prompt_prefix,
        )

        await generator.run(console)


def main() -> None:
    if not MODEL_BASE_URL:
        console.print(
            "[red]Error: BLOCKETHER_LLM_API_BASE_URL environment variable is not set.[/red]"
        )
        return

    blockether_model = OpenAIChat(
        id="gpt-4o", timeout=60000, base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY
    )
    asyncio.run(run_graph_dataset_generator(blockether_model))


if __name__ == "__main__":
    main()
