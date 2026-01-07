#!/usr/bin/env python3
"""HITL Consensus Example

Demonstrates human-in-the-loop feedback during consensus:
- Consensus pre-hook runs (Generation → Critique → Synthesis)
- HITL questions are generated based on uncertainties
- Agent pauses for user input (multi-choice questions)
- User answers via continue_run
- Refinement runs with user preferences

Based on research.py continuation pattern.

This module docstring is **necessary** as it documents the public API
purpose and usage pattern of this example file.
"""

import asyncio
import os
import tempfile

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from rich.console import Console
from rich.prompt import Prompt

from blockether_foundation.agents.hooks import (
    ConsensusHooksConfig,
    JudgeCriteria,
    ModelConfig,
)
from blockether_foundation.agents.models.zai import Zhipu
from blockether_foundation.agents.storage.in_memory import InMemoryDb

console = Console()


async def main() -> None:
    """Run simple HITL consensus example."""

    MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
    MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")

    if not MODEL_BASE_URL:
        raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

    gpt_4o = OpenAIChat(
        id="gpt-4o",
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
        modalities=["text"],
    )

    # glm_4_7 = Zhipu(
    #     id="glm-4.7",
    #     enable_coding_plan=True,
    #     enable_thinking=True,
    # )

    consensus_config = ConsensusHooksConfig(
        models=[
            ModelConfig(
                model=gpt_4o,
                name="Analyst",
                importance=0.5,
                perspective="Focus on thorough analysis and edge cases.",
            ),
            ModelConfig(
                model=gpt_4o,
                name="Pragmatist",
                importance=0.5,
                perspective="Focus on practical, actionable solutions.",
            ),
        ],
        judge_criteria=[
            JudgeCriteria(
                name="Completeness",
                description="Does response cover all aspects?",
                weight=1.0,
                threshold=0.6,
            ),
        ],
        hitl=True,
        hitl_max_questions=3,
        skip_triage=True,
        max_refinement_iterations=2,
    )

    agent = Agent(
        model=gpt_4o,
        name="HITLConsensusAgent",
        description="An agent that gathers user feedback during consensus.",
        pre_hooks=[consensus_config.pre_hook()],
        markdown=True,
        debug_mode=True,
        db=InMemoryDb(),
    )

    console.print()
    console.rule("[bold blue]HITL Consensus Example[/]")
    console.print()
    console.print("[dim]This example shows human-in-the-loop feedback during consensus.[/]")
    console.print("[dim]The agent will pause to ask you clarifying questions.[/]")
    console.print()

    user_query = Prompt.ask("[bold green]Enter your question[/]")

    console.print("\n[dim]Running consensus with HITL...[/]\n")
    response = await agent.arun(user_query)

    while response.is_paused:
        console.print("\n[bold yellow]Agent needs your input:[/]\n")

        for requirement in response.active_requirements:
            if requirement.needs_user_input and requirement.user_input_schema:
                for field in requirement.user_input_schema:
                    console.print(f"[bold]{field.description}[/]\n")
                    answer = Prompt.ask(f"Your answer for '{field.name}'")
                    field.value = answer

        console.print("\n[dim]Continuing with your feedback...[/]\n")
        response = await agent.acontinue_run(
            run_id=response.run_id,
            requirements=response.requirements,
        )

    console.print("\n[bold green]Final Response:[/]\n")
    console.print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
