#!/usr/bin/env python3
"""
Simple Consensus Research Example

This example demonstrates a minimal consensus workflow that:
1. Takes a research query from user input (or command-line argument)
2. Runs multi-model consensus with Tavily and Arxiv research tools
3. Displays the consensus result to terminal
4. Copies the final output to clipboard

Usage:
    # Provide query as command-line argument
    python examples/consensus/simple.py "your research query here"

    # Or run interactively (will prompt for input)
    python examples/consensus/simple.py
"""

import asyncio
import os
import sys

from agno.agent import Agent
from agno.tools.arxiv import ArxivTools
from agno.tools.tavily import TavilyTools
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from blockether_foundation.agents.hooks import (
    ConsensusHooksConfig,
    JudgeCriteria,
    ModelConfig,
)
from blockether_foundation.agents.models.zai import Zhipu
from blockether_foundation.utils import copy_to_clipboard
from agno.models.openai import OpenAIChat

console = Console()

MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")
if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

console.print("[dim]Initializing models...[/]")

glm_4_7 = OpenAIChat(
    id="gpt-4.1",
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
    modalities=["text"],
)

glm_4_7 = Zhipu(
    id="glm-4.7",
    enable_coding_plan=True,
    enable_thinking=True,
)


async def main():
    """Run simple consensus research."""
    console.print()
    console.rule("[bold blue]Simple Consensus Research[/]")
    console.print()

    consensus_config = ConsensusHooksConfig(
        output_directory="./consensus_reports",
        auto_save_html=True,
        skip_triage=True,
        models=[
            ModelConfig(
                model=glm_4_7,
                name="Researcher",
                importance=0.5,
                perspective=(
                    "Focus on comprehensive research using Tavily for web searches "
                    "and Arxiv for academic papers. Gather factual information and "
                    "cite sources when possible."
                ),
                tools=[
                    TavilyTools(),
                    ArxivTools(),
                ],
            ),
            ModelConfig(
                model=glm_4_7,
                name="Analyst",
                importance=0.3,
                perspective=(
                    "Focus on analyzing and synthesizing the research findings. "
                    "Identify key insights, patterns, and actionable conclusions. "
                    "Provide a balanced, well-reasoned analysis."
                ),
            ),
            ModelConfig(
                model=glm_4_7,
                name="ConcisenessChecker",
                importance=0.2,
                perspective=(
                    "Focus on ensuring the final output is clear and concise. "
                    "The final response should avoid unnecessary information and be to the point."
                    "The more concise the better! Minus points for fluff."
                ),
            ),
        ],
        judge_criteria=[
            JudgeCriteria(
                name="Research Quality",
                description="Is the research thorough and well-sourced?",
                weight=1.0,
                threshold=0.6,
            ),
            JudgeCriteria(
                name="Clarity",
                description="Is the output clear and well-organized?",
                weight=0.8,
                threshold=0.6,
            ),
            JudgeCriteria(
                name="Conciseness",
                description="Is the response free of unnecessary information and to the point?",
                weight=0.8,
                threshold=0.6,
            ),
        ],
        max_refinement_iterations=2,
        judge_threshold=0.7,
    )

    # Create agent with consensus pre-hook
    agent = Agent(
        model=glm_4_7,
        name="ConsensusResearcher",
        description="Research agent that uses multi-model consensus.",
        instructions=[
            "You are a research assistant that receives pre-analyzed consensus from multiple AI models.",
            "The consensus result is provided in <multi_model_consensus> tags.",
            "Your job is to present the consensus findings clearly to the user.",
            "Keep your response concise but informative.",
            "Do not use markdown formatting - use plain text.",
            "When using the claude to make the prompt better please use the following skill: /prompt-engineering-skills:prompt-optimizer",
        ],
        tools=[
            TavilyTools(),
            ArxivTools(),
        ],
        pre_hooks=[consensus_config.pre_hook()],
        markdown=False,
        debug_mode=True,
    )

    console.print("[bold]This example runs multi-model consensus with:[/]")
    console.print("  - [cyan]Tavily[/] for web research")
    console.print("  - [cyan]Arxiv[/] for academic papers")
    console.print("  - [cyan]2 model perspectives[/] (Researcher + Analyst)")
    console.print()
    console.print("[dim]Results will be displayed and copied to clipboard.[/]")
    console.print()

    # Get research query from command-line argument or user prompt
    if len(sys.argv) > 1:
        # Use command-line argument
        query = " ".join(sys.argv[1:])
        console.print(f"[bold green]Query:[/] {query}")
    else:
        # Fall back to interactive prompt
        query = Prompt.ask("[bold green]Enter your research query[/]")

    if not query.strip():
        console.print("[yellow]No query provided. Exiting.[/]")
        return

    console.print()
    console.print("[dim]Running consensus research... (this may take a minute)[/]")
    console.print()

    try:
        response = await agent.arun(query)

        if response.content:
            result_text = str(response.content)

            # Display result
            console.print()
            console.print(Panel(result_text, title="[bold green]Consensus Result[/]", expand=False))
            console.print()

            # Copy to clipboard
            if copy_to_clipboard(result_text):
                console.print("[green]Result copied to clipboard.[/]")
            else:
                console.print("[yellow]Could not copy to clipboard.[/]")

        console.print()
        console.print("[dim]HTML report saved to ./consensus_reports/[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
