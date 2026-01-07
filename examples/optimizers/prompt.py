#!/usr/bin/env python3
"""
RACEF Framework Enforcer Agent Example.

This example demonstrates how to use RACEF_ENFORCER agent
to synthetically assess the quality of instructions from files or prompts.

Usage:
    python examples/racef.py

This example imports the pre-built RACEF_ENFORCER agent from
src/blockether_foundation/agents/racef.py and uses it with glm-4.7 model.

The agent accepts a file path or direct prompt and returns structured RACEF evaluation
with scores for each phase, improvement priorities, and recommendations.
"""

import asyncio
import os

from agno.db.sqlite import SqliteDb
from agno.tools.file import FileTools
from agno.utils import pprint
from rich.console import Console
from rich.prompt import Prompt

from blockether_foundation.agents.models.zai import Zhipu
from blockether_foundation.agents.vracef import VRACEF_ENFORCER
from blockether_foundation.utils import dataclass_copy

console = Console()

MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")
if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

console.print("[dim]Initializing RACEF Enforcer Agent Example...[/]")

glm_4_7 = Zhipu(
    id="glm-4.7",
    enable_coding_plan=True,
    enable_thinking=True,
)


async def main():
    """Run RACEF Enforcer Agent with interactive console interface."""
    console.print()
    console.print("[bold blue]RACEF Framework Enforcer Agent (Example)[/]")
    console.print("[dim]─────────────────────────────────────[/]")
    console.print()
    console.print("[bold green]Purpose:[/]")
    console.print(
        "  Synthetically assess the quality of instructions from files "
        "or prompts against RACEF (Reasoning, Assessment, Context, Execution, "
        "Feedback) framework and provides structured feedback with scoring, "
        "improvement priorities, and recommendations."
    )
    console.print()
    console.print("[bold green]What this example does:[/]")
    console.print(
        "  • Imports pre-built RACEF_ENFORCER from src/blockether_foundation/agents/racef.py"
    )
    console.print("  • Uses dataclass_copy pattern to configure the agent")
    console.print("  • Uses glm-4.7 model (like researcher.py example)")
    console.print("  • Provides FileTools for reading instruction files")
    console.print("  • Provides interactive console interface for testing RACEF evaluation")
    console.print()
    console.print("[bold green]RACEF Phases Evaluated:[/]")
    console.print("  [cyan]R[/] - Reasoning: Clear, logical step-by-step thinking")
    console.print("  [cyan]A[/] - Assessment: Self-evaluation and confidence scoring")
    console.print("  [cyan]C[/] - Context: Understanding environment, constraints, patterns")
    console.print("  [cyan]E[/] - Execution: Implementation with proper structure and validation")
    console.print("  [cyan]F[/] - Feedback: Learning from results and iteration")
    console.print()
    console.print("[bold green]Evaluation Capabilities:[/]")
    console.print("  • Evaluates across all 5 RACEF phases with weighted criteria (0.0-1.0)")
    console.print(
        "  • Adapts evaluation based on task complexity (simple: 0.65, moderate: 0.75, complex: 0.80)"
    )
    console.print(
        "  • Provides agent-type-specific criteria (frontend, backend, codebase analysis)"
    )
    console.print("  • Identifies critical failures that must be addressed (scores below 0.30)")
    console.print("  • Generates prioritized improvement recommendations")
    console.print(
        "  • Offers positive reinforcement for strengths and suggests how to leverage them"
    )
    console.print(
        "  • Makes specific recommendations for prompt engineers to improve agent system prompts"
    )
    console.print()
    console.print("[bold green]Usage:[/]")
    console.print("  1. Choose: (f) read instructions from file OR (p) provide direct prompt")
    console.print("  2. Get detailed RACEF evaluation with scores and recommendations")
    console.print("  3. Use recommendations to improve instruction quality")
    console.print()
    console.print("[bold green]Input Options:[/]")
    console.print("  [f]ile: Path to a file containing instructions/prompts")
    console.print("  [p]rompt: Direct text input of instructions to evaluate")
    console.print()
    console.print("[bold green]Output Format:[/]")
    console.print("  <racef_evaluation>")
    console.print("      <!-- Detailed XML evaluation with phase scores -->")
    console.print("  </racef_evaluation>")
    console.print()
    console.rule()

    agent = dataclass_copy(
        VRACEF_ENFORCER,
        model=glm_4_7,
        tools=[FileTools()],
        db=SqliteDb(),
        debug_mode=True,
    )

    console.print("[bold blue]RACEF Enforcer Agent Ready[/]")
    console.print("[dim]─────────────────────────────────────[/]")
    console.print()
    console.print("[dim]To test:[/]")
    console.print("  1. Choose file or prompt input")
    console.print("  2. Get detailed RACEF evaluation with scores and recommendations")
    console.print("  3. Use recommendations to improve instruction quality")
    console.print()
    console.print("[dim]Example Usage:[/]")
    console.print("  file: /path/to/instructions.md")
    console.print("  prompt: Here are the instructions I want to evaluate...")
    console.print()
    console.print("[dim]Example Output:[/]")
    console.print("  <racef_evaluation>")
    console.print("      <!-- Will receive structured XML evaluation -->")
    console.print("  </racef_evaluation>")
    console.print()
    console.rule()

    try:
        while True:
            console.print()
            console.print("[bold cyan]Choose input type (f=file, p=prompt, q=quit):[/] ")

            choice = (
                Prompt.ask(
                    "\n[bold cyan]Choose input type (f=file, p=prompt, q=quit):[/] ",
                    choices=["f", "p", "q"],
                    default="p",
                )
                .strip()
                .lower()
            )

            if choice == "q":
                console.print("[yellow]Exiting...[/]")
                break

            if choice == "f":
                file_path = Prompt.ask(
                    "\n[bold cyan]Enter file path containing instructions:[/] "
                ).strip()

                if not os.path.exists(file_path):
                    console.print(f"[red]Error: File not found: {file_path}[/]")
                    continue

                user_input = f"Please read and evaluate the instructions in this file: {file_path}"

            else:
                user_input = Prompt.ask(
                    "\n[bold cyan]Enter instructions to evaluate (or 'quit'):[/] ",
                    default="",
                ).strip()

                if "quit" in user_input.lower():
                    console.print("[yellow]Exiting...[/]")
                    break

                if not user_input:
                    console.print("[yellow]No input provided. Try again.[/]")
                    continue

                user_input = f"Please evaluate these instructions:\n\n{user_input}"

            response = await agent.arun(user_input)

            console.print()
            console.print("[bold blue]RACEF Evaluation:[/]")
            console.print()
            pprint.pprint_run_response(response)

    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


if __name__ == "__main__":
    asyncio.run(main())
