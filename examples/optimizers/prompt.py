#!/usr/bin/env python3
"""
V-RACEF Framework Orchestrator Example.

This example demonstrates how to use VRACEFOrchestrator
to synthetically assess the quality of instructions from files or prompts.

Usage:
    python examples/optimizers/prompt.py

This example imports VRACEFOrchestrator from
src/blockether_foundation/agents/vracef.py and uses it with glm-4.7 model.

The orchestrator coordinates specialized agents to evaluate instructions across
all V-RACEF phases with scores, improvement priorities, and recommendations.
"""

import asyncio
import os

from agno.agent import Agent
from rich.console import Console
from rich.prompt import Prompt

from blockether_foundation.agents.models.zai import Zhipu
from blockether_foundation.agents.vracef import VRACEFOrchestrator

console = Console()

MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")
if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

console.print("[dim]Initializing V-RACEF Orchestrator Example...[/]")

glm_4_7 = Zhipu(
    id="glm-4.7",
    enable_coding_plan=True,
    enable_thinking=True,
)


async def main():
    """Run V-RACEF Orchestrator with interactive console interface."""
    console.print()
    console.print("[bold blue]V-RACEF Framework Orchestrator (Example)[/]")
    console.print("[dim]─────────────────────────────────────[/]")
    console.print()
    console.print("[bold green]Purpose:[/]")
    console.print(
        "  Synthetically assess the quality of instructions from files "
        "or prompts against V-RACEF (Verification, Reasoning, Assessment, "
        "Context, Execution, Feedback) framework and provides structured "
        "feedback with scoring, improvement priorities, and recommendations."
    )
    console.print()
    console.print("[bold green]What this example does:[/]")
    console.print(
        "  • Imports VRACEFOrchestrator from src/blockether_foundation/agents/vracef.py"
    )
    console.print("  • Uses glm-4.7 model for evaluation")
    console.print(
        "  • Coordinates specialized agents for each V-RACEF phase evaluation"
    )
    console.print("  • Provides interactive console interface for testing V-RACEF evaluation")
    console.print()
    console.print("[bold green]V-RACEF Phases Evaluated:[/]")
    console.print("  [cyan]V[/] - Verification: Fact-check claims using Chain of Verification (CoVe)")
    console.print("  [cyan]R[/] - Reasoning: Clear, logical step-by-step thinking")
    console.print("  [cyan]A[/] - Assessment: Self-evaluation and confidence scoring")
    console.print("  [cyan]C[/] - Context: Understanding environment, constraints, patterns")
    console.print("  [cyan]E[/] - Execution: Implementation with proper structure and validation")
    console.print("  [cyan]F[/] - Feedback: Learning from results and iteration")
    console.print()
    console.print("[bold green]Evaluation Capabilities:[/]")
    console.print("  • Evaluates across all 6 V-RACEF phases with weighted criteria (0.0-1.0)")
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
    console.print("  2. Get detailed V-RACEF evaluation with scores and recommendations")
    console.print("  3. Use recommendations to improve instruction quality")
    console.print()
    console.print("[bold green]Input Options:[/]")
    console.print("  [f]ile: Path to a file containing instructions/prompts")
    console.print("  [p]rompt: Direct text input of instructions to evaluate")
    console.print()
    console.print("[bold green]Output Format:[/]")
    console.print("  <vracef_evaluation>")
    console.print("      <!-- Detailed XML evaluation with phase scores -->")
    console.print("  </vracef_evaluation>")
    console.print()
    console.rule()

    orchestrator = VRACEFOrchestrator(model=glm_4_7)

    console.print("[bold blue]V-RACEF Orchestrator Ready[/]")
    console.print("[dim]─────────────────────────────────────[/]")
    console.print()
    console.print("[dim]To test:[/]")
    console.print("  1. Choose file or prompt input")
    console.print("  2. Get detailed V-RACEF evaluation with scores and recommendations")
    console.print("  3. Use recommendations to improve instruction quality")
    console.print()
    console.print("[dim]Example Usage:[/]")
    console.print("  file: /path/to/instructions.md")
    console.print("  prompt: Here are the instructions I want to evaluate...")
    console.print()
    console.print("[dim]Example Output:[/]")
    console.print("  <vracef_evaluation>")
    console.print("      <!-- Will receive structured XML evaluation -->")
    console.print("  </vracef_evaluation>")
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

                with open(file_path) as f:
                    instructions_text = f.read()

                user_input = f"Please read and evaluate the instructions in this file: {file_path}"

            else:
                instructions_text = Prompt.ask(
                    "\n[bold cyan]Enter instructions to evaluate (or 'quit'):[/] ",
                    default="",
                ).strip()

                if "quit" in instructions_text.lower():
                    console.print("[yellow]Exiting...[/]")
                    break

                if not instructions_text:
                    console.print("[yellow]No input provided. Try again.[/]")
                    continue

                user_input = f"Please evaluate these instructions:\n\n{instructions_text}"

            # Create a test agent with the instructions to evaluate
            test_agent = Agent(
                name="TestAgent",
                instructions=instructions_text,
                description="Agent for V-RACEF evaluation",
            )

            # Run the orchestrator evaluation
            console.print()
            console.print("[dim]Running V-RACEF evaluation...[/]")

            evaluation = await orchestrator.evaluate_agent(
                agent=test_agent,
                test_input=user_input,
                expected_output="Expected output based on instructions",
            )

            console.print()
            console.print("[bold blue]V-RACEF Evaluation:[/]")
            console.print()

            # Print the XML output
            console.print(evaluation.to_xml())

            console.print()
            console.print("[bold green]Summary:[/]")
            console.print(f"  Overall Score: {evaluation.overall_assessment.weighted_score:.2f}")
            console.print(
                f"  Passes Threshold: {evaluation.overall_assessment.passes_threshold}"
            )
            console.print(f"  Threshold Used: {evaluation.overall_assessment.threshold_used:.2f}")

            if evaluation.critical_failures:
                console.print()
                console.print("[bold red]Critical Failures:[/]")
                for failure in evaluation.critical_failures:
                    console.print(f"  [{failure.phase.value}] {failure.description}")

            if evaluation.improvement_priorities:
                console.print()
                console.print("[bold yellow]Top Improvement Priorities:[/]")
                for priority in evaluation.improvement_priorities[:3]:
                    console.print(
                        f"  [{priority.phase.value}] {priority.description} "
                        f"(impact: {priority.expected_impact:.2f})"
                    )

    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


if __name__ == "__main__":
    asyncio.run(main())
