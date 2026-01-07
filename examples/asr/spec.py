#!/usr/bin/env python3
"""
Specification Agent with Audio Mode

This agent helps create specifications in a single markdown file.

Features:
1. Multi-model consensus for research and analysis
2. ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) support
3. SQLite storage for session persistence
4. Goal management for tracking specification progress

Usage:
    python examples/asr/spec.py

Audio Mode:
- Press SPACE to record audio
- Your speech will be transcribed and processed
- Responses will be spoken via TTS
"""

import asyncio
import os

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.base import Model
from agno.models.openai import OpenAIChat
from agno.tools.arxiv import ArxivTools
from agno.tools.file import FileTools
from agno.tools.tavily import TavilyTools
from agno.utils import pprint
from rich.console import Console

from blockether_foundation.agents.hooks import (
    ConsensusHooksConfig,
    JudgeCriteria,
    ModelConfig,
    TranscriptionHooksConfig,
    TTSHooksConfig,
)
from blockether_foundation.agents.models.zai import Zhipu
from blockether_foundation.agents.toolkits import GoalToolkit
from blockether_foundation.asr import LocalWhisperAudioTranscriber
from blockether_foundation.tts import PIPER_DEFAULT_MODEL, PiperTTS

console = Console()


MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")

if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

console.print("[dim]Initializing models...[/]")

gpt_4o = OpenAIChat(
    id="gpt-4o",
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
    modalities=["text"],
)

gpt_41 = OpenAIChat(
    id="gpt-4.1",
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
    modalities=["text"],
)

glm_4_7 = Zhipu(
    id="glm-4.7",
    enable_thinking=True,
    enable_coding_plan=True,
)


def configure_consensus(
    models: list[ModelConfig],
    triage_model: Model,
) -> ConsensusHooksConfig:
    """Configure consensus for specification research.

    Uses only TavilyTools, FileTools, and GoalTools for research.
    """
    return ConsensusHooksConfig(
        output_directory="./specification_consensus_reports",
        auto_save_html=False,
        models=models,
        judge_criteria=[
            JudgeCriteria(
                name="Specification Completeness",
                description="Does not specification cover all necessary aspects?",
                weight=1.0,
                threshold=0.6,
            ),
            JudgeCriteria(
                name="Clarity and Readability",
                description="Is specification clear and easy to understand?",
                weight=0.8,
                threshold=0.6,
            ),
            JudgeCriteria(
                name="Feasibility",
                description="Are specified requirements feasible?",
                weight=0.9,
                threshold=0.65,
            ),
            JudgeCriteria(
                name="Actionability",
                description="Can specification be implemented based on this?",
                weight=0.8,
                threshold=0.6,
            ),
        ],
        triage_model=triage_model,
        max_refinement_iterations=2,
        judge_threshold=0.65,
    )


async def main():
    """Run specification agent with audio mode, consensus, ASR/TTS, and SQLite storage."""

    console.print("[bold]Setting up Specification Agent...[/]")

    # Setup ASR
    console.print("[dim]Initializing Whisper transcriber (turbo model)...[/]")
    transcriber = LocalWhisperAudioTranscriber(model_id="turbo")

    # Setup TTS
    console.print(f"[dim]Downloading Piper TTS model: {PIPER_DEFAULT_MODEL}[/]")
    model_path = PiperTTS.pre_download(
        model_name=PIPER_DEFAULT_MODEL,
        download_root="./models/piper",
    )
    synthesizer = PiperTTS(model_path=model_path)

    consensus_config = configure_consensus(
        models=[
            ModelConfig(
                model=gpt_41,
                name="SpecificationArchitect",
                importance=0.35,
                tools=[TavilyTools(), ArxivTools()],
                perspective=(
                    "Focus on specification architecture and structure. "
                    "Ensure the specification is well-organized, complete, "
                    "and follows best practices for technical documentation."
                ),
            ),
            ModelConfig(
                model=gpt_41,
                name="RequirementsAnalyst",
                importance=0.30,
                perspective=(
                    "Focus on requirements analysis and completeness. "
                    "Identify missing requirements, edge cases, "
                    "and ensure all user needs are captured."
                ),
            ),
            ModelConfig(
                model=gpt_41,
                name="TechnicalReviewer",
                importance=0.20,
                perspective=(
                    "Focus on technical feasibility and implementation concerns. "
                    "Identify potential technical challenges, dependencies, "
                    "and provide implementation guidance."
                ),
            ),
            ModelConfig(
                model=gpt_41,
                name="QualityAssurance",
                importance=0.15,
                perspective=(
                    "Focus on quality, clarity, and readability. "
                    "Ensure the specification is clear, actionable, "
                    "and free of ambiguity."
                ),
            ),
        ],
        triage_model=gpt_4o,
    )

    # Initialize hooks configuration
    transcription_config = TranscriptionHooksConfig(
        transcriber=transcriber,
        language="en",
        effort=0.5,
        save_transcriptions=False,
        enable_interactive_recording=True,
    )

    tts_config = TTSHooksConfig(
        synthesizer=synthesizer,
        save_to_file=False,
        auto_speak=True,
        interruptible=True,
    )

    # Create agent with SQLite storage
    agent = Agent(
        model=glm_4_7,
        name="SpecificationAgent",
        description=(
            "You are a specification creation assistant that helps create "
            "comprehensive specifications in markdown format. You use multi-model "
            "consensus for research and analysis."
        ),
        instructions="""
        # Specification Generation Task

        **Primary Goal:** Generate comprehensive product specifications in markdown format. The focus is on clearly describing the product, its objectives, requirements, and constraints. Implementation details are out of scope.

        ---

        ## Context and Scope

        - **Product Focus:** [Specify: Software / Hardware / Service / General Product]
        - **Specification Purpose:** Define the product for
        - **Output Format:** Markdown (.md) file
        - **Research Tools:** You may use available tools to research and gather information as needed
        - **Boundaries:**
            - Focus on product description and requirements only
            - Do not address implementation approaches
            - Do not ask the user about implementation
            - Do not proceed to coding or solution design

        ---

        ## Required Specification Structure

        Your markdown specification must include the following sections:

        ```markdown
        # [Product Name] Specification

        ## 1. Overview
        - Brief description of the product
        - Problem it addresses
        - Target users/stakeholders

        ## 2. Objectives
        - Primary goals and success criteria
        - Business value and expected outcomes
        - Measurable objectives (quantitative where possible)

        ## 3. Product Description
        - Detailed product description
        - Key features and capabilities
        - User personas and use cases

        ## 4. Functional Requirements
        - List of required features and functionalities
        - Priority level for each requirement (Critical/High/Medium/Low)
        - Acceptance criteria for key features

        ## 5. Non-Functional Requirements
        - Performance requirements
        - Security requirements
        - Scalability considerations
        - Reliability and availability
        - Compliance and regulatory requirements

        ## 6. Constraints
        - Technical constraints
        - Budget constraints
        - Timeline constraints
        - Resource constraints

        ## 7. Assumptions and Dependencies
        - Key assumptions made
        - External dependencies
        - Third-party integrations required

        ## 8. Success Criteria
        - Clear, measurable criteria for success
        - Key performance indicators (KPIs)
        - Metrics for validation

        ## 9. Out of Scope
        - Explicitly state what is NOT included in this specification
        - Implementation approaches
        - Technology choices (unless specified as constraints)

        ## 10. Appendices (as needed)
        - Glossary of terms
        - References
        - Supporting documentation
        ```

        ---

        ## Quality Criteria

        Your specification should be:
        - **Complete:** All required sections filled with meaningful content
        - **Clear:** Unambiguous language, avoid jargon unless defined
        - **Concise:** Direct and to the point, avoid verbosity
        - **Specific:** Concrete requirements rather than vague statements
        - **Measurable:** Success criteria and objectives should be quantifiable where possible

        ---

        ## Assessment and Confidence

        For each major section, provide:
        1. **Confidence Score (0.0-1.0):** How confident are you in this section's accuracy and completeness?
        2. **Uncertainty Areas:** Identify any information that is uncertain or missing
        3. **Gaps Identified:** What additional information would improve the specification?
        """,
        tools=[
            FileTools(),
            GoalToolkit(),
        ],
        db=SqliteDb(),
        pre_hooks=[
            transcription_config.pre_hook(),
            consensus_config.pre_hook(),
        ],
        post_hooks=[tts_config.post_hook()],
        markdown=True,
        debug_mode=True,
        add_datetime_to_context=True,
        add_history_to_context=True,
        num_history_messages=5,
        num_history_sessions=3,
        add_culture_to_context=True,
        compress_tool_results=True,
        update_cultural_knowledge=True,
        retries=3,
    )

    console.print()
    console.rule("[bold blue]Specification Agent: Audio Mode + Consensus[/]")
    console.print()
    console.print("[bold]Consensus Models:[/]")
    console.print("  • [cyan]SpecificationArchitect[/] - 35% weight - structure & architecture")
    console.print("  • [cyan]RequirementsAnalyst[/] - 30% weight - requirements analysis")
    console.print("  • [cyan]TechnicalReviewer[/] - 20% weight - feasibility & technical review")
    console.print("  • [cyan]QualityAssurance[/] - 15% weight - quality & clarity")
    console.print()
    console.print("[bold]Capabilities:[/]")
    console.print("  • Multi-model consensus for research and analysis")
    console.print("  • Audio input via Whisper ASR")
    console.print("  • Audio output via Piper TTS")
    console.print("  • Research tools: Tavily, FileTools, GoalTools")
    console.print("  • SQLite storage for session persistence")
    console.print("  • Goal management for tracking specification progress")
    console.print()
    console.print("[bold]Usage:[/]")
    console.print("  • Press [yellow]SPACE[/] to record your audio request")
    console.print("  • Use goal tools to track specification sections")
    console.print()
    console.print("[dim]NOTE: Press SPACE to record audio and speak your specification request.[/]")
    console.rule()
    console.print()

    while True:
        try:
            console.print()
            console.print("[bold green]🎤 Press SPACE to record audio[/]")
            console.print()

            # Run the agent with transcription hook (empty input triggers ASR)
            response = await agent.arun(
                "Create or refine the specification based on the transcribed user request.",
            )

            # Handle tool confirmations
            for requirement in response.active_requirements:
                if requirement.needs_confirmation:
                    if requirement.tool_execution and requirement.tool_execution.tool_name:
                        console.print(
                            f"Tool [bold blue]{requirement.tool_execution.tool_name}[/]"
                            f"({requirement.tool_execution.tool_args}) requires confirmation."
                        )
                    from rich.prompt import Prompt

                    message = (
                        Prompt.ask("Do you want to continue?", choices=["y", "n"], default="y")
                        .strip()
                        .lower()
                    )

                    if message == "n":
                        requirement.reject()
                    else:
                        requirement.confirm()

            # Continue run if there are requirements
            if response.requirements:
                response = await agent.acontinue_run(
                    run_id=response.run_id, requirements=response.requirements
                )

            pprint.pprint_run_response(response)

        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


if __name__ == "__main__":
    asyncio.run(main())
