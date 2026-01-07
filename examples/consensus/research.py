#!/usr/bin/env python3
"""
Research Agent with Consensus Pre-Hook

This example demonstrates an agent that:
1. Uses multi-model consensus for research and initial analysis via claude CLI
2. Transcribes incoming audio using LocalWhisperAudioTranscriber
3. Generates audio responses using PiperTTS
4. Auto-plays synthesized audio using platform-specific audio commands

The consensus pre-hook runs the claude CLI to perform research and analysis
BEFORE the main agent processes the request. This is a RESEARCH-ONLY mode -
no code changes are made unless explicitly requested by the user.

Usage:
    python examples/researcher.py
"""

import asyncio
import os

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.tools.arxiv import ArxivTools
from agno.tools.file import FileTools
from agno.tools.tavily import TavilyTools
from agno.utils import pprint
from rich.console import Console
from rich.prompt import Prompt

from blockether_foundation.agents.hooks import (
    ConsensusHooksConfig,
    JudgeCriteria,
    ModelConfig,
    TranscriptionHooksConfig,
    TTSHooksConfig,
)
from blockether_foundation.agents.models.zai import Zhipu
from blockether_foundation.agents.toolkits import GoalToolkit
from blockether_foundation.agents.toolkits.claude import ClaudeToolkit
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

glm_4_7 = Zhipu(
    id="glm-4.7",
    enable_coding_plan=True,
    enable_thinking=True,
)


async def main():
    """Run research agent with consensus pre-hook, ASR and TTS."""

    console.print("[bold]Setting up consensus, ASR and TTS hooks...[/]")

    # Setup ASR
    transcriber = LocalWhisperAudioTranscriber(model_id="turbo")

    # Setup TTS
    console.print(f"[dim]Downloading Piper TTS model: {PIPER_DEFAULT_MODEL}[/]")
    model_path = PiperTTS.pre_download(
        model_name=PIPER_DEFAULT_MODEL,
        download_root="./models/piper",
    )
    synthesizer = PiperTTS(model_path=model_path)

    # Configure consensus with opencode research tool
    # The consensus uses multiple model perspectives to research the task
    consensus_config = ConsensusHooksConfig(
        output_directory="./consensus_reports",
        auto_save_html=True,
        models=[
            ModelConfig(
                model=glm_4_7,
                name="DeepReasoner",
                importance=0.4,
                perspective=(
                    "Focus on deep reasoning and systematic analysis. Think through the "
                    "problem step by step, consider edge cases, and provide thorough "
                    "analysis with coding plan. Use the claude research tool to gather "
                    "detailed information about the codebase and problem space."
                ),
                tools=[ClaudeToolkit()],
            ),
            ModelConfig(
                model=glm_4_7,
                name="QuickPlanner",
                importance=0.3,
                perspective=(
                    "Focus on quick assessment and practical implementation planning. "
                    "Identify the most direct path to solution, flag potential blockers, "
                    "and outline clear implementation steps. Be concise and actionable."
                ),
            ),
            ModelConfig(
                model=glm_4_7,
                name="CodeSpecialist",
                importance=0.3,
                perspective=(
                    "Focus on code quality, best practices, and implementation details. "
                    "Analyze the codebase structure, identify patterns, and provide "
                    "concrete code solutions. Use the claude research tool to examine "
                    "similar implementations in the codebase."
                ),
            ),
        ],
        judge_criteria=[
            JudgeCriteria(
                name="Research Completeness",
                description="Has the research covered all relevant aspects of the query?",
                weight=1.0,
                threshold=0.6,
            ),
            JudgeCriteria(
                name="Actionability",
                description="Are the findings actionable and clearly presented?",
                weight=0.8,
                threshold=0.6,
            ),
            JudgeCriteria(
                name="Accuracy",
                description="Are the findings accurate and well-supported?",
                weight=1.0,
                threshold=0.7,
            ),
            JudgeCriteria(
                name="DRYness",
                description="Is the research free of unnecessary repetition across models, providing unique insights? Are the proposed actions non-redundant?",
                weight=1.0,
                threshold=0.7,
            ),
        ],
        triage_model=gpt_4o,
        max_refinement_iterations=2,
        judge_threshold=0.65,
    )

    transcription_config = TranscriptionHooksConfig(
        transcriber=transcriber,
        language="en",
        effort=0.5,
        save_transcriptions=False,
        enable_interactive_recording=True,
    )

    tts_config = TTSHooksConfig(
        synthesizer=synthesizer, save_to_file=False, auto_speak=True, interruptible=True
    )

    agent = Agent(
        model=glm_4_7,
        name="ResearchAgent",
        description=(
            "You are a high-quality research assistant that helps with researching "
            "and planning programming tasks. You receive pre-analyzed consensus from "
            "multiple AI models (glm-4.7, gpt-4.1, gpt-5-mini) before responding."
        ),
        instructions=[
            "Your current goal: {current_goal_id}",
            "",
            "You will receive a transcribed user message from audio input. The message will be provided in <transcription> tags. "
            "Please fulfill the user's request based on transcription. "
            "You might be also asked to save the current context which you should do using FileTools under [current_date]_researcher_context.md filename in markdown format.",
            "",
            "GOAL MANAGEMENT SYSTEM:",
            "- Use create_goal to create new goals with priorities (P0=critical, P1=high, P2=medium, P3=low) and importance scores (0.0-1.0)",
            "- You can create hierarchical goals with subgoals using the parent_goal_id parameter",
            "- Use list_goals to see all goals with their status, priority, and importance",
            "- Use finish_goal to mark the current goal as completed and move to the next pending goal",
            "- Use skip_goal to skip a goal without completing it, moving to the next pending goal",
            "- Use refine_goal to update goal attributes (title, description, priority, importance)",
            "- Use jump_to_goal to switch to a specific goal and make it the current goal",
            "- Use set_goal_priority to change a goal's priority level",
            "- Use get_current_goal to see details of the current active goal",
            "",
            "IMPORTANT - RESEARCH-ONLY MODE BY DEFAULT:",
            "- A multi-model consensus (glm-4.7, gpt-4.1, gpt-5-mini) has already analyzed your request",
            "- The consensus provides research findings and implementation proposals",
            "- NO CODE CHANGES are made automatically - this is research only by default",
            "- It is UP TO THE USER to decide whether to proceed with implementation",
            "- If the user wants to implement changes, they must EXPLICITLY request it",
            "",
            "REQUESTS THAT DO NOT REQUIRE RESEARCH:",
            "- Simple questions or clarifications about previous findings",
            "- Casual conversation or greetings",
            "- Requests to summarize or explain existing consensus results",
            "- Follow-up questions that don't need new analysis",
            "- User confirmations like 'yes', 'proceed', 'go ahead'",
            "For these, respond directly without invoking research tools.",
            "THE USER REQUEST IS GIVEN AS A TRANSCRIPTION and WILL APPEAR IN <transcription> TAGS!",
            "IMPORTANT - TTS-FRIENDLY OUTPUT:",
            "- Your responses will be converted to speech using text-to-speech (PiperTTS)",
            "- Format your output as NATURAL SPOKEN LANGUAGE, not written documentation",
            "- AVOID markdown formatting (bold, italic, code blocks)",
            "- Use conversational language with proper punctuation and pauses",
            "- For bullet points: Speak them naturally as separate sentences or items",
            "- For numbered lists: Read them sequentially as spoken numbers",
            "- For code or technical terms: Explain them in simple, natural language",
            "- Add pauses (periods) where natural speech would pause",
            "- Keep sentences relatively short and clear for better speech synthesis",
            "",
            "Your role:",
            "1. Present consensus research findings clearly in spoken language",
            "2. Highlight key insights and recommendations from all three models naturally",
            "3. Ask the user if they want to proceed with any proposed changes",
            "4. Only use the run_claude tool if the user EXPLICITLY confirms implementation",
            "",
            "IMPLEMENTATION TOOL:",
            "- You have access to the run_claude tool for executing actual code changes",
            "- NEVER use this tool unless the user explicitly says 'yes', 'proceed', 'implement', etc.",
            "- Always confirm before executing any implementation",
        ],
        tools=[
            ClaudeToolkit(),
            GoalToolkit(),
            FileTools(
                enable_list_files=False,
                enable_read_file=False,
                enable_delete_file=False,
                enable_replace_file_chunk=False,
                enable_read_file_chunk=False,
            ),
            TavilyTools(),
            ArxivTools(),
        ],
        db=SqliteDb(),
        pre_hooks=[
            transcription_config.pre_hook(),
            consensus_config.pre_hook(),
        ],
        post_hooks=[tts_config.post_hook()],
        markdown=False,
        debug_mode=True,
        add_datetime_to_context=True,
        add_history_to_context=True,
        num_history_messages=3,
        num_history_sessions=2,
        add_culture_to_context=True,
        compress_tool_results=True,
        update_cultural_knowledge=True,
        retries=3,
    )

    console.print()
    console.rule("[bold blue]Research Agent: Consensus + ASR + TTS[/]")
    console.print()
    console.print("[bold]Consensus Models:[/]")
    console.print("  • [cyan]glm-4.7[/] (DeepReasoner) - 40% weight - reasoning & coding plan")
    console.print("  • [cyan]gpt-4.1[/] (ResearchAnalyst) - 25% weight - thorough research")
    console.print("  • [cyan]devstral-2512[/] (QuickPlanner) - 20% weight - quick assessment")
    console.print("  • [cyan]kat-coder-pro[/] (CodeSpecialist) - 15% weight - code quality")
    console.print()
    console.print("[bold]Capabilities:[/]")
    console.print(
        "  • [green]Smart triage[/]: Skips consensus for simple requests (greetings, confirmations)"
    )
    console.print("  • Multi-model consensus via [dim]claude --no-session-persistence -p[/]")
    console.print("  • [yellow]RESEARCH-ONLY[/] by default: No changes without user approval")
    console.print("  • Implementation tool available when user explicitly approves")
    console.print("  • Record audio interactively with SPACEBAR control")
    console.print("  • Transcribes incoming audio (Whisper)")
    console.print("  • Generates spoken responses (Piper TTS)")
    console.print("  • Auto-plays responses (afplay/aplay)")
    console.print()
    console.print("[dim]NOTE: Triage uses gpt-5-mini to quickly decide if consensus is needed.[/]")
    console.print("[dim]      Complex research tasks trigger full multi-model consensus.[/]")
    console.rule()

    while True:
        try:
            console.print()
            console.print("[bold green]🎤 Starting interactive audio recording...[/]")
            console.print("[dim]Press SPACE to start recording[/]")
            console.print("[dim]Speak, then press SPACE to stop[/]")
            console.print()

            response = await agent.arun(
                "The above message in <transcription> tags is the user's request which you must respond to.",
            )

            for requirement in response.active_requirements:
                if requirement.needs_confirmation:
                    # Ask for confirmation
                    if requirement.tool_execution and requirement.tool_execution.tool_name:
                        console.print(
                            f"Tool [bold blue]{requirement.tool_execution.tool_name}[/]"
                            f"({requirement.tool_execution.tool_args}) requires confirmation."
                        )
                    message = (
                        Prompt.ask("Do you want to continue?", choices=["y", "n"], default="y")
                        .strip()
                        .lower()
                    )

                    if message == "n":
                        requirement.reject()
                    else:
                        requirement.confirm()

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
