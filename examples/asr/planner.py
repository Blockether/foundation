#!/usr/bin/env python3
"""
Goal Planner Agent with ASR Support

An interactive voice-only agent that helps you create, manage, and track daily goals.
Uses the GoalToolkit to manage hierarchical goals with priorities and status tracking.

Features:
1. ASR (Automatic Speech Recognition) for voice input
2. TTS (Text-to-Speech) for spoken responses
3. Goal management with priorities (P0-P3) and importance levels

Usage:
    python examples/asr/planner.py

Audio Mode:
- Press SPACE to record audio
- Speak your goal or command
- The agent will help you create, update, refine, or mark goals as done
"""

import asyncio
import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from rich.console import Console

from blockether_foundation.agents.hooks import (
    TranscriptionHooksConfig,
    TTSHooksConfig,
)
from blockether_foundation.agents.storage.in_memory import InMemoryDb
from blockether_foundation.agents.toolkits import GoalToolkit
from blockether_foundation.asr import LocalWhisperAudioTranscriber
from blockether_foundation.tts import PIPER_DEFAULT_MODEL, PiperTTS

console = Console()

# Model configuration
MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")

if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

console.print("[dim]Initializing model...[/]")

gpt_4_1 = OpenAIChat(
    id="gpt-4.1",
    base_url=MODEL_BASE_URL,
    api_key=MODEL_API_KEY,
    modalities=["text"],
)


async def main():
    """Run goal planner agent with ASR/TTS support."""

    console.print("[bold]Setting up Goal Planner Agent...[/]")

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

    # Create agent with GoalToolkit
    agent = Agent(
        model=gpt_4_1,
        name="GoalPlannerAgent",
        description=(
            "You are a personal goal planning assistant that helps users create, "
            "manage, and track their daily goals. You use voice interaction for "
            "a hands-free experience."
        ),
        instructions="""
        """,
        tools=[GoalToolkit()],
        db=InMemoryDb(),
        pre_hooks=[transcription_config.pre_hook()],
        post_hooks=[tts_config.post_hook()],
        markdown=False,
        debug_mode=True,
        add_datetime_to_context=True,
    )

    # Display welcome message
    console.print()
    console.rule("[bold blue]Goal Planner Agent: Voice-Only Goal Management[/]")
    console.print()
    console.print("[bold]Features:[/]")
    console.print("  - Voice input via Whisper ASR")
    console.print("  - Voice output via Piper TTS")
    console.print("  - Hierarchical goal management")
    console.print("  - Priority levels: P0 (critical) to P3 (low)")
    console.print()
    console.print("[bold]Commands you can say:[/]")
    console.print("  - 'Add a goal to [task description]'")
    console.print("  - 'What are my goals?' or 'List my goals'")
    console.print("  - 'I finished [goal]' or 'Mark [goal] as done'")
    console.print("  - 'Skip [goal]' or 'I can't do [goal]'")
    console.print("  - 'Change [goal] priority to high'")
    console.print("  - 'What's my current goal?'")
    console.print()
    console.print("[dim]Press SPACE to record audio. Ctrl+C to quit.[/]")
    console.rule()
    console.print()

    # ASR-only loop
    while True:
        try:
            console.print("[bold green]Press SPACE to record audio[/]")
            response = await agent.arun("")
            console.print(f"\n[bold]Assistant:[/] {response.content}\n")
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


if __name__ == "__main__":
    asyncio.run(main())
