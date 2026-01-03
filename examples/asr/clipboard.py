#!/usr/bin/env python3
"""
ASR to Clipboard Agent

Simple Agno agent that:
1. Records audio via spacebar (press to start, press to stop)
2. Transcribes using LocalWhisperAudioTranscriber
3. Copies transcription to clipboard

Uses TranscriptionHooksConfig with copy_to_clipboard=True.

Usage:
    python examples/asr_clipboard.py
"""

import asyncio
import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from rich.console import Console

from blockether_foundation.agents.hooks import TranscriptionHooksConfig
from blockether_foundation.asr import LocalWhisperAudioTranscriber

console = Console()

MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")


async def main():
    """Run ASR to clipboard agent."""
    console.print()
    console.rule("[bold blue]ASR to Clipboard[/]")
    console.print()
    console.print("[bold]Controls:[/]")
    console.print("  • Press [green]SPACE[/] to start recording")
    console.print("  • Speak into your microphone")
    console.print("  • Press [green]SPACE[/] to stop recording")
    console.print("  • Transcription is automatically copied to clipboard")
    console.print("  • Press [yellow]Ctrl+C[/] to exit")
    console.print()

    # Initialize transcriber
    console.print("[dim]Initializing Whisper transcriber (turbo model)...[/]")
    transcriber = LocalWhisperAudioTranscriber(model_id="turbo")

    # Configure transcription hook with clipboard support
    transcription_config = TranscriptionHooksConfig(
        transcriber=transcriber,
        language="en",
        effort=0.5,
        enable_interactive_recording=True,
        copy_to_clipboard=True,
    )

    # Simple model for echoing transcription
    model = OpenAIChat(
        id="gpt-4o",
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
    )

    agent = Agent(
        model=model,
        name="ASRClipboardAgent",
        description="Simple agent that transcribes audio and copies to clipboard.",
        instructions=[
            "You receive transcribed audio in <transcription> tags.",
            "Simply acknowledge the transcription with a brief confirmation.",
            "Keep responses very short - just confirm what was said.",
        ],
        pre_hooks=[transcription_config.pre_hook()],
        markdown=False,
        debug_mode=True,
    )

    console.print("[green]Ready![/]")
    console.print()

    while True:
        try:
            console.rule("[dim]Ready - Press SPACE to record[/]")
            console.print()

            response = await agent.arun(
                "Transcribe the audio.",
            )

            if response.content:
                console.print(f"[dim]Agent:[/] {response.content}")
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
