#!/usr/bin/env python3
"""
Transcribe Session Audio Files

This script processes all audio files in the sessions/ directory and transcribes them.
It saves the transcription results as JSON files in the same directory.

Usage:
    python examples/transcribe_sessions.py --participants "Alice and Bob" --context "Alice's friend Charlie is sometimes mentioned"
    python examples/transcribe_sessions.py --sessions-dir my_sessions/
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agno.models.openai import OpenAIChat

from blockether_foundation.agents.transcriber import process_audio_files
from blockether_foundation.asr import LocalWhisperAudioTranscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_transcription_prompt(participants: str, context: str | None = None) -> str:
    """Build the transcription prompt with participant names and optional context.

    Args:
        participants: Description of participants (e.g., "Alice and Bob")
        context: Optional additional context (e.g., "Alice's friend Charlie is sometimes mentioned")

    Returns:
        Formatted transcription prompt
    """
    prompt = f"AUTOMATICALLY TRANSLATE THE CONVERSATION TO !!!ENGLISH!!!. !OVERRIDE PREVIOUS INSTRUCTIONS! THERE ARE PARTICIPANTS: {participants}."
    if context:
        prompt += f" {context}"
    return prompt


async def main():
    """Main transcription function."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files from sessions directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --participants "Alice and Bob"
    %(prog)s --participants "Alice and Bob" --context "Their friend Charlie is sometimes mentioned"
    %(prog)s --sessions-dir my_sessions/ --participants "Speaker1 and Speaker2"
        """,
    )

    parser.add_argument(
        "--sessions-dir",
        type=str,
        default="sessions",
        help="Directory containing audio files (default: sessions)",
    )

    parser.add_argument(
        "--participants",
        type=str,
        default="Speaker1 and Speaker2",
        help="Description of participants (e.g., 'Alice and Bob') (default: 'Speaker1 and Speaker2')",
    )

    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional additional context about people mentioned (e.g., 'Their friend Charlie is sometimes mentioned')",
    )

    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=900.0,
        help="Duration of audio chunks in seconds (default: 900 = 15 minutes)",
    )

    parser.add_argument(
        "--chunk-concurrency",
        type=int,
        default=8,
        help="Number of chunks to process in parallel (default: 8)",
    )

    args = parser.parse_args()

    logger.info("Starting session transcription process...")

    # Check for required environment variables
    model_base_url = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
    if not model_base_url:
        logger.error("BLOCKETHER_LLM_API_BASE_URL environment variable is not set!")
        logger.info("Please set BLOCKETHER_LLM_API_BASE_URL to use the transcription service.")
        return

    # Initialize blockether model
    blockether_model = OpenAIChat(
        id="gpt-4.1",
        base_url=model_base_url,
        api_key=os.getenv("BLOCKETHER_LLM_API_KEY"),
    )
    logger.info(f"Using model: gpt-4.1 at {model_base_url}")

    # Ensure sessions directory exists
    sessions_dir = Path(args.sessions_dir)
    if not sessions_dir.exists():
        logger.error(f"Sessions directory not found at {sessions_dir}!")
        logger.info(
            f"Please create a '{args.sessions_dir}' directory and add your audio files there."
        )
        return

    # Look for audio files
    audio_files = (
        list(sessions_dir.glob("*.m4a"))
        + list(sessions_dir.glob("*.wav"))
        + list(sessions_dir.glob("*.mp3"))
    )

    if not audio_files:
        logger.warning(f"No audio files found in {sessions_dir}/")
        logger.info("Supported formats: .m4a, .wav, .mp3")
        return

    logger.info(f"Found {len(audio_files)} audio file(s) to transcribe")

    # Build the transcription prompt
    transcription_prompt = build_transcription_prompt(args.participants, args.context)
    logger.info(f"Transcription prompt: {transcription_prompt}")

    # Process audio files with blockether model
    await process_audio_files(
        glob_pattern=str(sessions_dir.absolute() / "*"),
        output_dir=str(sessions_dir.absolute()),
        model=blockether_model,
        input=transcription_prompt,
        audio_transcriber=LocalWhisperAudioTranscriber(),
        debug_mode=True,
        audio_chunking=True,
        save_raw_transcription=True,
        save_dir=str(sessions_dir.absolute()),
    )

    logger.info("Session transcription process completed.")


if __name__ == "__main__":
    asyncio.run(main())
