#!/usr/bin/env python3
"""
Transcribe Audio Files

This script processes all audio files in the input/ directory and transcribes them.
It saves the transcription results as JSON files in the same directory.

Usage:
    python examples/transcribe_audios.py --participants "Alice and Bob" --context "Alice's friend Charlie is sometimes mentioned"
    python examples/transcribe_audios.py --input-dir my_audios/
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
    prompt = f"Translate the conversation into fluent, natural English. Participants: {participants}. Preserve speaker labels, turn boundaries, and timestamps. Keep names and proper nouns unchanged. Provide a faithful translation (not a summary); if a phrase is ambiguous, include a short bracketed clarification. Output plain text transcript (or JSON if requested)."
    if context:
        prompt += f" {context}"
    return prompt


async def main():
    """Main transcription function."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files from input directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --participants "Alice and Bob"
    %(prog)s --participants "Alice and Bob" --context "Their friend Charlie is sometimes mentioned"
    %(prog)s --input-dir my_audios/ --participants "Speaker1 and Speaker2"
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Directory containing audio files (default: input)",
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

    logger.info("Starting audio transcription process...")

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

    # Ensure input directory exists
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found at {input_dir}!")
        logger.info(f"Please create a '{args.input_dir}' directory and add your audio files there.")
        return

    # Look for audio files
    audio_files = (
        list(input_dir.glob("*.m4a"))
        + list(input_dir.glob("*.wav"))
        + list(input_dir.glob("*.mp3"))
    )

    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}/")
        logger.info("Supported formats: .m4a, .wav, .mp3")
        return

    logger.info(f"Found {len(audio_files)} audio file(s) to transcribe")

    # Build the transcription prompt
    transcription_prompt = build_transcription_prompt(args.participants, args.context)
    logger.info(f"Transcription prompt: {transcription_prompt}")

    # Process audio files with blockether model
    await process_audio_files(
        glob_pattern=str(input_dir.absolute() / "*"),
        output_dir=str(input_dir.absolute()),
        model=blockether_model,
        input=transcription_prompt,
        audio_transcriber=LocalWhisperAudioTranscriber(),
        debug_mode=True,
        audio_chunking=True,
        save_raw_transcription=True,
        save_dir=str(input_dir.absolute()),
    )

    logger.info("Audio transcription process completed.")


if __name__ == "__main__":
    asyncio.run(main())
