#!/usr/bin/env python3
"""
Transcribe Session Audio Files

This script processes all audio files in the sessions/ directory and transcribes them.
It saves the transcription results as JSON files in the same directory.

Usage:
    python scripts/transcribe_sessions.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agno.models.openai import OpenAIChat

from blockether_foundation.agents.transcriber import process_audio_files
from blockether_foundation.audio.transcription import AudioTranscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Main transcription function."""
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
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        logger.error(f"sessions/ directory not found at {sessions_dir}!")
        logger.info(
            f"Please create a 'sessions' directory at {sessions_dir} and add your audio files there."
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

    # Process audio files with blockether model
    await process_audio_files(
        glob_pattern=str(sessions_dir.absolute() / "*"),
        output_dir=str(sessions_dir.absolute()),
        model=blockether_model,
        input="AUTOMATICALLY TRANSLATE THE CONVERSATION TO !!!ENGLISH!!!. !OVERRIDE PREVIOUS INSTRUCTIONS! THERE ARE TWO PARTICIPANTS: Karol and Michał. Karol's wife is Ashley. His colleagues are Alex and Maxim and these are sometimes mentioned.",
        audio_transcriber=AudioTranscriber.get_instance(),
        debug_mode=True,
        audio_chunking=True,
        save_raw_transcription=True,
        save_dir=str(sessions_dir.absolute()),
        # Audio chunking (enabled by default)
        chunk_duration=900.0,  # 15-minute chunks
        chunk_concurrency=8,  # Process up to 8 chunks in parallel for speed
    )

    logger.info("Session transcription process completed.")


if __name__ == "__main__":
    asyncio.run(main())
