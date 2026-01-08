#!/usr/bin/env python3
"""
Audio to Graph Extraction

This script processes a single audio file and extracts a knowledge graph from it.
The transcription is done in memory and not saved to disk.

Usage:
    python examples/graph/audio_to_graph.py sample.m4a --participants "Alice and Bob"
    python examples/graph/audio_to_graph.py sample.m4a --participants "Alice and Bob" --context "Alice's friend Charlie is sometimes mentioned"
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from blockether_foundation.agents.hooks.graph import GraphHooksConfig
from blockether_foundation.agents.transcriber import TranscriptionResult, process_audio_file
from blockether_foundation.asr import TEN_MINUTES, LocalWhisperAudioTranscriber

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
    prompt = f"Transcribe the conversation. Participants: {participants}. Preserve speaker labels, turn boundaries, and timestamps. Keep names and proper nouns unchanged."
    if context:
        prompt += f" {context}"
    return prompt


async def main():
    """Main audio-to-graph function."""
    parser = argparse.ArgumentParser(
        description="Extract knowledge graph from audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s sample.m4a --participants "Alice and Bob"
    %(prog)s sample.m4a --participants "Alice and Bob" --context "Their friend Charlie is sometimes mentioned"
        """,
    )

    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to process",
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
        default=TEN_MINUTES,
        help=f"Duration of audio chunks in seconds (default: {TEN_MINUTES} = 10 minutes)",
    )

    parser.add_argument(
        "--output-graph",
        type=str,
        default=None,
        help="Path to save the extracted graph (optional)",
    )

    parser.add_argument(
        "--output-transcription",
        type=str,
        default=None,
        help="Path to save/load the transcription text (default: <audio_stem>.transcription.txt)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-transcription even if transcription file exists",
    )

    args = parser.parse_args()

    # Determine default transcription file path based on audio filename
    audio_file_path = Path(args.audio_file)
    default_transcription_path = audio_file_path.with_suffix(".transcription.txt")
    transcription_path = (
        Path(args.output_transcription) if args.output_transcription else default_transcription_path
    )

    logger.info("Starting audio to graph extraction...")

    # Check for required environment variables
    model_base_url = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
    if not model_base_url:
        logger.error("BLOCKETHER_LLM_API_BASE_URL environment variable is not set!")
        logger.info("Please set BLOCKETHER_LLM_API_BASE_URL to use the transcription service.")
        return

    # Initialize blockether model
    blockether_model = OpenAIChat(
        id="gpt-5-mini",
        base_url=model_base_url,
        api_key=os.getenv("BLOCKETHER_LLM_API_KEY"),
        timeout=60000,
    )
    logger.info(f"Using model: gpt-5-mini at {model_base_url}")

    # Check audio file exists
    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        logger.error(f"Audio file not found at {audio_file}!")
        return

    logger.info(f"Processing audio file: {audio_file}")

    # Set up graph hooks
    graph_config = GraphHooksConfig(
        file_path=args.output_graph,
        agentic_ingestion=True,
        async_hooks=True,
    )
    graph = graph_config.graph

    # Build the transcription prompt
    transcription_prompt = build_transcription_prompt(args.participants, args.context)
    logger.info(f"Transcription prompt: {transcription_prompt}")

    # Check if transcription file already exists
    transcription: TranscriptionResult | None = None
    if transcription_path.exists() and not args.force:
        logger.info(f"Found existing transcription at {transcription_path}, loading...")
        try:
            with open(transcription_path) as f:
                transcription_text = f.read()
            transcription = TranscriptionResult.from_text(transcription_text)
            logger.info(f"Loaded transcription: {len(transcription.conversation)} dialogue lines")
        except Exception as e:
            logger.warning(f"Failed to load existing transcription: {e}")
            logger.info("Will re-transcribe the audio file")
            transcription = None

    # Transcribe if no existing transcription or --force flag
    if transcription is None:
        logger.info("Transcribing audio file...")
        transcription = await process_audio_file(
            file_path=str(audio_file),
            chunk_duration=args.chunk_duration,
            overlap=60.0,
            output_dir="/tmp",  # Temporary directory for intermediate files
            input=transcription_prompt,
            model=blockether_model,
            audio_transcriber=LocalWhisperAudioTranscriber("small"),
            debug_mode=True,
        )

        if not transcription:
            logger.error("Transcription failed!")
            return

        logger.info(f"Transcription completed: {len(transcription.conversation)} dialogue lines")

        # Save transcription to file
        transcription_text = transcription.to_text()
        with open(transcription_path, "w") as f:
            f.write(transcription_text)
        logger.info(f"Transcription saved to {transcription_path}")
    else:
        logger.info("Using existing transcription (skipped audio processing)")

    # Format transcription as text for graph extraction
    transcription_text = transcription.to_text()
    logger.info("Transcription text formatted for graph extraction")

    # Create agent with graph post-hook for automatic extraction
    graph_agent = Agent(
        id="graph-extraction-agent",
        name="Graph Extraction Agent",
        instructions=[
            "Extract entities and relationships from the provided transcription text.",
            "Focus on people, organizations, locations, dates, events, and key facts.",
            "Create relationships between entities based on what is explicitly stated.",
        ],
        post_hooks=[graph_config.post_hook()],
        model=blockether_model,
        debug_mode=True,
    )

    # Run graph extraction via the agent (post-hook will auto-populate graph)
    logger.info("Extracting knowledge graph from transcription...")
    response = await graph_agent.arun(transcription_text)

    if not response:
        logger.error("Graph extraction failed!")
        return

    # Save graph if output path provided
    if args.output_graph:
        graph_config.save_graph()
        logger.info(f"Graph saved to {args.output_graph}")

    # Print graph summary
    logger.info("=" * 60)
    logger.info("GRAPH SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total entities: {graph.entity_count}")
    logger.info(f"Total relationships: {graph.relationship_count}")

    # Print all entities
    logger.info("")
    logger.info("ENTITIES:")
    for entity in graph.index.entity_by_id.values():
        logger.info(f"  - [{entity.type}] {entity.name}: {entity.content}")

    # Print all relationships
    logger.info("")
    logger.info("RELATIONSHIPS:")
    for rel in graph.index.relationship_by_id.values():
        logger.info(f"  - {rel.source} --[{rel.type}]--> {rel.target}")

    logger.info("")
    logger.info("Audio to graph extraction completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
