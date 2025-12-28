#!/usr/bin/env python3
"""
Relationship Knowledge Extractor Agent

This script creates and runs an Agno agent that extracts meaningful knowledge about
Partner1 and Partner2 using Schema Therapy principles. The agent can operate in two modes:

1. BATCH MODE: Process conversation logs in batches and builds a knowledge graph that is saved to file.
2. CLI MODE: Interactive conversation mode where knowledge is extracted from the conversation in real-time.

The agent is designed with deep understanding of Schema Therapy to extract psychological
and emotional patterns, needs, and relationship dynamics for BOTH partners.

Usage:
    # Interactive CLI mode (new!)
    python graph_knowledge.py --mode cli [--partner1 NAME] [--partner2 NAME] [--output-file PATH]

    # Batch processing mode (default)
    python graph_knowledge.py [--input-file PATH] [--output-file PATH] [--partner1 NAME] [--partner2 NAME] [--lines-per-batch N]

Examples:
    # Interactive CLI conversation
    python graph_knowledge.py --mode cli --partner1 Alice --partner2 Bob

    # Process with default settings (requires partner names)
    python graph_knowledge.py --partner1 Alice --partner2 Bob

    # Custom input and output files with partner names
    python graph_knowledge.py --mode batch --input-file conversations.txt --output-file relationship_graph.json --partner1 Alice --partner2 Bob

    # Process in larger batches
    python graph_knowledge.py --partner1 Partner1 --partner2 Partner2 --lines-per-batch 200

Important:
    - In CLI mode, --lines-per-batch is prohibited and will cause an error
    - In CLI mode, --input-file is ignored
    - Default mode is batch for backward compatibility
"""

import argparse
import asyncio
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from agno.agent.agent import Agent
from agno.models.openai import OpenAIChat
from agno.utils.log import configure_agno_logging

from blockether_foundation.agents.hooks.graph import GraphHookIterativeConfig, GraphHooksConfig

# Initialize simple logging system
log_file = f"knowledge_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Set up logging to both console and file
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
# )

logger = logging.getLogger(__name__)

MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")

# Validate required environment variables
if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

print(
    f"Initializing blockether_model... model_base_url={MODEL_BASE_URL}, model_api_key={MODEL_API_KEY}"
)

blockether_model = OpenAIChat(
    id="gpt-5-mini", base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY, modalities=["text"]
)

# =============================================================================
# SCHEMA THERAPY KNOWLEDGE EXTRACTOR PROMPT TEMPLATE
# =============================================================================

RELATIONSHIP_KNOWLEDGE_EXTRACTOR_PROMPT_TEMPLATE = """
You are Dr. Schema - a Schema Therapy expert extracting deep psychological knowledge about {partner1} and {partner2}.

  CORE REQUIREMENTS:
  1. WHAT HAPPENED: Exact words, behaviors, sequence, emotions, timing
  2. WHY IT HAPPENED: Schema activations, childhood origins, unmet needs, coping styles
  3. CONTEXT: Relationship dynamics, external stressors, patterns, developmental stage
  4. SCHEMA MODES: Which mode speaks for each partner (Vulnerable Child, Angry Child, Healthy Adult, Detached Protector, etc.)

  KEY SCHEMAS (18 total across 5 domains):
  DISCONNECTION: Abandonment, Mistrust/Abuse, Emotional Deprivation, Defectiveness/Shame, Social Isolation
  IMPAIRED AUTONOMY: Dependence, Vulnerability, Enmeshment, Failure
  IMPAIRED LIMITS: Entitlement, Insufficient Self-Control
  OTHER-DIRECTED: Subjugation, Self-Sacrifice, Approval-Seeking
  OVERVIGILANCE: Negativity, Emotional Inhibition, Unrelenting Standards, Punitiveness

  COPING STYLES: Surrender, Avoidance, Overcompensation
  CHILD MODES: Vulnerable, Angry, Impulsive, Contented
  COPING MODES: Compliant Surrenderer, Detached Protector, Overcompensator
  PARENT MODES: Punitive, Demanding

  EXTRACTION FOCUS:
  - Schema interactions and chemistry between partners
  - Attachment styles (Secure, Anxious, Avoidant, Disorganized)
  - Healing moments and growth opportunities
  - Conflict patterns and repair attempts
  - Historical patterns and childhood origins

  Create rich, interconnected entities with quotes, emotional intensity, timing. Each situation should have the schema, mode, pattern and trigger attached if applicable! Remember to always extract the knowledge about all of the mentioned people in the conversation.
"""


def parse_conversation_lines(lines: list[str]) -> list[tuple[str | None, str | None, str]]:
    """Parse conversation lines into (timestamp, sender, message) tuples."""
    conversations: list[tuple[str | None, str | None, str]] = []
    current_message: list[str] = []
    current_sender: str | None = None
    current_timestamp: str | None = None

    line_pattern = r"^\[(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}:\d{2})\] (.+?): (.*)$"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(line_pattern, line)
        if match:
            # Save previous message if exists
            if current_sender and current_message:
                conversations.append((current_timestamp, current_sender, " ".join(current_message)))

            # Start new message
            current_timestamp = match.group(1)
            current_sender = match.group(2)
            current_message = [match.group(3)]
        else:
            # Continuation of current message
            if current_message:
                current_message.append(line)

    # Save last message
    if current_sender and current_message:
        conversations.append((current_timestamp, current_sender, " ".join(current_message)))

    return conversations


def format_conversations(
    conversations: list[tuple[str | None, str | None, str]],
) -> str | None:
    """Format conversation lines for knowledge extraction."""
    if not conversations:
        return None

    # Include all messages - we want knowledge about both partners
    all_messages = [
        f"[{timestamp}] {sender}: {message}" for timestamp, sender, message in conversations
    ]
    return "\n".join(all_messages) if all_messages else None


def create_knowledge_agent(
    output_file: Path, partner1: str, partner2: str, logger: logging.Logger | None = None
) -> Agent:
    """Create the relationship knowledge extraction agent with Schema Therapy integration.

    Args:
        output_file: Path to save the knowledge graph
        partner1: Name of the first partner
        partner2: Name of the second partner
        logger: Optional logger for detailed tracking
    """
    # Configure graph hooks with file persistence and iterative processing
    iterative_config = GraphHookIterativeConfig(
        enabled=True,
        max_iterations=2,
        quality_threshold=0.75,
        enable_gap_analysis=True,
        enable_entity_expansion=True,
        enable_query_refinement=True,
        bfs_expansion_depth=2,
        max_entities_to_expand=5,
    )

    # GraphHooksConfig will handle graph creation/loading automatically
    # It will load from output_file if it exists, or create a new graph if it doesn't
    graph_config = GraphHooksConfig(
        file_path=str(output_file),  # GraphHooksConfig handles creation/loading
        agentic_search=True,
        agentic_ingestion=True,
        async_hooks=True,
        iterative_config=iterative_config,
    )

    # Format the prompt template with partner names
    formatted_prompt = RELATIONSHIP_KNOWLEDGE_EXTRACTOR_PROMPT_TEMPLATE.format(
        partner1=partner1,
        partner2=partner2,
    )

    agent = Agent(
        name="Dr. Schema - Relationship Knowledge Extractor",
        model=blockether_model,
        description=f"World-class Schema Therapy expert extracting deep psychological knowledge about {partner1} and {partner2}",
        instructions=formatted_prompt,
        pre_hooks=[graph_config.pre_hook()],
        post_hooks=[graph_config.post_hook()],
        debug_mode=True,
    )

    return agent


async def process_conversations_interactive(
    output_file: Path,
    partner1: str,
    partner2: str,
) -> None:
    """Run interactive CLI conversation with graph knowledge extraction.

    Args:
        output_file: Path to save the knowledge graph
        partner1: Name of the first partner
        partner2: Name of the second partner
    """
    # Create the relationship knowledge agent without logger
    agent = create_knowledge_agent(output_file, partner1, partner2)

    # Display welcome message
    print("\n" + "=" * 80)
    print("Relationship Knowledge Extractor - Interactive CLI Mode")
    print("=" * 80)
    print(f"\nExtracting psychological knowledge about {partner1} and {partner2}")
    print("Knowledge will be automatically extracted from our conversation!")
    print(f"\nThe graph will be saved to: {output_file}")
    print("\nYou can now have a regular conversation with Dr. Schema.")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    print("=" * 80 + "\n")

    # Start interactive CLI
    await agent.acli_app()


async def process_conversations_in_batches(
    input_file: Path,
    output_file: Path,
    partner1: str,
    partner2: str,
    lines_per_batch: int = 100,
) -> None:
    """Process conversations in batches and extract knowledge with comprehensive logging.

    Args:
        input_file: Path to the conversation log file
        output_file: Path to save the knowledge graph
        partner1: Name of the first partner
        partner2: Name of the second partner
        lines_per_batch: Number of lines to process per batch
    """
    logger.info(f"Starting knowledge extraction for {partner1} and {partner2}")
    logger.info(f"Graph will be saved to: {output_file}")
    logger.info(f"Detailed log file: {log_file}")

    # Create the relationship knowledge agent with logger
    agent = create_knowledge_agent(output_file, partner1, partner2, logger)

    # Read all lines from input file
    with open(input_file, encoding="utf-8") as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)
    processed_lines = 0
    batch_num = 1

    logger.info(f"Processing {total_lines} lines in batches of {lines_per_batch}...")
    print(f"Processing {total_lines} lines in batches of {lines_per_batch}...")
    print("Enhanced extraction logging enabled - capturing comprehensive contextual information")

    # Process in batches
    while processed_lines < total_lines:
        # Get batch of lines
        start_idx = processed_lines
        end_idx = min(processed_lines + lines_per_batch, total_lines)
        batch_lines = all_lines[start_idx:end_idx]

        logger.info(f"\n=== Batch {batch_num}: Lines {start_idx + 1}-{end_idx} ===")
        print(f"\n=== Batch {batch_num}: Lines {start_idx + 1}-{end_idx} ===")

        # Parse conversations
        conversations = parse_conversation_lines(batch_lines)

        if conversations:
            # Format conversations for extraction
            focused_context = format_conversations(conversations)

            if focused_context:
                # Create enhanced extraction prompt with comprehensive context requirements
                extraction_prompt = f"""
    Extract COMPREHENSIVE psychological, emotional, situational knowledge about BOTH {partner1} and {partner2} from this conversation segment denoted as <conversation_segment>with the full context.
    EXTRACT SPECIFICALLY THE TOXIC BEHVIORS WITH CITATIONS AND DETAILED EXPLANATIONS BASED ON SCHEMA THERAPY PRINCIPLES.
    <conversation_segment>
    {focused_context}
    </conversation_segment>
    """

                # Run extraction with comprehensive logging
                try:
                    logger.info(f"Running enhanced extraction for batch {batch_num}...")
                    print(f"Running enhanced extraction for batch {batch_num}...")
                    await agent.arun(extraction_prompt)

                    # Log successful batch processing
                    logger.info(f"Batch {batch_num} processed successfully")
                    logger.info(f"Conversations processed: {len(conversations)}")

                    print(f"Batch {batch_num} processed successfully with comprehensive logging")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    print(f"Error processing batch {batch_num}: {e}")
            else:
                logger.warning(f"No content in batch {batch_num}")
                print(f"No content in batch {batch_num}")

            processed_lines = end_idx
            batch_num += 1

    # Log completion
    logger.info("Knowledge extraction completed successfully")
    logger.info(f"Total batches processed: {batch_num - 1}")
    print("\nKnowledge extraction completed successfully!")
    print(f"Detailed log file: {log_file}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract psychological knowledge from relationship conversations using Schema Therapy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive CLI mode
    %(prog)s --mode cli --partner1 Alice --partner2 Bob

    # Batch processing mode
    %(prog)s --mode batch --input-file conversations.txt --output-file relationship_graph.json --partner1 Alice --partner2 Bob
    %(prog)s --mode batch --lines-per-batch 200 --partner1 Partner1 --partner2 Partner2

    # Default mode (batch for backward compatibility)
    %(prog)s --partner1 Alice --partner2 Bob
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "cli"],
        default="batch",
        help="Mode of operation: 'batch' for processing conversation logs, 'cli' for interactive conversation (default: batch)",
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default="examples/input.txt",
        help="Path to conversation log file (only used in batch mode, default: examples/input.txt)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="relationship_knowledge_graph.json",
        help="Path to save knowledge graph (default: relationship_knowledge_graph.json)",
    )

    parser.add_argument(
        "--partner1",
        type=str,
        default="Partner1",
        help="Name of the first partner (default: Partner1)",
    )

    parser.add_argument(
        "--partner2",
        type=str,
        default="Partner2",
        help="Name of the second partner (default: Partner2)",
    )

    parser.add_argument(
        "--lines-per-batch",
        type=int,
        default=100,
        help="Number of lines to process in each batch (only valid in batch mode, default: 100)",
    )

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == "cli":
        if args.lines_per_batch != 100:
            print("Error: --lines-per-batch is only valid in batch mode")
            return 1
        if args.input_file != "examples/input.txt":
            print("Warning: --input-file is ignored in CLI mode")
    else:  # batch mode
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file {input_path} does not exist")
            return 1

    output_path = Path(args.output_file)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Graph will be saved to: {output_path.absolute()}")

    try:
        if args.mode == "cli":
            # Run interactive CLI mode
            await process_conversations_interactive(
                output_file=output_path,
                partner1=args.partner1,
                partner2=args.partner2,
            )
        else:
            # Run batch mode (default)
            input_path = Path(args.input_file)
            await process_conversations_in_batches(
                input_file=input_path,
                output_file=output_path,
                partner1=args.partner1,
                partner2=args.partner2,
                lines_per_batch=args.lines_per_batch,
            )
            print("\nKnowledge extraction completed successfully!")
            print(f"Graph saved to: {output_path.absolute()}")
        return 0
    except Exception as e:
        print(f"\nError during knowledge extraction: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
