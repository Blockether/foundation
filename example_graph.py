"""
Interactive conversational example demonstrating GraphHooksConfig usage.

This example shows how the graph hooks automatically extract and store
knowledge from agent conversations, with duplicate detection and self-correction.

Run with: python example_graph.py

Try asking about:
- Programming languages (Python, JavaScript, etc.)
- Who created them
- Frameworks and libraries
- Then ask '/graph' to see what knowledge was extracted!
"""

import asyncio
import logging
import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool  # type: ignore

from src.blockether_foundation.agents.hooks.graph import GraphHooksConfig

# Set up logging to see what's happening behind the scenes
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@tool()
def show_graph_stats() -> str:
    """Tool to show current graph statistics."""
    # For now, return a placeholder message
    # The graph data is stored in the agent's session_data via the hooks
    return "Graph statistics are being tracked automatically. The graph contains entities and relationships extracted from our conversation."


MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")

# Validate required environment variables
if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

print(
    f"Initializing blockether_model... model_base_url={MODEL_BASE_URL}, model_api_key={MODEL_API_KEY}"
)

blockether_model = OpenAIChat(
    id="gpt-4.1", base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY, modalities=["text"]
)


async def main():
    """Run the interactive graph hooks example."""

    print("\n" + "=" * 80)
    print("Interactive Graph Hooks Demo")
    print("=" * 80)
    print("\nThis agent automatically extracts knowledge from conversations!")
    print("Knowledge is stored in a global graph that persists across sessions.")
    print("\nTry asking about:")
    print("  • Programming languages (Python, JavaScript, Rust, etc.)")
    print("  • Who created them")
    print("  • Frameworks and libraries (Django, React, etc.)")
    print("  • Technologies and tools\n")
    print("Then type '/graph' to see what knowledge was extracted!")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    print("=" * 80 + "\n")

    # Create configuration for global graph with async hooks (default)
    config = GraphHooksConfig(
        agentic_search=False,  # Can enable to query graph for context
    )

    # Create an agent with graph hooks and the graph tool
    agent = Agent(
        model=blockether_model,
        name="Knowledge Assistant",
        instructions=[
            "You are a helpful assistant that provides detailed, factual answers.",
            "When asked about programming languages, technologies, or people in tech, provide accurate information.",
            "Keep your responses concise but informative.",
            "The conversation is being analyzed to build a knowledge graph automatically.",
        ],
        pre_hooks=[config.pre_hook()],  # Query graph for context
        post_hooks=[config.post_hook()],  # Extract and store knowledge from responses
        tools=[show_graph_stats],  # Add tool to show graph stats
        markdown=True,
        debug_mode=True,
    )

    await agent.acli_app()


if __name__ == "__main__":
    asyncio.run(main())
