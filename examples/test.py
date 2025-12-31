#!/usr/bin/env python3
"""
Minimal Agno example using glm-4.7 with a simple output schema.
Drop this file into the examples/ directory and run it.
"""

import os
import sys
from pathlib import Path

# Ensure src is importable (matches project examples)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel, Field

from agno.agent import Agent
from blockether_foundation.agents.models.zai import Zhipu

MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")

if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

# Initialize glm-4.7 with the same flags used in your examples
glm_4_7 = Zhipu(
    id="glm-4.6v-flash",
    enable_coding_plan=True,
    enable_thinking=False,
)


class RandomOutput(BaseModel):
    """A tiny random output schema for demonstration."""

    request_id: str = Field(description="Unique request identifier")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    tags: list[str] = Field(default_factory=list, description="Arbitrary tags")
    notes: str | None = Field(default=None, description="Optional short notes")


def create_minimal_agent() -> Agent:
    instructions = """You are a concise assistant. Given the prompt, fill the structured schema
fields in a truthful, compact manner. Output must conform to the RandomOutput schema."""
    return Agent(
        name="Minimal GLM Agent",
        description="Demonstrates a simple Agno Agent using glm-4.7 and a Pydantic schema",
        instructions=[instructions],
        model=glm_4_7,
        output_schema=RandomOutput,
        debug_mode=True,
        markdown=False,
    )


def main() -> None:
    agent = create_minimal_agent()
    prompt = "Evaluate this short idea: 'A pocket translator that adapts to local dialects.' Give a short judgement."
    response = agent.run(prompt)
    # response.content should be parsed to RandomOutput if model/tooling supports structured outputs
    print("Raw response object:", response)
    print("\nParsed content:")
    print(response.content)


if __name__ == "__main__":
    main()
