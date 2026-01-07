#!/usr/bin/env python3
"""
Agno example using Zhipu with complex nested output_schema AND regular tools.
Tests that structured outputs via tools work correctly with complex schemas.
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

# Initialize Zhipu
glm_4_7 = Zhipu(
    id="glm-4.7",
    enable_coding_plan=True,
    enable_thinking=False,
)


# ============================================================================
# COMPLEX NESTED OUTPUT SCHEMA (similar to consensus/core.py patterns)
# ============================================================================


class KeyInsight(BaseModel):
    """A key insight with reasoning about why it matters."""

    insight: str = Field(description="The insight itself.")
    evidence: str | None = Field(default=None, description="Supporting evidence or examples.")
    importance: float = Field(
        ge=0.0, le=1.0, description="How important this insight is (0.0-1.0)."
    )


class StrengthWeakness(BaseModel):
    """A strength or weakness with supporting evidence."""

    description: str = Field(description="Description of the strength or weakness.")
    severity_or_value: float = Field(
        ge=0.0, le=1.0, description="Severity (for weakness) or value (for strength) score."
    )
    evidence: str = Field(description="Quote or reference supporting this assessment.")


class ConsideredAlternative(BaseModel):
    """An alternative approach that was considered."""

    approach: str = Field(description="Description of the alternative approach.")
    why_not_chosen: str | None = Field(
        default=None, description="Reason this approach was not selected."
    )
    potential_value: float = Field(
        ge=0.0, le=1.0, description="How valuable this alternative might be."
    )


class ConfidenceBreakdown(BaseModel):
    """Confidence scores broken down by different aspects."""

    factual_accuracy: float = Field(
        ge=0.0, le=1.0, description="Confidence in factual correctness."
    )
    completeness: float = Field(
        ge=0.0, le=1.0, description="Confidence that all aspects are covered."
    )
    logical_coherence: float = Field(
        ge=0.0, le=1.0, description="Confidence in logical consistency."
    )
    relevance_to_task: float = Field(
        ge=0.0, le=1.0, description="Confidence output addresses the task."
    )


class MissingConsideration(BaseModel):
    """An important aspect that was missed."""

    consideration: str = Field(description="The consideration that was missed.")
    suggested_action: str | None = Field(
        default=None, description="What should be done to address this."
    )
    importance: float = Field(ge=0.0, le=1.0, description="How important this consideration is.")


class SuggestedImprovement(BaseModel):
    """A suggestion for improvement."""

    what_to_change: str = Field(description="What should be changed.")
    why: str = Field(description="Why this change is needed.")
    proposed_fix: str = Field(description="The proposed fix or improvement.")
    expected_impact: float = Field(ge=0.0, le=1.0, description="Expected impact (0.0-1.0).")


class ComplexEvaluationOutput(BaseModel):
    """A complex nested evaluation output with multiple levels of structured data."""

    request_id: str = Field(description="Unique request identifier.")
    summary: str = Field(description="Brief summary of the evaluation.")

    # Nested: strengths and weaknesses
    strengths: list[StrengthWeakness] = Field(
        default=[], description="Identified strengths with evidence."
    )
    weaknesses: list[StrengthWeakness] = Field(
        default=[], description="Identified weaknesses with evidence."
    )

    # Nested: key insights
    key_insights: list[KeyInsight] = Field(
        default=[], description="Main insights with reasoning and impact."
    )

    # Nested: considered alternatives
    considered_alternatives: list[ConsideredAlternative] = Field(
        default=[], description="Alternative approaches considered."
    )

    # Nested: missing considerations
    missing_considerations: list[MissingConsideration] = Field(
        default=[], description="Important aspects that were missed."
    )

    # Nested: suggested improvements
    suggested_improvements: list[SuggestedImprovement] = Field(
        default=[], description="Suggested improvements with reasoning."
    )

    # Nested: confidence breakdown
    confidence_breakdown: ConfidenceBreakdown = Field(
        description="Confidence scores broken down by aspect."
    )

    # Overall assessment
    overall_score: float = Field(ge=0.0, le=10.0, description="Overall rating score.")
    recommendation: str = Field(description="Final recommendation or verdict.")


# ============================================================================
# TOOLS
# ============================================================================


def get_current_time() -> str:
    """Get the current time in a readable format."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def search_knowledge(query: str) -> str:
    """Search for information in the knowledge base."""
    return f"Found 5 relevant results for '{query}': detailed analysis 1, detailed analysis 2, detailed analysis 3, case study, expert opinion"


def get_weather_info(location: str) -> str:
    """Get current weather information for a location."""
    return f"Weather in {location}: 22°C, partly cloudy, humidity 65%, wind 12 km/h from NW"


def calculate_metric(value1: float, value2: float, operation: str) -> str:
    """Perform a calculation on two numeric values."""

    if operation == "add":
        result = value1 + value2
    elif operation == "multiply":
        result = value1 * value2
    elif operation == "divide":
        result = value1 / value2 if value2 != 0 else 0
    else:
        result = value1 - value2
    return f"Calculation result: {value1} {operation} {value2} = {result}"


# ============================================================================
# AGENT SETUP
# ============================================================================


def create_agent_with_complex_schema() -> Agent:
    """Create an agent with complex nested output_schema and multiple tools."""
    instructions = """You are an analytical assistant that provides thorough evaluations.

IMPORTANT - You MUST ALWAYS use the available tools:
1. ALWAYS call get_current_time at the beginning of your response
2. ALWAYS call search_knowledge to gather relevant information
3. Use other tools as needed for the evaluation"""

    return Agent(
        name="Zhipu Complex Schema Agent",
        description="Demonstrates complex nested structured outputs with multiple tools",
        instructions=[instructions],
        model=glm_4_7,
        tools=[get_current_time, search_knowledge, get_weather_info, calculate_metric],
        output_schema=ComplexEvaluationOutput,
        debug_mode=True,
        markdown=False,
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    agent = create_agent_with_complex_schema()

    print("=" * 70)
    print("Complex Nested Schema Test with Multiple Tools")
    print("=" * 70)

    prompt = (
        "Please evaluate the following idea in detail: "
        "'A smart home system that automatically adjusts temperature based on "
        "occupant behavior patterns and preferences.' "
        "Start by getting the current time, then search for relevant information, "
        "check the weather, and provide a comprehensive evaluation."
    )

    response = agent.run(prompt)

    print("\n" + "=" * 70)
    print("FINAL PARSED OUTPUT")
    print("=" * 70)
    print(f"Content Type: {type(response.content).__name__}")
    print("\nStructured Output:")
    print(response.content)

    # Also show the type info
    if hasattr(response.content, "__dict__"):
        print("\n" + "=" * 70)
        print("SCHEMA FIELDS")
        print("=" * 70)
        for field_name, field_info in response.content.model_fields.items():
            print(f"  {field_name}: {type(field_info).__name__}")


if __name__ == "__main__":
    main()
