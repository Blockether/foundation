"""Example: Multi-model consensus to evaluate graph schema completeness for a task."""

import os
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from blockether_foundation.agents.hooks import (
    ConsensusHooksConfig,
    ConsensusResult,
    JudgeCriteria,
    ModelConfig,
)
from blockether_foundation.graph.common import (
    ENTITY_TYPE_DEFINITIONS,
    GRAPH_SCHEMA_XML,
    RELATIONSHIP_TYPE_DEFINITIONS,
)

MODEL_BASE_URL = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
MODEL_API_KEY = os.getenv("BLOCKETHER_LLM_API_KEY")

if not MODEL_BASE_URL:
    raise ValueError("BLOCKETHER_LLM_API_BASE_URL environment variable is not set.")

gpt_4_1 = OpenAIChat(id="gpt-4.1", base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

gpt_5_mini = OpenAIChat(id="gpt-5-mini", base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)

glm_4_7 = OpenAIChat(id="glm-4.6", base_url=MODEL_BASE_URL, api_key=MODEL_API_KEY)


def create_schema_analysis_agent(task_description: str) -> Agent:
    """Create an agent that analyzes graph schema completeness for a given task."""
    model_configs = [
        ModelConfig(
            name="gpt-4.1",
            model=gpt_4_1,
            importance=0.5,
            perspective="Focus on semantic completeness and edge cases",
        ),
        # ModelConfig(
        #     name="gpt-5-mini",
        #     model=gpt_5_mini,
        #     importance=0.3,
        #     perspective="Focus on practical usability and simplicity",
        # ),
        ModelConfig(
            name="glm-4.7",
            model=glm_4_7,
            importance=0.2,
            perspective="Focus on formal structure and type consistency",
        ),
    ]

    consensus_config = ConsensusHooksConfig(
        async_hooks=False,
        models=model_configs,
        judge_criteria=[
            JudgeCriteria(
                name="Schema Coverage",
                description="Are all entity and relationship types needed for the task present in the schema?",
                weight=1.0,
                threshold=0.7,
            ),
            JudgeCriteria(
                name="Relationship Sufficiency",
                description="Can all meaningful connections for the task be expressed with the available relationship types?",
                weight=1.0,
                threshold=0.7,
            ),
            JudgeCriteria(
                name="Actionable Recommendations",
                description="Are the recommendations specific, justified, and implementable?",
                weight=0.8,
                threshold=0.6,
            ),
        ],
        max_refinement_iterations=2,
        judge_threshold=0.7,
        auto_save_html=True,
        output_directory=Path("./consensus_reports"),
    )

    entity_types_list = "\n".join(
        f"  - {etype}: {definition}" for etype, definition in ENTITY_TYPE_DEFINITIONS.items()
    )
    relationship_types_list = "\n".join(
        f"  - {rtype}: {definition}" for rtype, definition in RELATIONSHIP_TYPE_DEFINITIONS.items()
    )

    instructions = f"""You are a knowledge graph schema analyst. Your role is to evaluate whether
the given entity and relationship types are sufficient and complete for representing
information about a specific task or domain.

CURRENT GRAPH SCHEMA:

Entity Types:
{entity_types_list}

Relationship Types:
{relationship_types_list}

When analyzing schema completeness for a task, consider:
1. What entities would need to be represented for this task?
2. What relationships between those entities are meaningful?
3. Are there any entity types missing that would be essential?
4. Are there any relationship types missing that would be essential?
5. Are there any redundant or overlapping types?

Provide your analysis in a structured format with:
- COVERAGE ASSESSMENT: Which existing types apply to this task
- GAPS IDENTIFIED: Missing entity or relationship types (if any)
- RECOMMENDATIONS: Specific additions or changes with justification
- CONFIDENCE: How confident you are in the schema's completeness (0-100%)
"""

    return Agent(
        name="Graph Schema Analyst",
        description="Analyzes knowledge graph schema completeness for specific tasks",
        instructions=[instructions],
        model=gpt_4_1,
        pre_hooks=[consensus_config.pre_hook()],
        debug_mode=True,
    )


def analyze_schema_for_task(task_description: str) -> None:
    """Run consensus analysis on whether the graph schema is complete for a task."""
    agent = create_schema_analysis_agent(task_description)

    prompt = f"""Analyze whether the current graph schema (entity types and relationship types)
is complete and sufficient for the following task:

TASK: {task_description}

Provide a comprehensive analysis of:
1. Which existing entity types are relevant for this task
2. Which existing relationship types are relevant for this task  
3. Any entity types that are MISSING and would be needed
4. Any relationship types that are MISSING and would be needed
5. Your overall assessment: Is the schema complete for this task? Why or why not?
6. Specific recommendations for schema improvements (if any)
"""

    print(f"\n{'=' * 80}")
    print(f"ANALYZING SCHEMA COMPLETENESS FOR TASK:")
    print(f"{'=' * 80}")
    print(task_description)
    print(f"{'=' * 80}\n")

    response = agent.run(prompt)
    print("\n" + "=" * 80)
    print("FINAL RESPONSE (after consensus):")
    print("=" * 80)
    print(response.content)


if __name__ == "__main__":
    example_task = """
    Build a knowledge graph for a personal assistant that needs to:
    - Track the user's professional network (colleagues, mentors, clients)
    - Remember meeting notes and action items from conversations
    - Store user preferences and habits
    - Link documents and emails to relevant people and topics
    - Track project timelines and dependencies
    - Remember personal details about contacts (birthdays, interests, family)
    """

    analyze_schema_for_task(example_task)
