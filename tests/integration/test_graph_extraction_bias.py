"""Integration tests for graph extraction bias and correctness.

These tests verify that:
1. Mundane conversations are NOT pathologized
2. Insufficient evidence produces minimal/no psychological claims
3. Neutral messages are extracted factually, not interpretively
4. The extraction system doesn't hallucinate toxic behaviors

IMPORTANT: These tests use REAL examples to catch prompt bias issues.
"""

from __future__ import annotations

import pytest
from agno.run.agent import RunOutput

from blockether_foundation.agents.hooks.graph import (
    GraphHooksConfig,
    GraphIngestionIterativeConfig,
)
from blockether_foundation.graph import GraphDatabase

from .utils import create_agent_with_adapter


# =============================================================================
# TEST DATA: Real-world mundane conversations
# =============================================================================

MUNDANE_CONVERSATION_1 = """
[23/12/2024, 11:31:56] Alice: You wanna save this?
[23/12/2024, 11:32:11] Alice: I thought i sent this to you already ????
[23/12/2024, 11:49:28] Alice: Missed video call
[23/12/2024, 11:49:40] Bob: Video call - No answer
[23/12/2024, 11:49:52] Bob: Give me 2 minutes!! Hihi
[23/12/2024, 11:50:05] Alice: See the pic?
[23/12/2024, 11:50:14] Bob: Yes! I got it already thankfully
"""

MUNDANE_CONVERSATION_2 = """
[10/01/2025, 09:15:00] Sarah: Hey, did you pick up milk?
[10/01/2025, 09:16:22] John: Oh shoot, I forgot! I'll grab some on the way home
[10/01/2025, 09:17:01] Sarah: No worries, we can manage without
[10/01/2025, 09:17:45] John: Sorry about that
[10/01/2025, 09:18:10] Sarah: It's fine :) see you tonight
"""

MUNDANE_CONVERSATION_3 = """
[15/01/2025, 14:00:00] Partner1: The delivery guy just came
[15/01/2025, 14:00:30] Partner2: Nice! What did you order?
[15/01/2025, 14:01:15] Partner1: Those books we talked about
[15/01/2025, 14:02:00] Partner2: Cool, can't wait to see them
"""


# =============================================================================
# TOXIC KEYWORDS - These should NOT appear in extraction of mundane conversations
# =============================================================================

TOXIC_BEHAVIOR_KEYWORDS = [
    "toxic",
    "abandonment",
    "schema activation",
    "childhood origin",
    "punitive",
    "defectiveness",
    "shame",
    "mistrust",
    "abuse",
    "emotional deprivation",
    "vulnerable child",
    "angry child",
    "detached protector",
    "overcompensation",
    "passive aggressive",
    "manipulation",
    "gaslighting",
    "narcissistic",
    "borderline",
    "trauma",
    "dysfunction",
    "pathological",
    "maladaptive",
]


def count_toxic_keywords_in_entities(graph: GraphDatabase) -> dict[str, list[str]]:
    """Count occurrences of toxic keywords in entity content."""
    findings: dict[str, list[str]] = {}

    all_entities = graph.query_entities().execute().results
    for entity in all_entities:
        content_lower = (entity.content or "").lower()
        name_lower = entity.name.lower()

        for keyword in TOXIC_BEHAVIOR_KEYWORDS:
            if keyword in content_lower or keyword in name_lower:
                if entity.name not in findings:
                    findings[entity.name] = []
                findings[entity.name].append(keyword)

    return findings


# =============================================================================
# NEUTRAL EXTRACTION AGENT - No psychological bias
# =============================================================================

NEUTRAL_EXTRACTION_INSTRUCTIONS = """
You are a factual knowledge extractor. Extract ONLY what is explicitly stated:

EXTRACT:
- People mentioned by name
- Dates and times mentioned
- Events that actually occurred (calls, messages, deliveries)
- Factual statements made

DO NOT EXTRACT OR INFER:
- Psychological states or emotions (unless explicitly stated like "I'm happy")
- Intentions or motivations
- Relationship dynamics
- Any interpretation beyond the literal text

If someone says "I forgot", extract: "Person forgot something" - NOT psychological analysis.
If someone misses a call, extract: "Missed call event" - NOT abandonment patterns.
"""


def create_neutral_extraction_agent(
    graph_config: GraphHooksConfig,
) -> tuple:
    """Create an agent with neutral (non-biased) extraction instructions."""
    post_hook = graph_config.post_hook()
    agent_wrapper = create_agent_with_adapter(
        name="NeutralExtractionAgent",
        instructions=NEUTRAL_EXTRACTION_INSTRUCTIONS,
        post_hooks=[post_hook],  # type: ignore[list-item]
    )
    return agent_wrapper.agent, graph_config


# =============================================================================
# TESTS: Mundane conversations should NOT produce toxic behavior findings
# =============================================================================


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_mundane_conversation_no_toxic_keywords():
    """Verify mundane conversation extraction contains no toxic behavior keywords."""

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    agent, _ = create_neutral_extraction_agent(config)

    response: RunOutput = await agent.arun(
        f"Extract facts from this conversation:\n{MUNDANE_CONVERSATION_1}",
        session_id="mundane_test_1",
        user_id="test_user",
    )

    assert response.content is not None

    graph_db = config.graph
    toxic_findings = count_toxic_keywords_in_entities(graph_db)

    assert len(toxic_findings) == 0, (
        f"Mundane conversation should NOT produce toxic behavior keywords. "
        f"Found: {toxic_findings}"
    )


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_forgotten_milk_is_not_defectiveness():
    """Verify 'I forgot milk' is not interpreted as defectiveness/shame schema."""

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    agent, _ = create_neutral_extraction_agent(config)

    response: RunOutput = await agent.arun(
        f"Extract facts from this conversation:\n{MUNDANE_CONVERSATION_2}",
        session_id="milk_test",
        user_id="test_user",
    )

    assert response.content is not None

    graph_db = config.graph
    all_entities = graph_db.query_entities().execute().results

    # Check that no entity mentions shame, defectiveness, or guilt
    shame_keywords = ["shame", "defectiveness", "guilt", "failure", "inadequa"]
    for entity in all_entities:
        content_lower = (entity.content or "").lower()
        for keyword in shame_keywords:
            assert keyword not in content_lower, (
                f"Forgetting milk should NOT be interpreted as {keyword}. "
                f"Entity: {entity.name}, Content: {entity.content}"
            )


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_missed_call_is_not_abandonment():
    """Verify missed call is NOT interpreted as abandonment schema activation."""

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    agent, _ = create_neutral_extraction_agent(config)

    # Use the conversation that has missed calls
    response: RunOutput = await agent.arun(
        f"Extract facts from this conversation:\n{MUNDANE_CONVERSATION_1}",
        session_id="missed_call_test",
        user_id="test_user",
    )

    assert response.content is not None

    graph_db = config.graph
    all_entities = graph_db.query_entities().execute().results

    # Check that no entity mentions abandonment
    abandonment_keywords = ["abandon", "rejection", "unavailabl", "neglect"]
    for entity in all_entities:
        content_lower = (entity.content or "").lower()
        for keyword in abandonment_keywords:
            assert keyword not in content_lower, (
                f"Missed call should NOT be interpreted as {keyword}. "
                f"Entity: {entity.name}, Content: {entity.content}"
            )


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_question_marks_are_not_anxiety():
    """Verify multiple question marks are NOT interpreted as anxiety/emotional dysregulation."""

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    agent, _ = create_neutral_extraction_agent(config)

    response: RunOutput = await agent.arun(
        f"Extract facts from this conversation:\n{MUNDANE_CONVERSATION_1}",
        session_id="question_marks_test",
        user_id="test_user",
    )

    assert response.content is not None

    graph_db = config.graph
    all_entities = graph_db.query_entities().execute().results

    # Check that no entity mentions anxiety or emotional dysregulation
    anxiety_keywords = ["anxiety", "dysregulation", "emotional contagion", "crisis"]
    for entity in all_entities:
        content_lower = (entity.content or "").lower()
        for keyword in anxiety_keywords:
            assert keyword not in content_lower, (
                f"Question marks should NOT be interpreted as {keyword}. "
                f"Entity: {entity.name}, Content: {entity.content}"
            )


# =============================================================================
# TESTS: Verify factual extraction is working correctly
# =============================================================================


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_extracts_people_correctly():
    """Verify that people are correctly extracted from conversation."""

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    agent, _ = create_neutral_extraction_agent(config)

    response: RunOutput = await agent.arun(
        f"Extract facts from this conversation:\n{MUNDANE_CONVERSATION_2}",
        session_id="people_test",
        user_id="test_user",
    )

    assert response.content is not None

    graph_db = config.graph
    person_entities = graph_db.find_entities_by_type("person")
    person_names = {e.name.lower() for e in person_entities}

    # Should extract Sarah and John
    assert any("sarah" in name for name in person_names), (
        f"Should extract Sarah. Found: {person_names}"
    )
    assert any("john" in name for name in person_names), (
        f"Should extract John. Found: {person_names}"
    )


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_extracts_events_not_interpretations():
    """Verify that events are extracted factually, not interpretatively."""

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    agent, _ = create_neutral_extraction_agent(config)

    response: RunOutput = await agent.arun(
        f"Extract facts from this conversation:\n{MUNDANE_CONVERSATION_3}",
        session_id="events_test",
        user_id="test_user",
    )

    assert response.content is not None

    graph_db = config.graph
    all_entities = graph_db.query_entities().execute().results

    # Should have simple factual entities, not psychological interpretations
    entity_types = {e.type for e in all_entities}

    # Should NOT have schema or mode entities for a delivery conversation
    assert "schema" not in entity_types, (
        "Delivery conversation should NOT produce schema entities"
    )
    assert "mode" not in entity_types, (
        "Delivery conversation should NOT produce mode entities"
    )


# =============================================================================
# TESTS: Minimum evidence threshold
# =============================================================================


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_minimal_input_produces_minimal_extraction():
    """Verify that minimal input produces minimal extraction, not elaborate theories."""

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    agent, _ = create_neutral_extraction_agent(config)

    minimal_input = "[10:00] Alice: Hi\n[10:01] Bob: Hey"

    response: RunOutput = await agent.arun(
        f"Extract facts from this conversation:\n{minimal_input}",
        session_id="minimal_test",
        user_id="test_user",
    )

    assert response.content is not None

    graph_db = config.graph
    entity_count = graph_db.entity_count

    # "Hi" and "Hey" should produce very few entities (just the people, maybe)
    assert entity_count <= 5, (
        f"Minimal conversation should produce minimal extraction. "
        f"Found {entity_count} entities"
    )


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_no_extraction_from_system_messages():
    """Verify that system messages (missed call, etc.) don't produce psychological analysis."""

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    agent, _ = create_neutral_extraction_agent(config)

    system_only = """
    [11:49:28] Alice: Missed video call
    [11:49:40] Bob: Video call - No answer
    """

    response: RunOutput = await agent.arun(
        f"Extract facts from this conversation:\n{system_only}",
        session_id="system_messages_test",
        user_id="test_user",
    )

    assert response.content is not None

    graph_db = config.graph
    toxic_findings = count_toxic_keywords_in_entities(graph_db)

    assert len(toxic_findings) == 0, (
        f"System messages should NOT produce toxic behavior keywords. "
        f"Found: {toxic_findings}"
    )


# =============================================================================
# COMPARISON TEST: Show the difference between biased vs neutral extraction
# =============================================================================


BIASED_SCHEMA_THERAPY_INSTRUCTIONS = """
You are Dr. Schema - a Schema Therapy expert extracting deep psychological knowledge.

CORE REQUIREMENTS:
1. WHY IT HAPPENED: Schema activations, childhood origins, unmet needs
2. SCHEMA MODES: Which mode speaks for each partner (Vulnerable Child, Angry Child, etc.)

EXTRACT:
- Schema activations and triggers
- Toxic behaviors with citations
- Attachment patterns
- Childhood origins of current behaviors
"""


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_biased_vs_neutral_extraction_comparison():
    """Compare biased vs neutral extraction to demonstrate the problem.

    This test shows that the SAME input produces wildly different outputs
    depending on prompt bias.
    """

    # Run with neutral agent
    neutral_config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    neutral_agent, _ = create_neutral_extraction_agent(neutral_config)

    await neutral_agent.arun(
        f"Extract facts from this conversation:\n{MUNDANE_CONVERSATION_1}",
        session_id="comparison_neutral",
        user_id="test_user",
    )

    neutral_toxic_count = len(count_toxic_keywords_in_entities(neutral_config.graph))

    # Run with biased agent
    biased_config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    biased_post_hook = biased_config.post_hook()
    biased_wrapper = create_agent_with_adapter(
        name="BiasedExtractionAgent",
        instructions=BIASED_SCHEMA_THERAPY_INSTRUCTIONS,
        post_hooks=[biased_post_hook],  # type: ignore[list-item]
    )
    biased_agent = biased_wrapper.agent

    await biased_agent.arun(
        f"Extract facts from this conversation:\n{MUNDANE_CONVERSATION_1}",
        session_id="comparison_biased",
        user_id="test_user",
    )

    biased_toxic_count = len(count_toxic_keywords_in_entities(biased_config.graph))

    # The neutral agent should find FEWER toxic keywords than the biased one
    # This demonstrates that prompt bias causes over-extraction of pathology
    print(f"\nComparison Results:")
    print(f"  Neutral agent toxic keywords: {neutral_toxic_count}")
    print(f"  Biased agent toxic keywords: {biased_toxic_count}")

    # The neutral agent should have 0 or very few toxic findings
    assert neutral_toxic_count <= 1, (
        f"Neutral extraction should find minimal toxic keywords. Found: {neutral_toxic_count}"
    )
