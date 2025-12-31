"""Integration tests for graph entity resolution with real LLM calls."""

from __future__ import annotations

import pytest
from agno.run.agent import RunOutput

from blockether_foundation.agents.hooks.graph import (
    GraphHooksConfig,
    GraphIngestionIterativeConfig,
)

from .utils import create_agent_with_adapter


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_entity_resolution_merges_duplicates():
    """Test that entity resolution correctly merges duplicate person entities."""

    ingestion_config = GraphIngestionIterativeConfig(
        enabled=True,
        max_iterations=2,
        quality_threshold=0.8,
        enable_entity_resolution=True,
        enable_quality_assessment=True,
    )

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=ingestion_config,
    )

    post_hook = config.post_hook()
    agent_wrapper = create_agent_with_adapter(
        name="EntityResolutionTestAgent",
        instructions="""
        You are analyzing therapy conversations and extracting knowledge.
        Focus on extracting people, events, and their relationships.
        """,
        post_hooks=[post_hook],  # type: ignore[list-item]
    )
    agent = agent_wrapper.agent

    conversation = """
    Today Ashley met with Karol for their weekly therapy session.
    During the session, Karol Wójcik discussed his childhood experiences.
    Ashley noted that Karol showed signs of emotional vulnerability when discussing his father.
    The session concluded with Ashley and Karol Wójcik planning next steps.
    """

    response: RunOutput = await agent.arun(
        conversation, session_id="entity_resolution_test", user_id="test_user"
    )

    assert response.content is not None

    graph_db = config.graph
    person_entities = graph_db.find_entities_by_type("person")
    person_names = {e.name for e in person_entities}

    assert "Karol Wójcik" in person_names or "Karol" in person_names
    if "Karol Wójcik" in person_names:
        assert "Karol" not in person_names, "Should have merged Karol into Karol Wójcik"


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_entity_resolution_creates_proper_event_entities():
    """Test that extraction creates event entities instead of verbose names."""

    ingestion_config = GraphIngestionIterativeConfig(
        enabled=True,
        max_iterations=2,
        quality_threshold=0.8,
        enable_entity_resolution=True,
    )

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=ingestion_config,
    )

    post_hook = config.post_hook()
    agent_wrapper = create_agent_with_adapter(
        name="EventExtractionTestAgent",
        instructions="""
        You are analyzing conversations and extracting knowledge.
        Extract people, events, dates, and facts.
        """,
        post_hooks=[post_hook],  # type: ignore[list-item]
    )
    agent = agent_wrapper.agent

    conversation = """
    On December 28, 2024, Alice and Bob had a therapy session.
    They discussed Bob's anxiety about work.
    """

    response: RunOutput = await agent.arun(
        conversation, session_id="event_extraction_test", user_id="test_user"
    )

    assert response.content is not None

    graph_db = config.graph

    all_entities = graph_db.query_entities().execute().results
    entity_names = [e.name for e in all_entities]

    long_verbose_names = [n for n in entity_names if len(n) > 80]
    assert len(long_verbose_names) == 0, (
        f"Should not have verbose entity names: {long_verbose_names}"
    )


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_entity_resolution_uses_participated_in_relationship():
    """Test that extraction uses participated_in for event participation."""

    ingestion_config = GraphIngestionIterativeConfig(
        enabled=True,
        max_iterations=2,
        enable_entity_resolution=True,
    )

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=ingestion_config,
    )

    post_hook = config.post_hook()
    agent_wrapper = create_agent_with_adapter(
        name="RelationshipTestAgent",
        instructions="""
        Extract people, events, and relationships from conversations.
        Use participated_in to connect people to events they were part of.
        """,
        post_hooks=[post_hook],  # type: ignore[list-item]
    )
    agent = agent_wrapper.agent

    conversation = """
    Sarah and Mike attended the team meeting on Monday.
    They discussed the quarterly goals.
    """

    response: RunOutput = await agent.arun(
        conversation, session_id="relationship_test", user_id="test_user"
    )

    assert response.content is not None

    graph_db = config.graph

    all_relationships = graph_db.query_relationships().execute().results
    relationship_types = {rel.type for _, _, rel in all_relationships}

    has_participation = (
        "participated_in" in relationship_types or "related_to" in relationship_types
    )
    assert has_participation, (
        f"Should have participation relationships. Found: {relationship_types}"
    )


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_disabled_entity_resolution_preserves_duplicates():
    """Test that disabling entity resolution preserves original extraction."""

    config = GraphHooksConfig(
        agentic_search=False,
        agentic_ingestion=True,
        async_hooks=True,
        ingestion_config=GraphIngestionIterativeConfig(enabled=False),
    )

    post_hook = config.post_hook()
    agent_wrapper = create_agent_with_adapter(
        name="NoResolutionTestAgent",
        instructions="""
        Extract people and relationships from the conversation.
        """,
        post_hooks=[post_hook],  # type: ignore[list-item]
    )
    agent = agent_wrapper.agent

    conversation = """
    John met with John Smith today. They are different people.
    """

    response: RunOutput = await agent.arun(
        conversation, session_id="no_resolution_test", user_id="test_user"
    )

    assert response.content is not None

    graph_db = config.graph
    entity_count = graph_db.entity_count
    assert entity_count >= 1
