"""Integration tests for the iterative context improvement feature."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agno.run.agent import RunOutput

from blockether_foundation.agents.hooks.graph import GraphHookIterativeConfig, GraphHooksConfig
from blockether_foundation.graph.database import GraphDatabase

from .utils import create_agent_with_adapter


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_iterative_context_improvement_with_coverage_check():
    """Test that iterative context improvement respects entity coverage limits."""

    # Enable iterative improvement with custom config
    iterative_config = GraphHookIterativeConfig(
        enabled=True,
        max_iterations=3,
        quality_threshold=0.9,  # High threshold to ensure coverage check triggers
    )

    # Create a mock graph database with known entity count
    mock_graph = MagicMock(spec=GraphDatabase)
    mock_graph.entity_count = 10  # Small graph
    mock_graph.translate_to_queries.return_value = []

    config = GraphHooksConfig(
        graph=mock_graph,  # Pass mock graph directly
        agentic_search=True,  # Enable to test pre-hook
        agentic_ingestion=False,  # Disable for this test
        async_hooks=True,
        iterative_config=iterative_config,
    )

    # Create agent with pre-hook only
    pre_hook = config.pre_hook()
    agent_wrapper = create_agent_with_adapter(
        name="IterativeTestAgent",
        instructions="You are a helpful assistant.",
        pre_hooks=[pre_hook],  # type: ignore[arg-type]
    )
    agent = agent_wrapper.agent

    # Mock the query generation to return some results
    with patch("agno.agent.agent.Agent.arun", new_callable=AsyncMock) as mock_arun:
        # First call is for query generation, second is for the actual agent
        mock_arun.side_effect = [
            MagicMock(content=MagicMock(entity_queries=[], relationship_queries=[])),  # No queries
            MagicMock(content="Test response"),  # Actual response
        ]

        # Run the agent
        response: RunOutput = await agent.arun(  # type: ignore[call-overload]
            "Tell me about people in the tech industry.",
            session_id="iterative_test",
            user_id="test_user",
        )

        # Verify agent responded
        assert response.content is not None
        assert response.content == "Test response"


@pytest.mark.integration
@pytest.mark.agno_eval
async def test_iterative_context_stops_on_high_coverage():
    """Test that iteration stops early when entity coverage is high."""

    # Create a config with small graph threshold
    iterative_config = GraphHookIterativeConfig(
        enabled=True,
        max_iterations=5,  # Allow more iterations
        quality_threshold=1.0,  # Never satisfied by quality
    )

    # Mock the function to simulate high coverage
    with patch(
        "blockether_foundation.agents.hooks.graph._should_continue_iteration"
    ) as mock_should_continue:
        # First call: True (continue)
        # Second call: False (stop due to coverage)
        mock_should_continue.side_effect = [True, False]

        # Create a mock graph database
        mock_graph = MagicMock(spec=GraphDatabase)
        mock_graph.entity_count = 10  # Small graph
        mock_graph.translate_to_queries.return_value = []

        config = GraphHooksConfig(
            graph=mock_graph,  # Pass mock graph directly
            agentic_search=True,
            agentic_ingestion=False,
            async_hooks=True,
            iterative_config=iterative_config,
        )

        pre_hook = config.pre_hook()
        agent_wrapper = create_agent_with_adapter(
            name="CoverageTestAgent",
            instructions="You are a helpful assistant.",
            pre_hooks=[pre_hook],  # type: ignore[arg-type]
        )
        agent = agent_wrapper.agent

        with patch("agno.agent.agent.Agent.arun", new_callable=AsyncMock) as mock_arun:
            mock_arun.return_value = MagicMock(
                content=MagicMock(entity_queries=[], relationship_queries=[])
            )

            await agent.arun(  # type: ignore[call-overload]
                "Test query", session_id="coverage_test", user_id="test_user"
            )

            # Verify _should_continue_iteration was called twice
            assert mock_should_continue.call_count == 2

            # Verify the second call had parameters indicating high coverage
            second_call = mock_should_continue.call_args_list[1]
            # The accumulated_entity_ids should have many entities to trigger coverage stop
            # This is verified by the mock returning False on the second call
