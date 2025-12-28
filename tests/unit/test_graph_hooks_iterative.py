"""Tests for the iterative context improvement in graph hooks."""

from unittest.mock import MagicMock, patch

from blockether_foundation.agents.hooks.graph import (
    GraphHookIterativeConfig,
    GraphHooksConfig,
    _deduplicate_and_accumulate,
    _should_continue_iteration,
)

DEFAULT_QUALITY_THRESHOLD = 0.75
DEFAULT_MAX_ENTITIES_TO_EXPAND = 3
CUSTOM_MAX_ITERATIONS = 5
CUSTOM_QUALITY_THRESHOLD = 0.9
CUSTOM_MAX_ENTITIES_TO_EXPAND = 5


class TestGraphHookIterativeConfig:
    """Test the iterative configuration class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = GraphHookIterativeConfig()
        assert config.enabled is False
        assert config.max_iterations == 2
        assert config.quality_threshold == DEFAULT_QUALITY_THRESHOLD
        assert config.enable_gap_analysis is True
        assert config.enable_entity_expansion is True
        assert config.enable_query_refinement is True
        assert config.bfs_expansion_depth == 1
        assert config.max_entities_to_expand == DEFAULT_MAX_ENTITIES_TO_EXPAND

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = GraphHookIterativeConfig(
            enabled=True,
            max_iterations=5,
            quality_threshold=0.9,
            enable_gap_analysis=False,
            bfs_expansion_depth=2,
            max_entities_to_expand=5,
        )
        assert config.enabled is True
        assert config.max_iterations == CUSTOM_MAX_ITERATIONS
        assert config.quality_threshold == CUSTOM_QUALITY_THRESHOLD
        assert config.enable_gap_analysis is False
        assert config.enable_entity_expansion is True
        assert config.enable_query_refinement is True
        assert config.bfs_expansion_depth == 2
        assert config.max_entities_to_expand == CUSTOM_MAX_ENTITIES_TO_EXPAND


class TestShouldContinueIteration:
    """Test the _should_continue_iteration function."""

    def test_at_max_iteration(self):
        """Test that iteration stops at max iteration."""
        config = GraphHookIterativeConfig()

        result = _should_continue_iteration(
            accumulated_entity_ids={"e1"},
            total_graph_entities=100,
            iteration=3,
            max_iterations=3,
            config=config,
        )
        assert result is False, "Should stop at max iteration"

    def test_small_graph_high_coverage(self):
        """Test stopping on small graphs with high coverage."""
        config = GraphHookIterativeConfig()

        # Small graph (10 entities) with 9 retrieved (90% coverage)
        result = _should_continue_iteration(
            accumulated_entity_ids={f"e{i}" for i in range(9)},
            total_graph_entities=10,
            iteration=2,
            max_iterations=3,
            config=config,
        )
        assert result is False, "Should stop on small graph with 90% coverage"

    def test_small_graph_low_coverage(self):
        """Test continuing on small graphs with low coverage."""
        config = GraphHookIterativeConfig()

        # Small graph (10 entities) with 5 retrieved (50% coverage)
        result = _should_continue_iteration(
            accumulated_entity_ids={f"e{i}" for i in range(5)},
            total_graph_entities=10,
            iteration=2,
            max_iterations=3,
            config=config,
        )
        assert result is True, "Should continue on small graph with 50% coverage"

    def test_medium_graph_moderate_coverage(self):
        """Test stopping on medium graphs with moderate coverage."""
        config = GraphHookIterativeConfig()

        # Medium graph (50 entities) with 35 retrieved (70% coverage)
        result = _should_continue_iteration(
            accumulated_entity_ids={f"e{i}" for i in range(35)},
            total_graph_entities=50,
            iteration=2,
            max_iterations=3,
            config=config,
        )
        assert result is False, "Should stop on medium graph with 70% coverage"

    def test_large_graph_low_coverage(self):
        """Test continuing on large graphs with low coverage."""
        config = GraphHookIterativeConfig()

        # Large graph (100 entities) with 30 retrieved (30% coverage)
        result = _should_continue_iteration(
            accumulated_entity_ids={f"e{i}" for i in range(30)},
            total_graph_entities=100,
            iteration=2,
            max_iterations=3,
            config=config,
        )
        assert result is True, "Should continue on large graph with 30% coverage"

    def test_large_graph_many_entities(self):
        """Test stopping on large graphs with many entities retrieved."""
        config = GraphHookIterativeConfig()

        # Large graph (100 entities) with 60 retrieved after iteration 2
        result = _should_continue_iteration(
            accumulated_entity_ids={f"e{i}" for i in range(60)},
            total_graph_entities=100,
            iteration=2,
            max_iterations=3,
            config=config,
        )
        assert result is False, "Should stop on large graph with 60+ entities after iteration 2"

    def test_division_by_zero_protection(self):
        """Test that the function handles empty graphs gracefully."""
        config = GraphHookIterativeConfig()

        # Empty graph should not crash
        result = _should_continue_iteration(
            accumulated_entity_ids=set(),
            total_graph_entities=0,
            iteration=1,
            max_iterations=3,
            config=config,
        )
        # Should continue since we haven't reached max iteration
        assert result is True, "Should handle empty graph gracefully"


class TestDeduplicateAndAccumulate:
    """Test the _deduplicate_and_accumulate function."""

    def test_empty_accumulated_context(self):
        """Test accumulation when starting from empty."""
        new_context = """
        <entities>
            <entity id="e1" type="person">John</entity>
            <entity id="e2" type="organization">Acme</entity>
        </entities>
        """

        result_context, result_ids = _deduplicate_and_accumulate(
            new_context=new_context,
            accumulated_context="",
            accumulated_entity_ids=set(),
        )

        assert result_context == new_context
        assert result_ids == {"e1", "e2"}

    def test_with_existing_context(self):
        """Test accumulation with existing context."""
        existing_context = """
        <entities>
            <entity id="e1" type="person">John</entity>
        </entities>
        """
        new_context = """
        <entities>
            <entity id="e2" type="organization">Acme</entity>
            <entity id="e3" type="project">ProjectX</entity>
        </entities>
        """

        result_context, result_ids = _deduplicate_and_accumulate(
            new_context=new_context,
            accumulated_context=existing_context,
            accumulated_entity_ids={"e1"},
        )

        assert existing_context in result_context
        assert new_context in result_context
        assert result_ids == {"e1", "e2", "e3"}

    def test_no_new_entities(self):
        """Test when no new entities are added."""
        existing_context = """
        <entities>
            <entity id="e1" type="person">John</entity>
        </entities>
        """
        new_context = """
        <entities>
            <entity id="e1" type="person">John</entity>
        </entities>
        """

        result_context, result_ids = _deduplicate_and_accumulate(
            new_context=new_context,
            accumulated_context=existing_context,
            accumulated_entity_ids={"e1"},
        )

        assert result_context == existing_context
        assert result_ids == {"e1"}

    def test_mixed_new_and_duplicate_entities(self):
        """Test with mix of new and duplicate entities."""
        existing_context = """
        <entities>
            <entity id="e1" type="person">John</entity>
        </entities>
        """
        new_context = """
        <entities>
            <entity id="e1" type="person">John</entity>
            <entity id="e2" type="organization">Acme</entity>
            <entity id="e3" type="person">Jane</entity>
        </entities>
        """

        result_context, result_ids = _deduplicate_and_accumulate(
            new_context=new_context,
            accumulated_context=existing_context,
            accumulated_entity_ids={"e1"},
        )

        assert "e1" in result_context
        assert "e2" in result_context
        assert "e3" in result_context
        assert result_ids == {"e1", "e2", "e3"}


class TestGraphHooksConfigWithIterative:
    """Test GraphHooksConfig with iterative configuration."""

    def test_default_iterative_config(self):
        """Test that default iterative config is used."""
        config = GraphHooksConfig()
        assert config.iterative_config is not None
        assert config.iterative_config.enabled is False
        assert config.iterative_config.max_iterations == 2

    def test_custom_iterative_config(self):
        """Test that custom iterative config is used."""
        iterative_config = GraphHookIterativeConfig(
            enabled=True,
            max_iterations=5,
        )
        config = GraphHooksConfig(iterative_config=iterative_config)
        assert config.iterative_config == iterative_config
        assert config.iterative_config.enabled is True
        assert config.iterative_config.max_iterations == CUSTOM_MAX_ITERATIONS

    @patch("blockether_foundation.agents.hooks.graph._create_pre_graph_database_hook")
    def test_pre_hook_passes_iterative_config(self, mock_create_hook):
        """Test that pre-hook receives the iterative config."""
        mock_create_hook.return_value = MagicMock()

        config = GraphHooksConfig(
            agentic_search=True,
            async_hooks=True,
            iterative_config=GraphHookIterativeConfig(enabled=True, max_iterations=5),
        )

        pre_hook = config.pre_hook()
        pre_hook  # Call to trigger the mock

        mock_create_hook.assert_called_once_with(
            graph=config.graph,
            agentic_search=True,
            iterative_config=config.iterative_config,
            return_sync_wrapper=False,
        )

    @patch("blockether_foundation.agents.hooks.graph._create_pre_graph_database_hook")
    def test_sync_pre_hook_passes_iterative_config(self, mock_create_hook):
        """Test that sync pre-hook receives the iterative config."""
        mock_create_hook.return_value = MagicMock()

        config = GraphHooksConfig(
            agentic_search=True,
            async_hooks=False,
            iterative_config=GraphHookIterativeConfig(enabled=True, max_iterations=3),
        )

        pre_hook = config.pre_hook()
        pre_hook  # Call to trigger the mock

        mock_create_hook.assert_called_once_with(
            graph=config.graph,
            agentic_search=True,
            iterative_config=config.iterative_config,
            return_sync_wrapper=True,
        )
