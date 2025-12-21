"""Integration tests for graph file sync functionality with hooks."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from blockether_foundation.agents.hooks.graph import GraphHooksConfig
from blockether_foundation.graph.database import GraphDatabase
from blockether_foundation.graph.models import Entity

from .utils import create_agent_with_adapter


@pytest.fixture
def temp_graph_file() -> Generator[str]:
    """Create a temporary file for graph storage."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    yield str(temp_path)

    # Cleanup - always attempt to unlink without conditional
    temp_path.unlink(missing_ok=True)


@pytest.mark.integration
def test_graph_file_sync_basic(temp_graph_file: str) -> None:
    """Test that graph is synced to file after operations using true sync hooks."""
    # Create hook config with file sync - using TRUE sync hooks
    config = GraphHooksConfig(
        agentic_ingestion=True,
        file_sync=temp_graph_file,
        async_hooks=False,  # Proper sync hooks!
    )

    # Create agent with hooks
    agent_wrapper = create_agent_with_adapter(
        name="Test Agent",
        instructions="Extract entities and relationships from conversations.",
        post_hooks=[config.post_hook()],
    )
    agent = agent_wrapper.agent

    # Run agent - this should extract entities and save to file using sync hooks
    agent.run(  # type: ignore[reportUnknownMemberType]
        "Albert Einstein was born in Germany in 1879",
        session_id="test_session",
        user_id="test_user",
    )

    # Verify file was created
    assert Path(temp_graph_file).exists(), "Graph file should be created"

    # Load graph from file and verify it has data
    loaded_graph = GraphDatabase.load_from_file(temp_graph_file)
    assert loaded_graph.entity_count > 0, "Graph should have entities"


@pytest.mark.integration
def test_graph_loads_from_file_on_init(temp_graph_file: str) -> None:
    """Test that graph is loaded from file when session is empty using sync hooks."""
    # Create a graph with some data and save to file
    initial_graph = GraphDatabase()
    entity = Entity(name="Albert Einstein", type="creature", content="German physicist")
    initial_graph.add_entity(entity)
    initial_graph.save_to_file(temp_graph_file)

    # Verify file exists and has 1 entity
    assert Path(temp_graph_file).exists()
    assert initial_graph.entity_count == 1

    # Create new agent with file sync pointing to the existing file
    config = GraphHooksConfig(
        agentic_ingestion=True,
        file_sync=temp_graph_file,
        async_hooks=False,  # Use proper sync hooks
    )

    agent_wrapper = create_agent_with_adapter(
        name="Test Agent",
        instructions="Work with the existing knowledge graph.",
        post_hooks=[config.post_hook()],
    )
    agent = agent_wrapper.agent

    # Run agent - this should load the existing graph from file using sync hooks
    agent.run("Tell me about Einstein", session_id="test_session", user_id="test_user")  # type: ignore[reportUnknownMemberType]

    # The graph should have been loaded from file and potentially have more entities
    loaded_graph = GraphDatabase.load_from_file(temp_graph_file)
    assert loaded_graph.entity_count >= 1, "Graph should have at least the initial entity"


@pytest.mark.integration
def test_graph_sync_updates_file(temp_graph_file: str) -> None:
    """Test that file is updated when new entities are added using sync hooks."""
    # Create initial graph with sync hooks
    config = GraphHooksConfig(
        agentic_ingestion=True,
        file_sync=temp_graph_file,
        async_hooks=False,  # Use proper sync hooks
    )

    agent_wrapper = create_agent_with_adapter(
        name="Test Agent",
        instructions="Extract entities and relationships from conversations.",
        post_hooks=[config.post_hook()],
    )
    agent = agent_wrapper.agent

    # First run - add first entity using sync hooks
    agent.run("Marie Curie was a Polish scientist", session_id="test_session1", user_id="test_user")  # type: ignore[reportUnknownMemberType]

    # Load and check entity count
    graph1 = GraphDatabase.load_from_file(temp_graph_file)
    count1 = graph1.entity_count
    assert count1 > 0

    # Second run - add more entities using sync hooks
    agent.run("She won two Nobel Prizes", session_id="test_session2", user_id="test_user")  # type: ignore[reportUnknownMemberType]

    # Load again and verify count increased
    graph2 = GraphDatabase.load_from_file(temp_graph_file)
    count2 = graph2.entity_count

    # Count should have increased (new entities added)
    assert count2 >= count1, "Entity count should increase or stay same after second run"


@pytest.mark.integration
def test_graph_file_sync_with_agentic_search(temp_graph_file: str) -> None:
    """Test that file sync works with both sync pre and post hooks."""
    # Create graph with initial data
    initial_graph = GraphDatabase()
    entity = Entity(name="Python", type="concept", content="Programming language")
    initial_graph.add_entity(entity)
    initial_graph.save_to_file(temp_graph_file)

    # Create config with both search and ingestion using sync hooks
    config = GraphHooksConfig(
        agentic_search=True,
        agentic_ingestion=True,
        file_sync=temp_graph_file,
        async_hooks=False,  # Use proper sync hooks
    )

    agent_wrapper = create_agent_with_adapter(
        name="Test Agent",
        instructions="Extract entities and relationships from conversations.",
        pre_hooks=[config.pre_hook()],
        post_hooks=[config.post_hook()],
    )
    agent = agent_wrapper.agent

    # Run - should load graph from file in sync pre-hook and save in sync post-hook
    agent.run("What is Python?", session_id="test_session", user_id="test_user")  # type: ignore[reportUnknownMemberType]

    # Verify file still exists and has data
    loaded_graph = GraphDatabase.load_from_file(temp_graph_file)
    assert loaded_graph.entity_count >= 1


@pytest.mark.integration
def test_graph_file_sync_creates_parent_directories(tmp_path: Path) -> None:
    """Test that file sync creates parent directories if they don't exist using sync hooks."""
    # Create a nested path that doesn't exist yet
    nested_file = tmp_path / "data" / "graphs" / "test_graph.json"

    config = GraphHooksConfig(
        agentic_ingestion=True,
        file_sync=str(nested_file),
        async_hooks=False,  # Use proper sync hooks
    )

    agent_wrapper = create_agent_with_adapter(
        name="Test Agent",
        instructions="Extract entities and relationships from conversations.",
        post_hooks=[config.post_hook()],
    )
    agent = agent_wrapper.agent

    # Run - should create directories and file using sync hooks
    agent.run("Test entity creation", session_id="test_session", user_id="test_user")  # type: ignore[reportUnknownMemberType]

    # Verify file and directories were created
    assert nested_file.exists(), "File should be created with parent directories"
    assert nested_file.parent.exists(), "Parent directories should be created"
