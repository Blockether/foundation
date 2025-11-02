"""Pytest configuration and fixtures for Blockether Foundation tests.

This module provides common fixtures and configuration for the test suite,
including database setup, logging configuration, and test utilities.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_logging():
    """Configure logging for tests."""
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.NullHandler()],  # Suppress output during tests
    )

    # Set specific logger levels
    logging.getLogger("blockether_foundation").setLevel(logging.DEBUG)
    logging.getLogger("agno").setLevel(logging.WARNING)

    yield

    # Clean up logging
    logging.getLogger().handlers.clear()


@pytest.fixture
def temp_database():
    """Placeholder fixture for temporary database testing."""
    # TODO: Implement when database classes are added
    yield None


@pytest.fixture
def mock_database():
    """Placeholder fixture for mock database testing."""
    # TODO: Implement when database classes are added
    yield None


@pytest.fixture
def agno_in_memory_db():
    """Placeholder fixture for Agno in-memory database testing."""
    # TODO: Implement when database classes are added
    yield None


@pytest.fixture
def blockether_in_memory_db():
    """Placeholder fixture for Blockether in-memory database testing."""
    # TODO: Implement when database classes are added
    yield None


@pytest.fixture
def sample_agent_template_data():
    """Sample agent template data for testing."""
    return {
        "id": "test-agent-template",
        "name": "Test Agent Template",
        "description": "A test agent template for unit testing",
        "base_configuration": {
            "name": "Test Agent",
            "description": "Test agent description",
            "type": "ace",
            "model": {"provider": "openai", "model": "gpt-4", "temperature": 0.7},
            "instructions": "You are a test agent.",
            "tools": [],
            "memory": None,
            "knowledge": None,
            "metadata": {"test": True},
        },
        "category": "test",
        "tags": ["test", "agent", "template"],
        "inheritable_fields": {
            "name",
            "description",
            "instructions",
            "tools",
            "memory",
            "knowledge",
            "metadata",
            "tags",
        },
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_workflow_template_data():
    """Sample workflow template data for testing."""
    return {
        "id": "test-workflow-template",
        "name": "Test Workflow Template",
        "description": "A test workflow template for unit testing",
        "category": "test",
        "workflow_definition": {
            "name": "Test Workflow",
            "description": "Test workflow description",
            "steps": [
                {
                    "id": "step1",
                    "name": "Input Step",
                    "type": "agent",
                    "config": {"agent_type": "input_processor"},
                },
                {
                    "id": "step2",
                    "name": "Processing Step",
                    "type": "agent",
                    "config": {"agent_type": "main_processor"},
                },
                {
                    "id": "step3",
                    "name": "Output Step",
                    "type": "agent",
                    "config": {"agent_type": "output_formatter"},
                },
            ],
            "connections": [{"from": "step1", "to": "step2"}, {"from": "step2", "to": "step3"}],
            "metadata": {"total_steps": 3},
        },
        "builder_metadata": {"ui_layout": "horizontal", "theme": "default"},
        "parameterizable_fields": ["agent_type", "model"],
        "tags": ["test", "workflow", "template"],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_team_template_data():
    """Sample team template data for testing."""
    return {
        "id": "test-team-template",
        "name": "Test Team Template",
        "description": "A test team template for unit testing",
        "coordination_strategy": "sequential",
        "roles": [
            {
                "id": "role1",
                "name": "Team Lead",
                "type": "agent",
                "responsibilities": ["coordination", "decision_making"],
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "instructions": "Lead the team effectively.",
                },
            },
            {
                "id": "role2",
                "name": "Specialist",
                "type": "agent",
                "responsibilities": ["analysis", "expertise"],
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "instructions": "Provide specialized analysis.",
                },
            },
        ],
        "metadata": {"team_size": 2, "coordination_pattern": "lead_specialist"},
        "tags": ["test", "team", "template"],
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_builder_metadata():
    """Sample builder metadata for testing."""
    return {
        "template_id": "test-template",
        "created_by": "test-user",
        "created_at": "2024-01-01T12:00:00Z",
        "builder_positions": {
            "x": 100.0,
            "y": 200.0,
            "width": 300.0,
            "height": 400.0,
            "z_index": 1,
        },
        "ui_settings": {
            "theme": "dark",
            "panel_states": {
                "configuration": "expanded",
                "tools": "collapsed",
                "memory": "expanded",
            },
            "custom_colors": {"primary": "#2563eb", "secondary": "#64748b"},
        },
        "coordination_strategy": "sequential",
    }


# Test markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests (fast, use mocks)")
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower, use real resources)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (run only when explicitly requested)"
    )


# Custom assertion helpers for better test readability
def assert_template_structure(template_data: dict, expected_type: str):
    """Assert that template data has the correct structure for its type."""
    assert isinstance(template_data, dict)
    assert "id" in template_data
    assert "name" in template_data
    assert "created_at" in template_data
    assert "updated_at" in template_data

    if expected_type == "agent":
        assert "base_configuration" in template_data
        assert isinstance(template_data["base_configuration"], dict)
    elif expected_type == "workflow":
        assert "workflow_definition" in template_data
        assert isinstance(template_data["workflow_definition"], dict)
    elif expected_type == "team":
        assert "coordination_strategy" in template_data
        assert isinstance(template_data["coordination_strategy"], str)
        assert "roles" in template_data
        assert isinstance(template_data["roles"], list)


def assert_builder_metadata_structure(metadata: dict):
    """Assert that builder metadata has the correct structure."""
    assert isinstance(metadata, dict)
    assert "created_at" in metadata
    assert "builder_positions" in metadata
    assert isinstance(metadata["builder_positions"], dict)

    # Check position structure
    positions = metadata["builder_positions"]
    assert "x" in positions
    assert "y" in positions
    assert isinstance(positions["x"], (int, float))
    assert isinstance(positions["y"], (int, float))


# Test data generators for parameterized tests
def generate_invalid_template_data():
    """Generate various invalid template data for error testing."""
    return [
        # Invalid data types
        "string_instead_of_dict",
        123,
        None,
        [],
        # Missing required fields
        {"name": "Missing ID and base config"},
        {"id": "test"},  # Missing base_configuration for agent
        {"id": "test", "base_configuration": {}},  # Missing name
        # Invalid ID types
        {"id": {"invalid": "object"}, "name": "test", "base_configuration": {}},
        {"id": [], "name": "test", "base_configuration": {}},
    ]


def generate_invalid_builder_metadata():
    """Generate various invalid builder metadata for error testing."""
    return [
        # Invalid entity types
        ("invalid_entity", "test-id", {"template_id": "test"}),
        ("", "test-id", {"template_id": "test"}),
        # Invalid entity IDs
        ("agent", "", {"template_id": "test"}),
        ("agent", 123, {"template_id": "test"}),
        ("agent", None, {"template_id": "test"}),
        # Invalid metadata
        ("agent", "test-id", "string_instead_of_dict"),
        ("agent", "test-id", 123),
        ("agent", "test-id", None),
        ("agent", "test-id", []),
    ]


# Make assertion helpers available to tests
pytest.assert_template_structure = assert_template_structure
pytest.assert_builder_metadata_structure = assert_builder_metadata_structure
pytest.generate_invalid_template_data = generate_invalid_template_data
pytest.generate_invalid_builder_metadata = generate_invalid_builder_metadata
