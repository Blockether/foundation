"""
Unit test configuration for Blockether Foundation modules.

This module provides fixtures and configuration for unit testing
of individual components without integration dependencies.
"""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_static_model():
    """Mock StaticModel for testing."""
    mock_model = Mock()
    mock_model.encode.return_value = Mock()
    return mock_model
