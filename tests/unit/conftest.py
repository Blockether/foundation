"""
Unit test configuration for Blockether Foundation modules.

This module provides fixtures and configuration for unit testing
of individual components without integration dependencies.
"""

import pytest


@pytest.fixture
def reset_encoder_state():
    """Reset PotionEncoder singleton state between tests."""
    from blockether_foundation.encoder.potion import PotionEncoder

    # Store original state
    original_model = PotionEncoder._model
    original_initialized = PotionEncoder._initialized

    # Reset state
    PotionEncoder._model = None
    PotionEncoder._initialized = False

    yield

    # Restore original state
    PotionEncoder._model = original_model
    PotionEncoder._initialized = original_initialized


@pytest.fixture
def mock_static_model():
    """Mock StaticModel for testing."""
    from unittest.mock import Mock

    mock_model = Mock()
    mock_model.encode.return_value = Mock()
    return mock_model
