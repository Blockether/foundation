"""
Unit test configuration for ACE modules.

This module provides fixtures and configuration for unit testing
of individual ACE components without integration dependencies.
"""

from unittest.mock import Mock

import dspy  # type: ignore[import-untyped]
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
    mock_model = Mock()
    mock_model.encode.return_value = Mock()
    return mock_model

@pytest.fixture
def mock_lm():
    """Mock LM for unit testing."""
    return Mock()


@pytest.fixture
def mock_generator_module():
    """Mock GeneratorModule for unit testing."""
    return Mock()


@pytest.fixture
def mock_reflector_module():
    """Mock ReflectorModule for unit testing."""
    return Mock()


@pytest.fixture
def mock_curator_module():
    """Mock CuratorModule for unit testing."""
    return Mock()


@pytest.fixture
def mock_dspy_predict():
    """Mock dspy.Predict for unit testing."""
    return Mock()


@pytest.fixture
def mock_dspy_lm():
    """Mock dspy.LM for unit testing."""

    # Create a proper mock that inherits from BaseLM
    class MockBaseLM(dspy.BaseLM):
        def __init__(self):
            # Don't call super().__init__() to avoid model requirement
            self.kwargs = {"temperature": 0.7}
            self.model = "mock-model"  # DSPy expects this attribute

        def generate(self, prompt=None, **kwargs):
            # Return a mock response (legacy method)
            mock_response = Mock()
            mock_response.output = "Mock response for testing"
            return mock_response

        def forward(self, prompt=None, messages=None, **kwargs):
            # Implement the required forward method for DSPy
            # Return a response that matches OpenAI format

            class MockChoice:
                def __init__(self):
                    self.message = Mock()
                    self.message.content = "Mock reasoning\nMock final answer\nstrategies-00000"

            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]
                    self.usage = Mock()
                    self.usage.prompt_tokens = 10
                    self.usage.completion_tokens = 5

            return MockResponse()

        def create_prompt(self, prompt=None, **kwargs):
            return f"Mock LLM prompt for: {prompt}"

    return MockBaseLM()


@pytest.fixture
def mock_configured_dspy(mock_dspy_lm):
    """Mock configured DSPy LM for unit testing.

    This fixture mimics the integration test configured_dspy fixture
    but for unit tests without real LLM calls.
    """
    # Configure DSPy with mock LM
    original_lm = getattr(dspy.settings, "lm", None)
    dspy.configure(lm=mock_dspy_lm)

    yield mock_dspy_lm

    # Restore original LM
    if original_lm:
        dspy.configure(lm=original_lm)
    else:
        # Reset to no LM
        dspy.settings.lm = None


# Simple mock responses for DSPy modules
@pytest.fixture
def mock_generator_response():
    """Mock generator response for testing."""
    from blockether_foundation.ace.models import GeneratorOutput

    return GeneratorOutput(
        reasoning="Breaking down the problem step by step using the available strategies.",
        final_answer="Mock answer: 4",
        bullet_ids="strategies-00000",
    )


@pytest.fixture
def mock_reflector_response():
    """Mock reflector response for testing."""
    from blockether_foundation.ace.models import ReflectorOutput

    return ReflectorOutput(
        analysis="The approach was effective and the strategies used were appropriate.",
        helpful_bullets=["strategies-00000"],
        harmful_bullets=[],
        neutral_bullets=[],
        suggestions="Continue using this step-by-step approach.",
    )


@pytest.fixture
def mock_curator_response():
    """Mock curator response for testing."""
    from blockether_foundation.ace.models import CuratorOutput, DeltaBatch, DeltaOperation

    # Create a simple mock delta operation
    delta_batch = DeltaBatch(
        reasoning="The strategy should be reinforced as it was helpful.",
        operations=[
            DeltaOperation(type="TAG", bullet_id="strategies-00000", metadata={"helpful": 1})
        ],
    )

    return CuratorOutput(delta=delta_batch)


# Fixtures for mock outputs
@pytest.fixture
def generator_output():
    """Mock generator output for testing."""
    mock_output = Mock()
    mock_output.reasoning = "Breaking down the problem: 15% = 0.15, so 0.15 × 240 = 36"
    mock_output.final_answer = "36"
    mock_output.bullet_ids = "strategies-00000,strategies-00001"
    mock_output.confidence = 0.95
    return mock_output


@pytest.fixture
def reflector_output():
    """Mock reflector output for testing."""
    mock_output = Mock()
    mock_output.analysis = "The calculation is correct and follows the percentage formula properly."
    mock_output.helpful_bullets = "strategies-00000"
    mock_output.harmful_bullets = ""
    mock_output.neutral_bullets = "strategies-00001"
    mock_output.suggestions = "Could show more explicit step-by-step work"
    return mock_output


@pytest.fixture
def curator_output():
    """Mock curator output for testing."""
    mock_output = Mock()
    mock_output.reasoning = "The percentage calculation strategy should be reinforced."
    mock_output.operations = (
        '[{"type": "TAG", "bullet_id": "strategies-00000", "metadata": {"helpful": 1}}]'
    )
    return mock_output


def mock_llm_response(content: str):
    """Create a mock LLM response with the given content."""
    mock_response = Mock()
    mock_response.output = content
    return mock_response
