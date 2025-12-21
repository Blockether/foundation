"""Tests for dataclass_copy utility function."""

from dataclasses import dataclass, field

import pytest

from blockether_foundation.utils import dataclass_copy

# Constants to avoid magic values
DEFAULT_AGE = 30
UPDATED_AGE = 31
JANE_AGE = 25
UPDATED_AGE_2 = 26
BOB_AGE = 40
TEST_VALUE = 42


@dataclass
class SampleDataclass:
    """A simple dataclass for testing."""

    name: str
    age: int
    email: str | None = None
    # Internal field that's not in __init__ (simulates Agent behavior)
    _internal_field: str | None = field(default=None, init=False)


@pytest.mark.unit
def test_dataclass_copy_basic() -> None:
    """Test basic dataclass copying functionality."""
    obj = SampleDataclass(name="John", age=DEFAULT_AGE, email="john@example.com")

    # Copy without changes - returns same object when no changes
    copied = dataclass_copy(obj)
    assert copied == obj
    assert copied is obj  # Same instance (no changes)

    # Copy with changes
    updated = dataclass_copy(obj, age=UPDATED_AGE)
    assert updated.name == "John"
    assert updated.age == UPDATED_AGE
    assert updated.email == "john@example.com"
    assert updated is not obj  # Different instance (has changes)


@pytest.mark.unit
def test_dataclass_copy_filters_internal_fields() -> None:
    """Test that internal fields starting with _ are filtered out."""
    obj = SampleDataclass(name="Jane", age=JANE_AGE)
    obj._internal_field = "internal value"  # Set internal field  # type: ignore[attr-defined]

    # Try to copy with internal field - should be filtered out, and since no valid changes,
    # it should return the same object
    copied = dataclass_copy(obj, _internal_field="new value")
    assert copied is obj  # Same object (no valid changes)
    assert copied.name == "Jane"
    assert copied.age == JANE_AGE

    # Also test with valid changes - internal field should still be ignored
    copied_with_change = dataclass_copy(obj, age=UPDATED_AGE_2, _internal_field="new value")
    assert copied_with_change.name == "Jane"
    assert copied_with_change.age == UPDATED_AGE_2
    # Internal field should be None since we create a new instance
    assert copied_with_change._internal_field is None  # type: ignore[attr-defined]


@pytest.mark.unit
def test_dataclass_copy_non_dataclass() -> None:
    """Test that non-dataclasses are returned unchanged."""

    class NotADataclass:
        def __init__(self, value: int) -> None:
            self.value = value

    obj = NotADataclass(TEST_VALUE)
    result = dataclass_copy(obj, value=100)
    assert result is obj  # Should return the same object
    assert result.value == TEST_VALUE  # Should not modify


@pytest.mark.unit
def test_dataclass_copy_no_valid_changes() -> None:
    """Test that objects are returned unchanged when no valid fields are provided."""
    obj = SampleDataclass(name="Bob", age=BOB_AGE)

    # Try to update non-existent field
    result = dataclass_copy(obj, non_existent="value")
    assert result is obj  # Should return the same object


@pytest.mark.unit
def test_dataclass_copy_with_agent_like_class() -> None:
    """Test dataclass_copy with a class that simulates Agent behavior."""

    # This simulates the Agent class behavior where some fields are in dataclass
    # but not in __init__ parameters
    @dataclass
    class AgentLike:
        id: str
        name: str
        model: str | None = None
        db: str | None = None
        # Field that exists in dataclass but not in __init__
        _run_hooks_in_background: bool | None = field(default=None, init=False)

        # Custom __init__ that doesn't accept _run_hooks_in_background
        def __init__(self, *, id: str, name: str, model: str | None = None, db: str | None = None):
            self.id = id
            self.name = name
            self.model = model
            self.db = db
            self._run_hooks_in_background = None

    agent = AgentLike(id="test-agent", name="Test Agent")

    # This should work without errors
    copied = dataclass_copy(agent, model="new-model", db="new-db")

    assert copied.id == "test-agent"
    assert copied.name == "Test Agent"
    assert copied.model == "new-model"
    assert copied.db == "new-db"
    assert copied._run_hooks_in_background is None  # type: ignore[attr-defined]
