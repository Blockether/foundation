"""Tests for dataclass_copy utility function."""

import pytest
from dataclasses import dataclass, field
from typing import Optional

from blockether_foundation.utils import dataclass_copy


@dataclass
class TestDataclass:
    """A simple dataclass for testing."""
    name: str
    age: int
    email: Optional[str] = None
    # Internal field that's not in __init__ (simulates Agent behavior)
    _internal_field: Optional[str] = field(default=None, init=False)


def test_dataclass_copy_basic():
    """Test basic dataclass copying functionality."""
    obj = TestDataclass(name="John", age=30, email="john@example.com")

    # Copy without changes - returns same object when no changes
    copied = dataclass_copy(obj)
    assert copied == obj
    assert copied is obj  # Same instance (no changes)

    # Copy with changes
    updated = dataclass_copy(obj, age=31)
    assert updated.name == "John"
    assert updated.age == 31
    assert updated.email == "john@example.com"
    assert updated is not obj  # Different instance (has changes)


def test_dataclass_copy_filters_internal_fields():
    """Test that internal fields starting with _ are filtered out."""
    obj = TestDataclass(name="Jane", age=25)
    obj._internal_field = "internal value"  # Set internal field

    # Try to copy with internal field - should be filtered out, and since no valid changes,
    # it should return the same object
    copied = dataclass_copy(obj, _internal_field="new value")
    assert copied is obj  # Same object (no valid changes)
    assert copied.name == "Jane"
    assert copied.age == 25

    # Also test with valid changes - internal field should still be ignored
    copied_with_change = dataclass_copy(obj, age=26, _internal_field="new value")
    assert copied_with_change.name == "Jane"
    assert copied_with_change.age == 26
    # Internal field should be None since we create a new instance
    assert copied_with_change._internal_field is None


def test_dataclass_copy_non_dataclass():
    """Test that non-dataclasses are returned unchanged."""
    class NotADataclass:
        def __init__(self, value):
            self.value = value

    obj = NotADataclass(42)
    result = dataclass_copy(obj, value=100)
    assert result is obj  # Should return the same object
    assert result.value == 42  # Should not modify


def test_dataclass_copy_no_valid_changes():
    """Test that objects are returned unchanged when no valid fields are provided."""
    obj = TestDataclass(name="Bob", age=40)

    # Try to update non-existent field
    result = dataclass_copy(obj, non_existent="value")
    assert result is obj  # Should return the same object


def test_dataclass_copy_with_agent_like_class():
    """Test dataclass_copy with a class that simulates Agent behavior."""
    # This simulates the Agent class behavior where some fields are in dataclass
    # but not in __init__ parameters
    @dataclass
    class AgentLike:
        id: str
        name: str
        model: Optional[str] = None
        db: Optional[str] = None
        # Field that exists in dataclass but not in __init__
        _run_hooks_in_background: Optional[bool] = field(default=None, init=False)

        # Custom __init__ that doesn't accept _run_hooks_in_background
        def __init__(self, *, id: str, name: str, model: Optional[str] = None, db: Optional[str] = None):
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
    assert copied._run_hooks_in_background is None