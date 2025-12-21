"""Tests for the models module."""

import os
import tempfile
from typing import Any, TypeVar, cast

import pytest

from blockether_foundation.models import BaseModelSerializable, ChainOfThoughts

# TypeVar for properly typed BaseModelSerializable subclasses
T = TypeVar("T", bound=BaseModelSerializable)


# Define test models at module level so they can be pickled
class DemoModel(BaseModelSerializable):
    """Simple test model with basic types."""

    name: str
    value: int


class ComplexDemoModel(BaseModelSerializable):
    """Test model with complex nested types."""

    data: dict[str, Any]
    items: list[Any]


class ModelWithObjDemo(BaseModelSerializable):
    """Test model with object attribute."""

    obj: Any


# Test constants
TEST_VALUE = 42
TEST_CONFIDENCE = 0.85


class TestBaseModelSerializable:
    """Test cases for BaseModelSerializable."""

    @pytest.mark.unit
    def test_json_serialization(self) -> None:
        """Test JSON serialization methods."""
        # Create test instance
        model = DemoModel(name="test", value=TEST_VALUE)

        # Test to_json_string
        json_str = model.to_json_string()
        assert "test" in json_str
        assert str(TEST_VALUE) in json_str

        # Test from_json_string
        loaded = cast(DemoModel, DemoModel.from_json_string(json_str))
        assert loaded.name == "test"
        assert loaded.value == TEST_VALUE

    @pytest.mark.unit
    def test_json_file_serialization(self) -> None:
        """Test JSON file serialization methods."""
        model = DemoModel(name="test", value=TEST_VALUE)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        # Test to_json_file
        model.to_json_file(filepath)

        # Test from_json_file
        loaded = cast(DemoModel, DemoModel.from_json_file(filepath))
        assert loaded.name == "test"
        assert loaded.value == TEST_VALUE

        # Clean up
        os.unlink(filepath)

    @pytest.mark.unit
    def test_pickle_serialization(self) -> None:
        """Test pickle serialization methods."""
        model = DemoModel(name="test", value=TEST_VALUE)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            filepath = f.name

        # Test to_pickle_file
        model.to_pickle_file(filepath)

        # Test from_pickle_file
        loaded = cast(DemoModel, DemoModel.from_pickle_file(filepath))
        assert loaded.name == "test"
        assert loaded.value == TEST_VALUE

        # Clean up
        os.unlink(filepath)

    @pytest.mark.unit
    def test_base64_serialization(self) -> None:
        """Test base64 serialization methods."""
        model = DemoModel(name="test", value=TEST_VALUE)

        # Test to_base64
        base64_str = model.to_base64()
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

        # Test from_base64
        loaded = cast(DemoModel, DemoModel.from_base64(base64_str))
        assert loaded.name == "test"
        assert loaded.value == TEST_VALUE

    @pytest.mark.unit
    def test_json_string_roundtrip_with_complex_data(self) -> None:
        """Test JSON serialization with complex nested data."""
        model = ComplexDemoModel(
            data={"key": "value", "nested": {"x": 1, "y": 2}}, items=[1, 2, 3, {"a": "b"}]
        )

        # Test roundtrip
        json_str = model.to_json_string()
        loaded = cast(ComplexDemoModel, ComplexDemoModel.from_json_string(json_str))
        assert loaded.data == model.data
        assert loaded.items == model.items

    @pytest.mark.unit
    def test_pickle_roundtrip_with_object(self) -> None:
        """Test pickle serialization with object attributes."""
        # Create with a simple object
        test_obj: dict[str, Any] = {"complex": "object", "number": 123}
        model = ModelWithObjDemo(obj=test_obj)

        # Test base64 roundtrip
        base64_str = model.to_base64()
        loaded = cast(ModelWithObjDemo, ModelWithObjDemo.from_base64(base64_str))
        assert loaded.obj == test_obj


class TestChainOfThoughts:
    """Test cases for ChainOfThoughts model."""

    @pytest.mark.unit
    def test_chain_of_thoughts_creation(self) -> None:
        """Test creating a ChainOfThoughts instance."""
        thoughts = ChainOfThoughts(
            reasoning="This is my reasoning process", confidence=TEST_CONFIDENCE, importance=0.8
        )

        assert thoughts.reasoning == "This is my reasoning process"
        assert thoughts.confidence == TEST_CONFIDENCE

    @pytest.mark.unit
    def test_chain_of_thoughts_validation(self) -> None:
        """Test ChainOfThoughts validation."""
        # Test valid confidence values
        thoughts1 = ChainOfThoughts(reasoning="test", confidence=0.0, importance=0.5)
        assert thoughts1.confidence == 0.0

        thoughts2 = ChainOfThoughts(reasoning="test", confidence=1.0, importance=0.9)
        assert thoughts2.confidence == 1.0

        # Test invalid confidence values
        with pytest.raises(ValueError):
            ChainOfThoughts(reasoning="test", confidence=-0.1)

        with pytest.raises(ValueError):
            ChainOfThoughts(reasoning="test", confidence=1.1)

    @pytest.mark.unit
    def test_chain_of_thoughts_serialization(self) -> None:
        """Test ChainOfThoughts serialization."""
        thoughts = ChainOfThoughts(
            reasoning="Step 1: Analyze the problem\nStep 2: Consider alternatives\nStep 3: Make decision",
            confidence=0.92,
            importance=0.95,
        )

        # Test JSON serialization
        json_str = thoughts.to_json_string()
        loaded = cast(ChainOfThoughts, ChainOfThoughts.from_json_string(json_str))
        assert loaded.reasoning == thoughts.reasoning
        assert loaded.confidence == thoughts.confidence

    @pytest.mark.unit
    def test_chain_of_thoughts_inheritance(self) -> None:
        """Test that ChainOfThoughts inherits BaseModelSerializable."""
        thoughts = ChainOfThoughts(reasoning="My reasoning", confidence=0.75, importance=0.8)

        # Should have BaseModelSerializable methods
        assert isinstance(thoughts, BaseModelSerializable)
        # Test that the methods exist and are callable
        assert isinstance(thoughts, BaseModelSerializable)
        assert callable(thoughts.to_json_string)
        assert callable(thoughts.to_base64)
        assert callable(thoughts.to_pickle_file)
        # Test that the methods actually work
        assert thoughts.to_json_string() is not None
        assert thoughts.to_base64() is not None

    @pytest.mark.unit
    def test_chain_of_thoughts_base64_roundtrip(self) -> None:
        """Test ChainOfThoughts base64 serialization roundtrip."""
        original = ChainOfThoughts(
            reasoning="Complex reasoning with multiple steps and detailed analysis", confidence=0.88, importance=0.9
        )

        base64_str = original.to_base64()
        loaded = cast(ChainOfThoughts, ChainOfThoughts.from_base64(base64_str))

        assert loaded.reasoning == original.reasoning
        assert loaded.confidence == original.confidence
        assert loaded is not original  # Different instances
