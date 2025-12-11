"""Tests for the models module."""

import json
import pickle
import tempfile
import pytest

from blockether_foundation.models import BaseModelSerializable, ChainOfThoughts


# Define test models at module level so they can be pickled
class DemoModel(BaseModelSerializable):
    name: str
    value: int


class ComplexDemoModel(BaseModelSerializable):
    data: dict
    items: list


class ModelWithObjDemo(BaseModelSerializable):
    obj: object


class TestBaseModelSerializable:
    """Test cases for BaseModelSerializable."""

    @pytest.mark.unit
    def test_json_serialization(self) -> None:
        """Test JSON serialization methods."""
        # Create test instance
        model = DemoModel(name="test", value=42)

        # Test to_json_string
        json_str = model.to_json_string()
        assert "test" in json_str
        assert "42" in json_str

        # Test from_json_string
        loaded = DemoModel.from_json_string(json_str)
        assert loaded.name == "test"
        assert loaded.value == 42

    @pytest.mark.unit
    def test_json_file_serialization(self) -> None:
        """Test JSON file serialization methods."""
        model = DemoModel(name="test", value=42)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            # Test to_json_file
            model.to_json_file(filepath)

            # Test from_json_file
            loaded = DemoModel.from_json_file(filepath)
            assert loaded.name == "test"
            assert loaded.value == 42
        finally:
            import os
            os.unlink(filepath)

    @pytest.mark.unit
    def test_pickle_serialization(self) -> None:
        """Test pickle serialization methods."""
        model = DemoModel(name="test", value=42)

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            # Test to_pickle_file
            model.to_pickle_file(filepath)

            # Test from_pickle_file
            loaded = DemoModel.from_pickle_file(filepath)
            assert loaded.name == "test"
            assert loaded.value == 42
        finally:
            import os
            os.unlink(filepath)

    @pytest.mark.unit
    def test_base64_serialization(self) -> None:
        """Test base64 serialization methods."""
        model = DemoModel(name="test", value=42)

        # Test to_base64
        base64_str = model.to_base64()
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

        # Test from_base64
        loaded = DemoModel.from_base64(base64_str)
        assert loaded.name == "test"
        assert loaded.value == 42

    @pytest.mark.unit
    def test_json_string_roundtrip_with_complex_data(self) -> None:
        """Test JSON serialization with complex nested data."""
        model = ComplexDemoModel(
            data={"key": "value", "nested": {"x": 1, "y": 2}},
            items=[1, 2, 3, {"a": "b"}]
        )

        # Test roundtrip
        json_str = model.to_json_string()
        loaded = ComplexDemoModel.from_json_string(json_str)
        assert loaded.data == model.data
        assert loaded.items == model.items

    @pytest.mark.unit
    def test_pickle_roundtrip_with_object(self) -> None:
        """Test pickle serialization with object attributes."""
        # Create with a simple object
        test_obj = {"complex": "object", "number": 123}
        model = ModelWithObjDemo(obj=test_obj)

        # Test base64 roundtrip
        base64_str = model.to_base64()
        loaded = ModelWithObjDemo.from_base64(base64_str)
        assert loaded.obj == test_obj


class TestChainOfThoughts:
    """Test cases for ChainOfThoughts model."""

    @pytest.mark.unit
    def test_chain_of_thoughts_creation(self) -> None:
        """Test creating a ChainOfThoughts instance."""
        thoughts = ChainOfThoughts(
            reasoning="This is my reasoning process",
            confidence=0.85
        )

        assert thoughts.reasoning == "This is my reasoning process"
        assert thoughts.confidence == 0.85

    @pytest.mark.unit
    def test_chain_of_thoughts_validation(self) -> None:
        """Test ChainOfThoughts validation."""
        # Test valid confidence values
        thoughts1 = ChainOfThoughts(reasoning="test", confidence=0.0)
        assert thoughts1.confidence == 0.0

        thoughts2 = ChainOfThoughts(reasoning="test", confidence=1.0)
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
            confidence=0.92
        )

        # Test JSON serialization
        json_str = thoughts.to_json_string()
        loaded = ChainOfThoughts.from_json_string(json_str)
        assert loaded.reasoning == thoughts.reasoning
        assert loaded.confidence == thoughts.confidence

    @pytest.mark.unit
    def test_chain_of_thoughts_inheritance(self) -> None:
        """Test that ChainOfThoughts inherits BaseModelSerializable."""
        thoughts = ChainOfThoughts(
            reasoning="My reasoning",
            confidence=0.75
        )

        # Should have BaseModelSerializable methods
        assert hasattr(thoughts, 'to_json_string')
        assert hasattr(thoughts, 'to_base64')
        assert hasattr(thoughts, 'to_pickle_file')

    @pytest.mark.unit
    def test_chain_of_thoughts_base64_roundtrip(self) -> None:
        """Test ChainOfThoughts base64 serialization roundtrip."""
        original = ChainOfThoughts(
            reasoning="Complex reasoning with multiple steps and detailed analysis",
            confidence=0.88
        )

        base64_str = original.to_base64()
        loaded = ChainOfThoughts.from_base64(base64_str)

        assert loaded.reasoning == original.reasoning
        assert loaded.confidence == original.confidence
        assert loaded is not original  # Different instances