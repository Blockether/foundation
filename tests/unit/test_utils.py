"""Tests for utility functions."""

from unittest.mock import Mock, patch

import pytest

from blockether_foundation.utils import none_invariant


class TestNoneInvariant:
    """Test cases for none_invariant function."""

    @pytest.mark.unit
    def test_none_invariant_success_with_string(self):
        """Test none_invariant returns value when condition returns non-None string."""
        result = none_invariant(lambda: "test_value", "Value should not be None")
        assert result == "test_value"

    @pytest.mark.unit
    def test_none_invariant_success_with_number(self):
        """Test none_invariant returns value when condition returns non-None number."""
        result = none_invariant(lambda: 42, "Number should not be None")
        assert result == 42

    @pytest.mark.unit
    def test_none_invariant_success_with_object(self):
        """Test none_invariant returns value when condition returns non-None object."""
        test_obj = {"key": "value"}
        result = none_invariant(lambda: test_obj, "Object should not be None")
        assert result == test_obj

    @pytest.mark.unit
    def test_none_invariant_success_with_empty_string(self):
        """Test none_invariant accepts empty string as valid non-None value."""
        result = none_invariant(lambda: "", "Empty string is valid")
        assert result == ""

    @pytest.mark.unit
    def test_none_invariant_success_with_empty_list(self):
        """Test none_invariant accepts empty list as valid non-None value."""
        result = none_invariant(lambda: [], "Empty list is valid")
        assert result == []

    @pytest.mark.unit
    def test_none_invariant_success_with_false_value(self):
        """Test none_invariant accepts False as valid non-None value."""
        result = none_invariant(lambda: False, "False is valid")
        assert result is False

    @pytest.mark.unit
    def test_none_invariant_success_with_zero_value(self):
        """Test none_invariant accepts 0 as valid non-None value."""
        result = none_invariant(lambda: 0, "Zero is valid")
        assert result == 0

    @pytest.mark.unit
    def test_none_invariant_fails_when_condition_returns_none(self):
        """Test none_invariant raises AssertionError when condition returns None."""
        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Value should not be None")

        error_message = str(exc_info.value)
        assert "Value should not be None" in error_message
        assert "test_utils" in error_message  # Module name should be included

    @pytest.mark.unit
    def test_none_invariant_includes_module_name_in_error(self):
        """Test none_invariant includes caller module name in error message."""
        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Custom error message")

        error_message = str(exc_info.value)
        assert "[test_utils]" in error_message
        assert "Custom error message" in error_message

    @pytest.mark.unit
    def test_none_invariant_with_complex_condition(self):
        """Test none_invariant with condition that performs computation."""

        def compute_value():
            data = [1, 2, 3, 4, 5]
            return sum(data) if data else None

        result = none_invariant(compute_value, "Computation should succeed")
        assert result == 15

    @pytest.mark.unit
    def test_none_invariant_with_condition_taking_arguments(self):
        """Test none_invariant with condition callable that takes arguments."""

        def get_value_or_none(data: list[str], index: int):
            try:
                return data[index]
            except IndexError:
                return None

        # Success case
        result = none_invariant(
            lambda: get_value_or_none(["a", "b", "c"], 1), "Value should exist at index 1"
        )
        assert result == "b"

        # Failure case
        with pytest.raises(AssertionError):
            none_invariant(
                lambda: get_value_or_none(["a", "b", "c"], 10), "Value should not be None"
            )

    @pytest.mark.unit
    @patch("inspect.currentframe")
    @patch("inspect.getmodule")
    def test_none_invariant_with_custom_module_name(self, mock_getmodule, mock_currentframe):
        """Test none_invariant with mocked module inspection."""
        # Setup mocks
        mock_frame = Mock()
        mock_currentframe.return_value = mock_frame
        mock_frame.f_back = Mock()

        mock_module = Mock()
        mock_module.__name__ = "custom_module.test"
        mock_getmodule.return_value = mock_module

        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Test message")

        error_message = str(exc_info.value)
        assert "[custom_module.test]" in error_message
        assert "Test message" in error_message

    @pytest.mark.unit
    @patch("inspect.currentframe")
    def test_none_invariant_handles_none_caller_frame(self, mock_currentframe):
        """Test none_invariant handles case when currentframe returns None."""
        mock_currentframe.return_value = None

        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Test message")

        error_message = str(exc_info.value)
        assert "[unknown]" in error_message
        assert "Test message" in error_message

    @pytest.mark.unit
    @patch("inspect.currentframe")
    @patch("inspect.getmodule")
    def test_none_invariant_handles_none_module(self, mock_getmodule, mock_currentframe):
        """Test none_invariant handles case when getmodule returns None."""
        mock_frame = Mock()
        mock_currentframe.return_value = mock_frame
        mock_frame.f_back = Mock()
        mock_getmodule.return_value = None

        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Test message")

        error_message = str(exc_info.value)
        assert "[unknown]" in error_message
        assert "Test message" in error_message

    @pytest.mark.unit
    def test_none_invariant_type_hints(self):
        """Test none_invariant preserves type hints correctly."""
        # Test with string return type
        result: str = none_invariant(lambda: "typed_value", "Should return string")
        assert result == "typed_value"
        assert isinstance(result, str)

        # Test with int return type
        result_int: int = none_invariant(lambda: 123, "Should return int")
        assert result_int == 123
        assert isinstance(result_int, int)

    @pytest.mark.unit
    def test_none_invariant_nested_calls(self):
        """Test none_invariant works correctly in nested scenarios."""

        def get_nested_value():
            return none_invariant(lambda: "nested", "Inner value should not be None")

        result = none_invariant(get_nested_value, "Outer call should succeed")
        assert result == "nested"

    @pytest.mark.unit
    def test_none_invariant_with_exception_in_condition(self):
        """Test none_invariant propagates exceptions from condition."""

        def failing_condition():
            raise ValueError("Condition execution failed")

        with pytest.raises(ValueError, match="Condition execution failed"):
            none_invariant(failing_condition, "This should not be reached")

    @pytest.mark.unit
    def test_none_invariant_multiple_assertions_in_same_test(self):
        """Test multiple none_invariant calls work correctly in same test."""
        # All successful calls
        result1 = none_invariant(lambda: "first", "First should not be None")
        result2 = none_invariant(lambda: "second", "Second should not be None")
        result3 = none_invariant(lambda: 42, "Third should not be None")

        assert result1 == "first"
        assert result2 == "second"
        assert result3 == 42

        # Failed call should still include proper context
        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Fourth should not be None")

        error_message = str(exc_info.value)
        assert "Fourth should not be None" in error_message
        assert "test_utils" in error_message
