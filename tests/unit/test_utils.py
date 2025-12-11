"""Tests for utility functions."""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from blockether_foundation.utils import none_invariant

# Test constants
TEST_STRING_VALUE = "test_value"
TEST_NUMBER_VALUE = 42
TEST_OBJECT_VALUE = {"key": "value"}
TEST_EMPTY_STRING = ""
TEST_EMPTY_LIST: list[Any] = []
TEST_BOOLEAN_FALSE = False
TEST_ZERO_VALUE = 0
TEST_SUM_VALUE = 15
TEST_INDEX_VALUE = "b"
TEST_TYPED_STRING_VALUE = "typed_value"
TEST_TYPED_INT_VALUE = 123
TEST_NESTED_VALUE = "nested"
TEST_FIRST_VALUE = "first"
TEST_SECOND_VALUE = "second"
TEST_THIRD_VALUE = 42

# Test data constants
TEST_LIST_DATA = [1, 2, 3, 4, 5]
TEST_INDEXED_LIST = ["a", "b", "c"]
SUCCESS_INDEX = 1
FAILURE_INDEX = 10
CUSTOM_MODULE_NAME = "custom_module.test"


class TestNoneInvariant:
    """Test cases for none_invariant function."""

    @pytest.mark.unit
    def test_none_invariant_success_with_string(self: Any) -> None:
        """Test none_invariant returns value when condition returns non-None string."""
        result = none_invariant(lambda: TEST_STRING_VALUE, "Value should not be None")
        assert result == TEST_STRING_VALUE

    @pytest.mark.unit
    def test_none_invariant_success_with_number(self) -> None:
        """Test none_invariant returns value when condition returns non-None number."""
        result = none_invariant(lambda: TEST_NUMBER_VALUE, "Number should not be None")
        assert result == TEST_NUMBER_VALUE

    @pytest.mark.unit
    def test_none_invariant_success_with_object(self) -> None:
        """Test none_invariant returns value when condition returns non-None object."""
        result = none_invariant(lambda: TEST_OBJECT_VALUE, "Object should not be None")
        assert result == TEST_OBJECT_VALUE

    @pytest.mark.unit
    def test_none_invariant_success_with_empty_string(self) -> None:
        """Test none_invariant accepts empty string as valid non-None value."""
        result = none_invariant(lambda: TEST_EMPTY_STRING, "Empty string is valid")
        assert result == TEST_EMPTY_STRING

    @pytest.mark.unit
    def test_none_invariant_success_with_empty_list(self) -> None:
        """Test none_invariant accepts empty list as valid non-None value."""
        result = none_invariant(lambda: TEST_EMPTY_LIST, "Empty list is valid")
        assert result == TEST_EMPTY_LIST

    @pytest.mark.unit
    def test_none_invariant_success_with_false_value(self) -> None:
        """Test none_invariant accepts False as valid non-None value."""
        result = none_invariant(lambda: TEST_BOOLEAN_FALSE, "False is valid")
        assert result is TEST_BOOLEAN_FALSE

    @pytest.mark.unit
    def test_none_invariant_success_with_zero_value(self) -> None:
        """Test none_invariant accepts 0 as valid non-None value."""
        result = none_invariant(lambda: TEST_ZERO_VALUE, "Zero is valid")
        assert result == TEST_ZERO_VALUE

    @pytest.mark.unit
    def test_none_invariant_fails_when_condition_returns_none(self) -> None:
        """Test none_invariant raises AssertionError when condition returns None."""
        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Value should not be None")

        error_message = str(exc_info.value)
        assert "Value should not be None" in error_message
        assert "test_utils" in error_message  # Module name should be included

    @pytest.mark.unit
    def test_none_invariant_includes_module_name_in_error(self) -> None:
        """Test none_invariant includes caller module name in error message."""
        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Custom error message")

        error_message = str(exc_info.value)
        assert "[test_utils]" in error_message
        assert "Custom error message" in error_message

    @pytest.mark.unit
    def test_none_invariant_with_complex_condition(self) -> None:
        """Test none_invariant with condition that performs computation."""

        def compute_value() -> int:
            data = TEST_LIST_DATA
            return sum(data) if data else None

        result = none_invariant(compute_value, "Computation should succeed")
        assert result == TEST_SUM_VALUE

    @pytest.mark.unit
    def test_none_invariant_with_condition_taking_arguments(self) -> None:
        """Test none_invariant with condition callable that takes arguments."""

        def get_value_or_none(data: list[str], index: int) -> str | None:
            try:
                return data[index]
            except IndexError:
                return None

        # Success case
        result = none_invariant(
            lambda: get_value_or_none(TEST_INDEXED_LIST, SUCCESS_INDEX), "Value should exist at index 1"
        )
        assert result == TEST_INDEX_VALUE

        # Failure case
        with pytest.raises(AssertionError):
            none_invariant(
                lambda: get_value_or_none(TEST_INDEXED_LIST, FAILURE_INDEX), "Value should not be None"
            )

    @pytest.mark.unit
    @patch("inspect.currentframe")
    @patch("inspect.getmodule")
    def test_none_invariant_with_custom_module_name(
        self, mock_getmodule: Mock, mock_currentframe: Mock
    ) -> None:
        """Test none_invariant with mocked module inspection."""
        # Setup mocks
        mock_frame = Mock()
        mock_currentframe.return_value = mock_frame
        mock_frame.f_back = Mock()

        mock_module = Mock()
        mock_module.__name__ = CUSTOM_MODULE_NAME
        mock_getmodule.return_value = mock_module

        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Test message")

        error_message = str(exc_info.value)
        assert f"[{CUSTOM_MODULE_NAME}]" in error_message
        assert "Test message" in error_message

    @pytest.mark.unit
    @patch("inspect.currentframe")
    def test_none_invariant_handles_none_caller_frame(self, mock_currentframe: Mock) -> None:
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
    def test_none_invariant_handles_none_module(
        self, mock_getmodule: Mock, mock_currentframe: Mock
    ) -> None:
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
    def test_none_invariant_type_hints(self) -> None:
        """Test none_invariant preserves type hints correctly."""
        # Test with string return type
        result: str = none_invariant(lambda: TEST_TYPED_STRING_VALUE, "Should return string")
        assert result == TEST_TYPED_STRING_VALUE
        assert isinstance(result, str)

        # Test with int return type
        result_int: int = none_invariant(lambda: TEST_TYPED_INT_VALUE, "Should return int")
        assert result_int == TEST_TYPED_INT_VALUE
        assert isinstance(result_int, int)

    @pytest.mark.unit
    def test_none_invariant_nested_calls(self) -> None:
        """Test none_invariant works correctly in nested scenarios."""

        def get_nested_value() -> str:
            return none_invariant(lambda: TEST_NESTED_VALUE, "Inner value should not be None")

        result = none_invariant(get_nested_value, "Outer call should succeed")
        assert result == TEST_NESTED_VALUE

    @pytest.mark.unit
    def test_none_invariant_with_exception_in_condition(self) -> None:
        """Test none_invariant propagates exceptions from condition."""

        def failing_condition() -> None:
            raise ValueError("Condition execution failed")

        with pytest.raises(ValueError, match="Condition execution failed"):
            none_invariant(failing_condition, "This should not be reached")

    @pytest.mark.unit
    def test_none_invariant_multiple_assertions_in_same_test(self) -> None:
        """Test multiple none_invariant calls work correctly in same test."""
        # All successful calls
        result1 = none_invariant(lambda: TEST_FIRST_VALUE, "First should not be None")
        result2 = none_invariant(lambda: TEST_SECOND_VALUE, "Second should not be None")
        result3 = none_invariant(lambda: TEST_THIRD_VALUE, "Third should not be None")

        assert result1 == TEST_FIRST_VALUE
        assert result2 == TEST_SECOND_VALUE
        assert result3 == TEST_THIRD_VALUE

        # Failed call should still include proper context
        with pytest.raises(AssertionError) as exc_info:
            none_invariant(lambda: None, "Fourth should not be None")

        error_message = str(exc_info.value)
        assert "Fourth should not be None" in error_message
        assert "test_utils" in error_message
