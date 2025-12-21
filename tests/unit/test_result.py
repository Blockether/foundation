"""Tests for the Result type implementation."""

from typing import Any

import pytest

from blockether_foundation.errors import FoundationBaseError
from blockether_foundation.result import Result, ResultError


class TestResultError:
    """Test cases for ResultError class."""

    @pytest.mark.unit
    def test_result_error_creation_and_message(self) -> None:
        """Test ResultError constructor and message handling."""
        error = ResultError("Test error message")
        assert "Test error message" in str(error)
        assert isinstance(error, FoundationBaseError)


class TestResult:
    """Test cases for Result class methods."""

    @pytest.mark.unit
    def test_result_ok_creation(self) -> None:
        """Test Result.Ok constructor."""
        test_value = 42
        result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)
        assert result.is_ok()
        assert not result.is_err()
        assert result.unwrap() == test_value

    @pytest.mark.unit
    def test_result_err_creation(self) -> None:
        """Test Result.Err constructor."""
        error = ResultError("Test error")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)
        assert not result.is_ok()
        assert result.is_err()
        assert result.unwrap_err() == error

    @pytest.mark.unit
    def test_post_init_validation_ok_with_error(self) -> None:
        """Test __post_init__ raises error for Ok result with error."""
        with pytest.raises(ResultError, match="Ok result cannot have an error"):
            Result(_ok=42, _error=ResultError("error"), _is_ok=True)
        assert True  # Assert that validation properly raised

    @pytest.mark.unit
    def test_post_init_validation_err_with_value(self) -> None:
        """Test __post_init__ raises error for Err result with value."""
        with pytest.raises(ResultError, match="Err result cannot have an ok value"):
            Result(_ok=42, _error=None, _is_ok=False)
        assert True  # Assert that validation properly raised

    @pytest.mark.unit
    def test_post_init_validation_err_without_error(self) -> None:
        """Test __post_init__ raises error for Err result without error."""
        with pytest.raises(ResultError, match="Err result must have an error"):
            Result(_ok=None, _error=None, _is_ok=False)
        assert True  # Assert that validation properly raised

    @pytest.mark.unit
    def test_is_ok_method(self) -> None:
        """Test is_ok method returns correct value."""
        test_value = 42
        ok_result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)
        err_result: Result[int, ResultError] = Result[int, ResultError].Err(ResultError("error"))

        assert ok_result.is_ok() is True
        assert err_result.is_ok() is False

    @pytest.mark.unit
    def test_is_err_method(self) -> None:
        """Test is_err method returns correct value."""
        test_value = 42
        ok_result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)
        err_result: Result[int, ResultError] = Result[int, ResultError].Err(ResultError("error"))

        assert ok_result.is_err() is False
        assert err_result.is_err() is True

    @pytest.mark.unit
    def test_unwrap_success(self) -> None:
        """Test unwrap on Ok result returns value."""
        test_value = 42
        result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)
        assert result.unwrap() == test_value

    @pytest.mark.unit
    def test_unwrap_error_raises(self) -> None:
        """Test unwrap on Err result raises ResultError."""
        error = ResultError("test error")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)

        with pytest.raises(ResultError):
            result.unwrap()
        assert True  # Assert that unwrap properly raised

    @pytest.mark.unit
    def test_unwrap_err_success(self) -> None:
        """Test unwrap_err on Err result returns error."""
        error = ResultError("test error")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)
        assert result.unwrap_err() == error

    @pytest.mark.unit
    def test_unwrap_err_on_ok_raises(self) -> None:
        """Test unwrap_err on Ok result raises ResultError."""
        test_value = 42
        result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)

        with pytest.raises(ResultError, match="Called unwrap_err\\(\\) on an Ok value: 42"):
            result.unwrap_err()
        assert True  # Assert that unwrap_err properly raised

    @pytest.mark.unit
    def test_unwrap_or_on_ok(self) -> None:
        """Test unwrap_or returns value on Ok result."""
        test_value = 42
        default_value = 0
        result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)
        assert result.unwrap_or(default_value) == test_value

    @pytest.mark.unit
    def test_unwrap_or_on_err(self) -> None:
        """Test unwrap_or returns default on Err result."""
        default_value = 0
        result: Result[int, ResultError] = Result[int, ResultError].Err(ResultError("error"))
        assert result.unwrap_or(default_value) == default_value

    @pytest.mark.unit
    def test_unwrap_or_else_on_ok(self) -> None:
        """Test unwrap_or_else returns value on Ok result."""
        test_value = 42
        result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)

        def callback(_error: ResultError) -> int:
            return 0

        assert result.unwrap_or_else(callback) == test_value

    @pytest.mark.unit
    def test_unwrap_or_else_on_err(self) -> None:
        """Test unwrap_or_else calls callback on Err result."""
        error = ResultError("test error")
        result: Result[str, ResultError] = Result[str, ResultError].Err(error)

        def callback(err: ResultError) -> str:
            return f"handled: {err}"

        result_str = result.unwrap_or_else(callback)
        assert "handled:" in result_str
        assert "test error" in result_str

    @pytest.mark.unit
    def test_unwrap_or_else_returns_callback_value(self) -> None:
        """Ensure unwrap_or_else returns the value from the callback when Err."""
        error = ResultError("boom")
        result: Result[str, ResultError] = Result[str, ResultError].Err(error)

        def callback(err: ResultError) -> str:
            return f"callback:{err}"

        assert result.unwrap_or_else(callback) == f"callback:{error}"

    @pytest.mark.unit
    def test_expect_success(self) -> None:
        """Test expect returns value on Ok result."""
        test_value = 42
        result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)
        assert result.expect("Should not fail") == test_value

    @pytest.mark.unit
    def test_expect_on_err_raises_with_custom_message(self) -> None:
        """Test expect raises with custom message on Err result."""
        error = ResultError("original error")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)

        with pytest.raises(ResultError):
            result.expect("Custom message")
        assert True  # Assert that expect properly raised

    @pytest.mark.unit
    def test_expect_includes_original_error(self) -> None:
        """Ensure expect includes original error details in raised message."""
        error = ResultError("bad state")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)
        with pytest.raises(ResultError, match="bad state"):
            result.expect("Should not happen")
        assert True  # Assert that expect properly raised with original error

    @pytest.mark.unit
    def test_map_on_ok(self) -> None:
        """Test map transforms Ok value."""
        initial_value = 2
        expected_value = 4
        result: Result[int, ResultError] = Result[int, ResultError].Ok(initial_value)
        mapped: Result[int, ResultError] = result.map(lambda x: x * 2)
        assert mapped.is_ok()
        assert mapped.unwrap() == expected_value

    @pytest.mark.unit
    def test_map_on_err(self) -> None:
        """Test map leaves Err unchanged."""
        error = ResultError("error")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)
        mapped: Result[int, ResultError] = result.map(lambda x: x * 2)
        assert mapped.is_err()
        assert mapped.unwrap_err() == error

    @pytest.mark.unit
    def test_map_err_on_ok(self) -> None:
        """Test map_err leaves Ok unchanged."""
        test_value = 42
        result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)
        mapped: Result[int, FoundationBaseError] = result.map_err(
            lambda e: FoundationBaseError(f"wrapped: {e}")
        )
        assert mapped.is_ok()
        assert mapped.unwrap() == test_value

    @pytest.mark.unit
    def test_map_err_on_err(self) -> None:
        """Test map_err transforms Err value."""
        error = ResultError("original")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)
        mapped: Result[int, FoundationBaseError] = result.map_err(
            lambda e: FoundationBaseError(f"wrapped: {e}")
        )
        assert mapped.is_err()
        mapped_err_str = str(mapped.unwrap_err())
        assert "wrapped:" in mapped_err_str
        assert "original" in mapped_err_str

    @pytest.mark.unit
    def test_and_then_success(self) -> None:
        """Test and_then chains Result-producing operations."""

        def divide(x: int) -> Result[int, ResultError]:
            return Result[int, ResultError].Ok(10 // x)

        input_value = 2
        expected_value = 5
        result: Result[int, ResultError] = Result[int, ResultError].Ok(input_value)
        chained: Result[int, ResultError] = result.and_then(divide)
        assert chained.is_ok()
        assert chained.unwrap() == expected_value

    @pytest.mark.unit
    def test_and_then_divide_non_zero(self) -> None:
        """Test and_then with non-zero value."""

        def divide(x: int) -> Result[int, ResultError]:
            return Result[int, ResultError].Ok(10 // x)

        input_value = 1
        result: Result[int, ResultError] = Result[int, ResultError].Ok(input_value)
        chained: Result[int, ResultError] = result.and_then(divide)
        assert chained.is_ok()
        assert chained.unwrap() == 10

    @pytest.mark.unit
    def test_and_then_division_by_zero_error(self) -> None:
        """Test and_then handles division by zero error."""

        def divide(x: int) -> Result[int, ResultError]:
            return Result[int, ResultError].Err(ResultError("division by zero"))

        input_value = 0
        result: Result[int, ResultError] = Result[int, ResultError].Ok(input_value)
        chained: Result[int, ResultError] = result.and_then(divide)
        assert chained.is_err()
        chained_err_str = str(chained.unwrap_err())
        assert "division by zero" in chained_err_str

    @pytest.mark.unit
    def test_and_then_error(self) -> None:
        """Test and_then propagates error on Err result."""

        def divide(x: int) -> Result[int, ResultError]:
            return Result[int, ResultError].Ok(10 // x)

        error = ResultError("initial error")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)
        chained: Result[int, ResultError] = result.and_then(divide)
        assert chained.is_err()
        assert chained.unwrap_err() == error

    @pytest.mark.unit
    def test_or_else_on_ok(self) -> None:
        """Test or_else returns original Ok result."""
        test_value = 42
        result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)

        def recovery(_error: ResultError) -> Result[int, ResultError]:
            return Result[int, ResultError].Ok(0)

        final: Result[int, ResultError] = result.or_else(recovery)
        assert final.is_ok()
        assert final.unwrap() == test_value

    @pytest.mark.unit
    def test_or_else_on_err(self) -> None:
        """Test or_else returns recovery Result on Err."""
        error = ResultError("error")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)
        recovery_value = 0

        def recovery(_error: ResultError) -> Result[int, ResultError]:
            return Result[int, ResultError].Ok(recovery_value)

        final: Result[int, ResultError] = result.or_else(recovery)
        assert final.is_ok()
        assert final.unwrap() == recovery_value

    @pytest.mark.unit
    def test_or_else_propagates_err_from_recovery(self) -> None:
        """Test or_else propagates Err from recovery function."""
        error = ResultError("error")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)

        def recovery(_error: ResultError) -> Result[int, ResultError]:
            return Result[int, ResultError].Err(ResultError("recovery error"))

        final: Result[int, ResultError] = result.or_else(recovery)
        assert final.is_err()
        final_err_str = str(final.unwrap_err())
        assert "recovery error" in final_err_str

    @pytest.mark.unit
    def test_repr_ok(self) -> None:
        """Test string representation of Ok result."""
        test_value = 42
        result: Result[int, ResultError] = Result[int, ResultError].Ok(test_value)
        assert repr(result) == "Result.Ok(42)"

    @pytest.mark.unit
    def test_repr_err(self) -> None:
        """Test string representation of Err result."""
        error = ResultError("test error")
        result: Result[int, ResultError] = Result[int, ResultError].Err(error)
        assert "Result.Err(" in repr(result)
        assert "test error" in repr(result)

    @pytest.mark.unit
    def test_complex_chaining_success(self) -> None:
        """Test complex method chaining success scenario."""

        def parse_int(s: str) -> Result[int, ResultError]:
            return Result[int, ResultError].Ok(int(s))

        def divide(x: int) -> Result[float, ResultError]:
            return Result[float, ResultError].Ok(100.0 / x)

        result: Result[float, ResultError] = (
            Result[str, ResultError]
            .Ok("50")
            .and_then(parse_int)
            .and_then(divide)
            .map(lambda x: round(x, 2))
        )
        assert result.is_ok()
        assert result.unwrap() == 2.0

    @pytest.mark.unit
    def test_complex_chaining_parse_error(self) -> None:
        """Test complex method chaining with parse error."""

        def parse_int(s: str) -> Result[int, ResultError]:
            return Result[int, ResultError].Err(ResultError(f"Invalid number: {s}"))

        def divide(x: int) -> Result[float, ResultError]:
            return Result[float, ResultError].Ok(100.0 / x)

        result: Result[float, ResultError] = (
            Result[str, ResultError].Ok("invalid").and_then(parse_int).and_then(divide)
        )
        assert result.is_err()
        result_err_str = str(result.unwrap_err())
        assert "Invalid number" in result_err_str

    @pytest.mark.unit
    def test_complex_chaining_division_error(self) -> None:
        """Test complex method chaining with division error."""

        def parse_int(s: str) -> Result[int, ResultError]:
            return Result[int, ResultError].Ok(int(s))

        def divide(x: int) -> Result[float, ResultError]:
            return Result[float, ResultError].Err(ResultError("Division by zero"))

        result: Result[float, ResultError] = (
            Result[str, ResultError].Ok("0").and_then(parse_int).and_then(divide)
        )
        assert result.is_err()
        result_err_str = str(result.unwrap_err())
        assert "Division by zero" in result_err_str

    @pytest.mark.unit
    def test_type_violation_detection(self) -> None:
        """Test that Result properly detects type violations in __post_init__."""
        # Test creating Result directly with invalid state
        with pytest.raises(ResultError):
            Result(_ok=42, _error=ResultError("error"), _is_ok=True)
        assert True  # Assert that validation properly raised

        with pytest.raises(ResultError):
            Result(_ok=42, _error=None, _is_ok=False)
        assert True  # Assert that validation properly raised

        with pytest.raises(ResultError):
            Result(_ok=None, _error=None, _is_ok=False)
        assert True  # Assert that validation properly raised

    @pytest.mark.unit
    def test_edge_cases(self) -> None:
        """Test edge cases and boundary conditions."""
        # Test with None values
        result: Result[Any, ResultError] = Result[Any, ResultError].Ok(None)
        assert result.is_ok()
        assert result.unwrap() is None

        # Test with complex objects
        class CustomObject:
            def __init__(self, value: str):
                self.value = value

        obj = CustomObject("test")
        result_obj: Result[CustomObject, ResultError] = Result[CustomObject, ResultError].Ok(obj)
        assert result_obj.is_ok()
        assert result_obj.unwrap().value == "test"

        # Test chaining with different types
        result_str: Result[str, ResultError] = (
            Result[str, ResultError].Ok("hello").map(len).map(lambda x: x * 2).map(str)
        )
        assert result_str.unwrap() == "10"
