"""Tests for the Result type implementation."""

import pytest

from blockether_foundation.errors import FoundationBaseError
from blockether_foundation.result import Result, ResultError


class TestResultError:
    """Test cases for ResultError class."""

    @pytest.mark.unit
    def test_result_error_creation_and_message(self):
        """Test ResultError constructor and message handling."""
        error = ResultError("Test error message")
        assert "Test error message" in str(error)
        assert isinstance(error, FoundationBaseError)


class TestResult:
    """Test cases for Result class methods."""

    @pytest.mark.unit
    def test_result_ok_creation(self):
        """Test Result.Ok constructor."""
        result = Result.Ok(42)
        assert result.is_ok()
        assert not result.is_err()
        assert result.unwrap() == 42

    @pytest.mark.unit
    def test_result_err_creation(self):
        """Test Result.Err constructor."""
        error = ResultError("Test error")
        result = Result.Err(error)
        assert not result.is_ok()
        assert result.is_err()
        assert result.unwrap_err() == error

    @pytest.mark.unit
    def test_post_init_validation_ok_with_error(self):
        """Test __post_init__ raises error for Ok result with error."""
        with pytest.raises(ResultError, match="Ok result cannot have an error"):
            Result(_ok=42, _error=ResultError("error"), _is_ok=True)

    @pytest.mark.unit
    def test_post_init_validation_err_with_value(self):
        """Test __post_init__ raises error for Err result with value."""
        with pytest.raises(ResultError, match="Err result cannot have an ok value"):
            Result(_ok=42, _error=None, _is_ok=False)

    @pytest.mark.unit
    def test_post_init_validation_err_without_error(self):
        """Test __post_init__ raises error for Err result without error."""
        with pytest.raises(ResultError, match="Err result must have an error"):
            Result(_ok=None, _error=None, _is_ok=False)

    @pytest.mark.unit
    def test_is_ok_method(self):
        """Test is_ok method returns correct value."""
        ok_result = Result.Ok(42)
        err_result = Result.Err(ResultError("error"))

        assert ok_result.is_ok() is True
        assert err_result.is_ok() is False

    @pytest.mark.unit
    def test_is_err_method(self):
        """Test is_err method returns correct value."""
        ok_result = Result.Ok(42)
        err_result = Result.Err(ResultError("error"))

        assert ok_result.is_err() is False
        assert err_result.is_err() is True

    @pytest.mark.unit
    def test_unwrap_success(self):
        """Test unwrap on Ok result returns value."""
        result = Result.Ok(42)
        assert result.unwrap() == 42

    @pytest.mark.unit
    def test_unwrap_error_raises(self):
        """Test unwrap on Err result raises ResultError."""
        error = ResultError("test error")
        result = Result.Err(error)

        with pytest.raises(ResultError):
            result.unwrap()

    @pytest.mark.unit
    def test_unwrap_err_success(self):
        """Test unwrap_err on Err result returns error."""
        error = ResultError("test error")
        result = Result.Err(error)
        assert result.unwrap_err() == error

    @pytest.mark.unit
    def test_unwrap_err_on_ok_raises(self):
        """Test unwrap_err on Ok result raises ResultError."""
        result = Result.Ok(42)

        with pytest.raises(ResultError, match="Called unwrap_err\\(\\) on an Ok value: 42"):
            result.unwrap_err()

    @pytest.mark.unit
    def test_unwrap_or_on_ok(self):
        """Test unwrap_or returns value on Ok result."""
        result = Result.Ok(42)
        assert result.unwrap_or(0) == 42

    @pytest.mark.unit
    def test_unwrap_or_on_err(self):
        """Test unwrap_or returns default on Err result."""
        result = Result.Err(ResultError("error"))
        assert result.unwrap_or(0) == 0

    @pytest.mark.unit
    def test_unwrap_or_else_on_ok(self):
        """Test unwrap_or_else returns value on Ok result."""
        result = Result.Ok(42)
        callback = lambda error: 0
        assert result.unwrap_or_else(callback) == 42

    @pytest.mark.unit
    def test_unwrap_or_else_on_err(self):
        """Test unwrap_or_else calls callback on Err result."""
        error = ResultError("test error")
        result = Result.Err(error)
        callback = lambda e: f"handled: {e}"
        result_str = result.unwrap_or_else(callback)
        assert "handled:" in result_str
        assert "test error" in result_str

    @pytest.mark.unit
    def test_expect_success(self):
        """Test expect returns value on Ok result."""
        result = Result.Ok(42)
        assert result.expect("Should not fail") == 42

    @pytest.mark.unit
    def test_expect_on_err_raises_with_custom_message(self):
        """Test expect raises with custom message on Err result."""
        error = ResultError("original error")
        result = Result.Err(error)

        with pytest.raises(ResultError):
            result.expect("Custom message")

    @pytest.mark.unit
    def test_map_on_ok(self):
        """Test map transforms Ok value."""
        result = Result.Ok(2)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_ok()
        assert mapped.unwrap() == 4

    @pytest.mark.unit
    def test_map_on_err(self):
        """Test map leaves Err unchanged."""
        error = ResultError("error")
        result = Result.Err(error)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err()
        assert mapped.unwrap_err() == error

    @pytest.mark.unit
    def test_map_err_on_ok(self):
        """Test map_err leaves Ok unchanged."""
        result = Result.Ok(42)
        mapped = result.map_err(lambda e: FoundationBaseError(f"wrapped: {e}"))
        assert mapped.is_ok()
        assert mapped.unwrap() == 42

    @pytest.mark.unit
    def test_map_err_on_err(self):
        """Test map_err transforms Err value."""
        error = ResultError("original")
        result = Result.Err(error)
        mapped = result.map_err(lambda e: FoundationBaseError(f"wrapped: {e}"))
        assert mapped.is_err()
        mapped_err_str = str(mapped.unwrap_err())
        assert "wrapped:" in mapped_err_str
        assert "original" in mapped_err_str

    @pytest.mark.unit
    def test_and_then_success(self):
        """Test and_then chains Result-producing operations."""

        def divide(x: int) -> Result[int, ResultError]:
            if x == 0:
                return Result.Err(ResultError("division by zero"))
            return Result.Ok(10 // x)

        result = Result.Ok(2)
        chained = result.and_then(divide)
        assert chained.is_ok()
        assert chained.unwrap() == 5

    @pytest.mark.unit
    def test_and_then_error(self):
        """Test and_then propagates error on Err result."""

        def divide(x: int) -> Result[int, ResultError]:
            return Result.Ok(10 // x)

        error = ResultError("initial error")
        result = Result.Err(error)
        chained = result.and_then(divide)
        assert chained.is_err()
        assert chained.unwrap_err() == error

    @pytest.mark.unit
    def test_and_then_err_from_callback(self):
        """Test and_then returns Err from callback."""

        def divide(x: int) -> Result[int, ResultError]:
            if x == 0:
                return Result.Err(ResultError("division by zero"))
            return Result.Ok(10 // x)

        result = Result.Ok(0)
        chained = result.and_then(divide)
        assert chained.is_err()
        chained_err_str = str(chained.unwrap_err())
        assert "division by zero" in chained_err_str

    @pytest.mark.unit
    def test_or_else_on_ok(self):
        """Test or_else returns original Ok result."""
        result = Result.Ok(42)
        fallback = lambda e: Result.Ok(0)
        final = result.or_else(fallback)
        assert final.is_ok()
        assert final.unwrap() == 42

    @pytest.mark.unit
    def test_or_else_on_err(self):
        """Test or_else returns fallback Result on Err."""
        error = ResultError("error")
        result = Result.Err(error)
        fallback = lambda e: Result.Ok(0)
        final = result.or_else(fallback)
        assert final.is_ok()
        assert final.unwrap() == 0

    @pytest.mark.unit
    def test_or_else_propagates_err_from_fallback(self):
        """Test or_else propagates Err from fallback function."""
        error = ResultError("error")
        result = Result.Err(error)
        fallback = lambda e: Result.Err(ResultError("fallback error"))
        final = result.or_else(fallback)
        assert final.is_err()
        final_err_str = str(final.unwrap_err())
        assert "fallback error" in final_err_str

    @pytest.mark.unit
    def test_repr_ok(self):
        """Test string representation of Ok result."""
        result = Result.Ok(42)
        assert repr(result) == "Result.Ok(42)"

    @pytest.mark.unit
    def test_repr_err(self):
        """Test string representation of Err result."""
        error = ResultError("test error")
        result = Result.Err(error)
        assert "Result.Err(" in repr(result)
        assert "test error" in repr(result)

    @pytest.mark.unit
    def test_complex_chaining(self):
        """Test complex method chaining scenarios."""

        def parse_int(s: str) -> Result[int, ResultError]:
            try:
                return Result.Ok(int(s))
            except ValueError:
                return Result.Err(ResultError(f"Invalid number: {s}"))

        def divide(x: int) -> Result[float, ResultError]:
            if x == 0:
                return Result.Err(ResultError("Division by zero"))
            return Result.Ok(100.0 / x)

        # Success chain
        result = Result.Ok("50").and_then(parse_int).and_then(divide).map(lambda x: round(x, 2))
        assert result.is_ok()
        assert result.unwrap() == 2.0

        # Error chain
        result = Result.Ok("invalid").and_then(parse_int).and_then(divide)
        assert result.is_err()
        result_err_str = str(result.unwrap_err())
        assert "Invalid number" in result_err_str

    @pytest.mark.unit
    def test_type_violation_detection(self):
        """Test that Result properly detects type violations in __post_init__."""
        # Test creating Result directly with invalid state
        with pytest.raises(ResultError):
            Result(_ok=42, _error=ResultError("error"), _is_ok=True)

        with pytest.raises(ResultError):
            Result(_ok=42, _error=None, _is_ok=False)

        with pytest.raises(ResultError):
            Result(_ok=None, _error=None, _is_ok=False)

    @pytest.mark.unit
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with None values
        result = Result.Ok(None)
        assert result.is_ok()
        assert result.unwrap() is None

        # Test with complex objects
        class CustomObject:
            def __init__(self, value: str):
                self.value = value

        obj = CustomObject("test")
        result = Result.Ok(obj)
        assert result.is_ok()
        assert result.unwrap().value == "test"

        # Test chaining with different types
        result = Result.Ok("hello").map(len).map(lambda x: x * 2).map(str)
        assert result.unwrap() == "10"
