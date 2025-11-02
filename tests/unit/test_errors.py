"""Tests for Foundation error classes."""

import pytest

from blockether_foundation.errors import FoundationBaseError


class CustomTestError(FoundationBaseError):
    pass


@pytest.mark.unit
def test_error_having_auto_solidity_like_message():
    """Test error string format matches Solidity-like pattern."""
    error = CustomTestError("Test error occurred")
    assert str(error) == "test_errors.CustomTestError: Test error occurred"


@pytest.mark.unit
def test_error_with_details():
    """Test error string representation with details (covers lines 33-34)."""
    from pydantic import BaseModel

    class ErrorDetails(BaseModel):
        code: int
        field: str

    details = ErrorDetails(code=404, field="resource")
    error = CustomTestError("Test error", details=details)

    error_str = str(error)
    assert "test_errors.CustomTestError: Test error" in error_str
    assert "'code': 404" in error_str
    assert "'field': 'resource'" in error_str


@pytest.mark.unit
def test_error_inheritance():
    """Test that custom errors properly inherit from FoundationBaseError."""
    error = CustomTestError("Test error")
    assert isinstance(error, FoundationBaseError)


@pytest.mark.unit
def test_error_without_details():
    """Test error with no additional details."""
    error = CustomTestError("Simple error")
    assert str(error) == "test_errors.CustomTestError: Simple error"


@pytest.mark.unit
def test_error_with_none_details():
    """Test error with None details."""
    error = CustomTestError("Test error", details=None)
    assert str(error) == "test_errors.CustomTestError: Test error"