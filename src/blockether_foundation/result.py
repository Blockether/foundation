"""Rust-like Result type for robust error handling.

This module provides a Result type similar to Rust's Result<T, E> that forces
explicit handling of both success and failure cases, preventing silent errors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .errors import FoundationBaseError

T = TypeVar("T")
U = TypeVar("U")
F = TypeVar("F", bound=FoundationBaseError)
E = TypeVar("E", bound=FoundationBaseError)


class ResultError(FoundationBaseError):
    """Raised when Result operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """A type that represents either success (Ok) or failure (Err).

    This is inspired by Rust's Result type and forces explicit error handling.
    A Result is either Ok(value) or Err(error), never both, never neither.

    Examples:
        >>> result = Result.Ok(42)
        >>> assert result.is_ok()
        >>> assert result.unwrap() == 42

        >>> error = ResultError("Something went wrong")
        >>> result = Result.Err(error)
        >>> assert result.is_err()
        >>> assert result.unwrap_or(0) == 0

    Attributes:
        _ok: The success value if this is Ok, None otherwise
        _error: The error value if this is Err, None otherwise
        _is_ok: Internal flag indicating if this is Ok or Err
    """

    _ok: T | None = None
    _error: E | None = None
    _is_ok: bool = False
    _is_err: bool = False

    def __post_init__(self) -> None:
        """Validate that Result is properly constructed."""
        if self._is_ok and self._error is not None:
            raise ResultError("Ok result cannot have an error")
        if not self._is_ok and self._ok is not None:
            raise ResultError("Err result cannot have an ok value")
        if not self._is_ok and self._error is None:
            raise ResultError("Err result must have an error")

    @classmethod
    def Ok(cls, value: T) -> Result[T, E]:
        """Create a successful Result.

        Args:
            value: The success value

        Returns:
            A Result representing success

        Example:
            >>> result = Result.Ok(42)
            >>> assert result.is_ok()
        """
        return cls(_ok=value, _error=None, _is_ok=True)

    @classmethod
    def Err(cls, error: E) -> Result[T, E]:
        """Create a failed Result.

        Args:
            error: The error value

        Returns:
            A Result representing failure

        Example:
            >>> error = ResultError("error message")
            >>> result = Result.Err(error)
            >>> assert result.is_err()
        """
        return cls(_ok=None, _error=error, _is_ok=False)

    def is_ok(self) -> bool:
        """Check if this Result is Ok.

        Returns:
            True if this is Ok, False if Err
        """
        return self._is_ok

    def is_err(self) -> bool:
        """Check if this Result is Err.

        Returns:
            True if this is Err, False if Ok
        """
        return not self._is_ok

    def unwrap(self) -> T:
        """Extract the Ok value, raising an exception if Err.

        Returns:
            The Ok value

        Raises:
            ResultError: If this Result is Err

        Example:
            >>> Result.Ok(42).unwrap()
            42
            >>> Result.Err(ResultError("error")).unwrap()  # Raises ResultError
        """
        if self._is_ok:
            return self._ok  # type: ignore

        raise ResultError(f"Called unwrap() on an Err value: {self._error}")

    def unwrap_err(self) -> E:
        """Extract the Err value, raising an exception if Ok.

        Returns:
            The Err value

        Raises:
            ResultError: If this Result is Ok

        Example:
            >>> error = ResultError("error")
            >>> Result.Err(error).unwrap_err()
            ResultError('error')
            >>> Result.Ok(42).unwrap_err()  # Raises ResultError
        """
        if self._is_ok or self._error is None:
            raise ResultError(f"Called unwrap_err() on an Ok value: {self._ok}")

        return self._error

    def unwrap_or(self, default: T) -> T:
        """Extract the Ok value or return a default if Err.

        Args:
            default: The default value to return if Err

        Returns:
            The Ok value if Ok, otherwise the default

        Example:
            >>> Result.Ok(42).unwrap_or(0)
            42
            >>> Result.Err(ResultError("error")).unwrap_or(0)
            0
        """
        if self._is_ok:
            assert self._ok is not None
            return self._ok
        return default

    def unwrap_or_else(self, callback: Callable[[E], T]) -> T:
        """Extract the Ok value or compute a default from the error.

        Args:
            callback: Function to compute default from error

        Returns:
            The Ok value if Ok, otherwise fn(error)

        Example:
            >>> error = ResultError("error")
            >>> Result.Err(error).unwrap_or_else(lambda e: f"Got: {e}")
            'Got: error'
        """
        if self._is_ok:
            assert self._ok is not None
            return self._ok
        assert self._error is not None
        return callback(self._error)

    def expect(self, message: str) -> T:
        """Extract the Ok value or raise with a custom message.

        Args:
            message: Custom error message

        Returns:
            The Ok value

        Raises:
            ResultError: If this Result is Err, with custom message

        Example:
            >>> Result.Ok(42).expect("Should be a number")
            42
            >>> Result.Err(ResultError("error")).expect("Should be a number")
            # Raises: ResultError: Should be a number: error
        """
        if self._is_ok:
            return self._ok  # type: ignore
        raise ResultError(f"{message}: {self._error}")

    def map(self, callback: Callable[[T], U]) -> Result[U, E]:
        """Transform the Ok value, leaving Err untouched.

        Args:
            fn: Function to transform the Ok value

        Returns:
            Result with transformed Ok value, or original Err

        Example:
            >>> Result.Ok(2).map(lambda x: x * 2)
            Result.Ok(4)
            >>> error = ResultError("error")
            >>> Result.Err(error).map(lambda x: x * 2)
            Result.Err(error)
        """
        if self._is_ok:
            assert self._ok is not None
            return Result(_ok=callback(self._ok), _error=None, _is_ok=True)
        assert self._error is not None
        return Result(_ok=None, _error=self._error, _is_ok=False)

    def map_err(self, callback: Callable[[E], F]) -> Result[T, F]:
        """Transform the Err value, leaving Ok untouched.

        Args:
            fn: Function to transform the Err value

        Returns:
            Result with transformed Err value, or original Ok

        Example:
            >>> def wrap_error(e: ResultError) -> FoundationBaseError:
            ...     return FoundationBaseError(f"Wrapped: {e}")
            >>> error = ResultError("original error")
            >>> Result.Err(error).map_err(wrap_error)
            Result.Err(FoundationBaseError('Wrapped: original error'))
            >>> Result.Ok(42).map_err(wrap_error)
            Result.Ok(42)
        """
        if self._is_ok:
            assert self._ok is not None
            return Result(_ok=self._ok, _error=None, _is_ok=True)
        assert self._error is not None
        return Result(_ok=None, _error=callback(self._error), _is_ok=False)

    def and_then(self, callback: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain Result-producing operations (flatMap/bind).

        Args:
            callback: Function that takes Ok value and returns another Result

        Returns:
            Result from callback if Ok, otherwise original Err

        Example:
            >>> def divide(x: int) -> Result[int, ResultError]:
            ...     if x == 0:
            ...         return Result.Err(ResultError("division by zero"))
            ...     return Result.Ok(10 // x)
            >>> Result.Ok(2).and_then(divide)
            Result.Ok(5)
            >>> Result.Ok(0).and_then(divide)
            Result.Err(ResultError("division by zero"))
        """
        if self._is_ok:
            return callback(self._ok)  # type: ignore
        return Result.Err(self._error)  # type: ignore

    def or_else(self, callback: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Provide fallback Result if this is Err.

        Args:
            fn: Function that takes Err value and returns another Result

        Returns:
            Original Result if Ok, otherwise Result from fn

        Example:
            >>> def fallback(e: ResultError) -> Result[int, FoundationBaseError]:
            ...     return Result.Ok(0)
            >>> error = ResultError("error")
            >>> Result.Err(error).or_else(fallback)
            Result.Ok(0)
            >>> Result.Ok(42).or_else(fallback)
            Result.Ok(42)
        """
        if self._is_ok:
            assert self._ok is not None
            return Result(_ok=self._ok, _error=None, _is_ok=True)
        assert self._error is not None
        return callback(self._error)

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self._is_ok:
            return f"Result.Ok({self._ok!r})"
        return f"Result.Err({self._error!r})"
