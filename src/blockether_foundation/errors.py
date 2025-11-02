from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel

D = TypeVar("D", BaseModel, None)


class FoundationBaseError(Exception):
    def __init__(
        self,
        message: str,
        details: D | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message
            details: Typed error details (must use FoundationErrorDetails union models)
        """
        super().__init__(message)
        self.message = message
        self.details = details
        self.timestamp = datetime.now(UTC)

    def __str__(self) -> str:
        """String representation for error messages."""
        if self.details and isinstance(self.details, BaseModel):
            details_str = str(self.details.model_dump())
            return f"{self.__class__.__module__}.{self.__class__.__name__}: {self.message} (details: {details_str})"
        return f"{self.__class__.__module__}.{self.__class__.__name__}: {self.message}"


class ConsensusFieldInitError(FoundationBaseError):
    """Raised when a consensus field is improperly initialized."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
