"""Telegram interface specific errors following project patterns."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from blockether_foundation.errors import FoundationBaseError


class BotValidationErrorDetails(BaseModel):
    """Details for bot validation errors."""
    bot_name: str
    validation_errors: list[str]
    provided_config: dict[str, Any]
    timestamp: datetime

    model_config = {"arbitrary_types_allowed": True}


class TelegramConfigurationDetails(BaseModel):
    """Details for Telegram configuration errors."""
    configuration_type: str
    expected_type: str
    received_value: Any
    timestamp: datetime

    model_config = {"arbitrary_types_allowed": True}


class BotNameConflictDetails(BaseModel):
    """Details for bot name conflict errors."""
    conflicting_names: list[str]
    all_bot_names: list[str]
    timestamp: datetime


class TelegramInterfaceError(FoundationBaseError):
    """Base error for Telegram interface operations."""

    def __init__(self, message: str, details: BaseModel | None = None) -> None:
        super().__init__(message, details)


class BotValidationError(TelegramInterfaceError):
    """Raised when bot configuration validation fails."""

    def __init__(self, bot_name: str, validation_errors: list[str], provided_config: dict[str, Any]) -> None:
        details = BotValidationErrorDetails(
            bot_name=bot_name,
            validation_errors=validation_errors,
            provided_config=provided_config,
            timestamp=datetime.now(UTC)
        )
        super().__init__(f"Bot configuration validation failed for '{bot_name}': {', '.join(validation_errors)}", details)
        self._bot_name = bot_name

    @property
    def bot_name(self) -> str:
        """Get the bot name from the error."""
        return self._bot_name


class TelegramConfigurationError(TelegramInterfaceError):
    """Raised when Telegram interface configuration is invalid."""

    def __init__(self, message: str, configuration_type: str, expected_type: str, received_value: Any) -> None:
        details = TelegramConfigurationDetails(
            configuration_type=configuration_type,
            expected_type=expected_type,
            received_value=received_value,
            timestamp=datetime.now(UTC)
        )
        super().__init__(message, details)


class BotNameConflictError(TelegramInterfaceError):
    """Raised when bot names are not unique across configurations."""

    def __init__(self, conflicting_names: list[str], all_bot_names: list[str]) -> None:
        details = BotNameConflictDetails(
            conflicting_names=conflicting_names,
            all_bot_names=all_bot_names,
            timestamp=datetime.now(UTC)
        )
        super().__init__(f"Bot names must be unique. Conflicting names: {', '.join(conflicting_names)}", details)
        self._conflicting_names = conflicting_names

    @property
    def conflicting_names(self) -> list[str]:
        """Get the conflicting bot names."""
        return self._conflicting_names