"""Tests covering Telegram-specific error classes."""

import pytest

from blockether_foundation.os.interfaces.telegram.errors import (
    BotNameConflictDetails,
    BotNameConflictError,
    BotValidationError,
    BotValidationErrorDetails,
    TelegramConfigurationDetails,
    TelegramConfigurationError,
)


@pytest.mark.unit
def test_bot_validation_error_details() -> None:
    error = BotValidationError(
        bot_name="helper-bot",
        validation_errors=["token missing", "name invalid"],
        provided_config={"name": "helper-bot"},
    )

    assert "helper-bot" in str(error)
    assert error.bot_name == "helper-bot"
    assert isinstance(error.details, BotValidationErrorDetails)
    assert error.details.validation_errors == ["token missing", "name invalid"]
    assert error.details.provided_config == {"name": "helper-bot"}


@pytest.mark.unit
def test_telegram_configuration_error_details() -> None:
    error = TelegramConfigurationError(
        message="Bad config",
        configuration_type="bot_configs",
        expected_type="list[BotConfig]",
        received_value="oops",
    )

    assert "Bad config" in str(error)
    assert isinstance(error.details, TelegramConfigurationDetails)
    assert error.details.configuration_type == "bot_configs"
    assert error.details.expected_type == "list[BotConfig]"
    assert error.details.received_value == "oops"


@pytest.mark.unit
def test_bot_name_conflict_error_details() -> None:
    error = BotNameConflictError(conflicting_names=["dup"], all_bot_names=["dup", "dup"])

    assert "conflicting names" in str(error).lower()
    assert error.conflicting_names == ["dup"]
    assert isinstance(error.details, BotNameConflictDetails)
    assert error.details.all_bot_names == ["dup", "dup"]
