"""Unit tests for Telegram validation helpers."""

from unittest.mock import patch

import pytest

from blockether_foundation.os.interfaces.telegram.errors import (
    BotNameConflictError,
    BotValidationError,
    BotValidationErrorDetails,
    TelegramConfigurationError,
)
from blockether_foundation.os.interfaces.telegram.models import BotConfig
from blockether_foundation.os.interfaces.telegram.validation import (
    validate_and_normalize_bot_configs,
    validate_bot_config_list,
    validate_bot_name,
    validate_single_bot_config,
)
from blockether_foundation.result import Result


def _make_bot_config(name: str) -> BotConfig:
    return BotConfig(name=name, token="A" * 16)


@pytest.mark.unit
def test_validate_and_normalize_accepts_valid_list() -> None:
    config = _make_bot_config("valid-bot")

    result = validate_and_normalize_bot_configs([config])

    assert result.is_ok()
    validated = result.unwrap()
    assert validated == [config]


@pytest.mark.unit
def test_validate_and_normalize_rejects_duplicate_names() -> None:
    config_one = _make_bot_config("duplicate-bot")
    config_two = _make_bot_config("duplicate-bot")

    result = validate_and_normalize_bot_configs([config_one, config_two])

    assert result.is_err()
    error = result.unwrap_err()
    assert isinstance(error, BotNameConflictError)
    assert error.conflicting_names == ["duplicate-bot"]


@pytest.mark.unit
def test_validate_bot_name_rejects_invalid_characters() -> None:
    result = validate_bot_name("invalid@name")
    assert result.is_err()


@pytest.mark.unit
def test_validate_bot_name_rejects_empty_string() -> None:
    result = validate_bot_name("   ")
    assert result.is_err()


@pytest.mark.unit
def test_validate_bot_name_rejects_overly_long_name() -> None:
    result = validate_bot_name("a" * 65)
    assert result.is_err()


@pytest.mark.unit
def test_validate_single_bot_config_errors_on_bad_values() -> None:
    config = BotConfig(
        name="bad name!",
        token="short",
        max_concurrent_updates=0,
        executor_timeout=4001,
    )
    result = validate_single_bot_config(config)
    assert result.is_err()
    error = result.unwrap_err()
    assert isinstance(error, BotValidationError)
    assert isinstance(error.details, BotValidationErrorDetails)
    assert any("cannot" in msg for msg in error.details.validation_errors)


@pytest.mark.unit
def test_validate_single_bot_config_requires_token() -> None:
    config = BotConfig(name="valid-name", token="   ")
    result = validate_single_bot_config(config)

    assert result.is_err()
    error = result.unwrap_err()
    assert isinstance(error.details, BotValidationErrorDetails)
    assert "Bot token cannot be empty" in error.details.validation_errors


@pytest.mark.unit
def test_validate_single_bot_config_numeric_upper_bounds() -> None:
    config = BotConfig(
        name="valid-name",
        token="A" * 32,
        max_concurrent_updates=2000,
        executor_timeout=0,
    )

    result = validate_single_bot_config(config)

    assert result.is_err()
    error = result.unwrap_err()
    assert isinstance(error.details, BotValidationErrorDetails)
    assert any("cannot exceed 1000" in msg for msg in error.details.validation_errors)
    assert any(
        "executor_timeout must be positive" in msg for msg in error.details.validation_errors
    )


@pytest.mark.unit
def test_validate_bot_config_list_rejects_empty_list() -> None:
    result = validate_bot_config_list([])
    assert result.is_err()


@pytest.mark.unit
def test_validate_bot_config_list_returns_first_error() -> None:
    valid = _make_bot_config("valid")
    invalid = BotConfig(name="", token="A" * 32)

    result = validate_bot_config_list([valid, invalid])

    assert result.is_err()
    assert isinstance(result.unwrap_err(), BotValidationError)


@pytest.mark.unit
def test_validate_and_normalize_bot_configs_requires_non_empty_list() -> None:
    result = validate_and_normalize_bot_configs([])
    assert result.is_err()
    assert isinstance(result.unwrap_err(), TelegramConfigurationError)


@pytest.mark.unit
def test_validate_single_bot_config_with_name_error_without_details() -> None:
    """Test name validation error without details attribute (covers line 74)."""

    # Create a config with invalid name
    config = BotConfig(
        name="invalid name with spaces",  # This will trigger name validation error
        token="A" * 46,
        webhook_secret="secret",
    )

    # Mock validate_bot_name to return an error without details.validation_errors
    with patch(
        "blockether_foundation.os.interfaces.telegram.validation.validate_bot_name"
    ) as mock_validate:
        # Create a mock error without details attribute
        error_without_details = BotValidationError(
            bot_name="invalid name",
            validation_errors=["Invalid format"],
            provided_config={"name": "invalid name"},
        )
        # Remove the details attribute to trigger the else branch
        delattr(error_without_details, "details")
        mock_validate.return_value = Result[None, BotValidationError].Err(error_without_details)

        # This should trigger the else branch on line 74
        result = validate_single_bot_config(config)
        assert result.is_err()
        validation_error = result.unwrap_err()
        assert isinstance(validation_error, BotValidationError)
        assert "Invalid bot name" in validation_error.message
