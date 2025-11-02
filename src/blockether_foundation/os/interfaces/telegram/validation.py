"""Validation utilities for Telegram interface using Result types."""

from __future__ import annotations

from typing import Any

from agno.utils.log import logger
from blockether_foundation.result import Result

from .errors import (
    BotValidationError,
    BotNameConflictError,
    TelegramConfigurationError,
)
from .models import BotConfig


def validate_bot_name(name: str) -> Result[None, BotValidationError]:
    """Validate a single bot name."""
    errors = []

    if not name or not name.strip():
        errors.append("Bot name cannot be empty")

    if len(name.strip()) > 64:
        errors.append("Bot name cannot exceed 64 characters")

    # Allow alphanumeric characters, hyphens, underscores, and spaces
    if not all(c.isalnum() or c in "-_ " for c in name):
        errors.append(
            "Bot name can only contain alphanumeric characters, hyphens, underscores, and spaces"
        )

    if errors:
        logger.debug(f"Bot name validation failed for '{name}': {errors}")
        return Result.Err(
            BotValidationError(
                bot_name=name, validation_errors=errors, provided_config={"name": name}
            )
        )

    logger.debug(f"Bot name validation passed for '{name}'")
    return Result.Ok(None)


def validate_single_bot_config(bot_config: BotConfig) -> Result[None, BotValidationError]:
    """Validate a single bot configuration."""
    errors = []

    # Validate name
    name_result = validate_bot_name(bot_config.name)
    if name_result.is_err():
        errors.extend(
            name_result.unwrap_err().details.validation_errors
            if name_result.unwrap_err().details
            else ["Invalid bot name"]
        )

    # Validate token
    if not bot_config.token or not bot_config.token.strip():
        errors.append("Bot token cannot be empty")

    if bot_config.token and len(bot_config.token) < 10:
        errors.append("Bot token appears to be invalid (too short)")

    # Validate numeric parameters
    if bot_config.max_concurrent_updates <= 0:
        errors.append("max_concurrent_updates must be positive")

    if bot_config.max_concurrent_updates > 1000:
        errors.append("max_concurrent_updates cannot exceed 1000")

    if bot_config.executor_timeout <= 0:
        errors.append("executor_timeout must be positive")

    if bot_config.executor_timeout > 3600:  # 1 hour max
        errors.append("executor_timeout cannot exceed 3600 seconds")

    if errors:
        logger.error(f"Bot configuration validation failed for '{bot_config.name}': {errors}")
        return Result.Err(
            BotValidationError(
                bot_name=bot_config.name,
                validation_errors=errors,
                provided_config=bot_config.model_dump(),
            )
        )

    logger.debug(f"Bot configuration validation passed for '{bot_config.name}'")
    return Result.Ok(None)


def validate_bot_config_list(
    bot_configs: list[BotConfig],
) -> Result[
    list[BotConfig], TelegramConfigurationError | BotValidationError | BotNameConflictError
]:
    """Validate a list of bot configurations."""
    if not bot_configs:
        logger.error("Bot configuration list validation failed: empty list")
        return Result.Err(
            TelegramConfigurationError(
                message="Bot configuration list cannot be empty",
                configuration_type="bot_configs",
                expected_type="non-empty list[BotConfig]",
                received_value=bot_configs,
            )
        )

    # Validate each bot configuration
    for bot_config in bot_configs:
        validation_result = validate_single_bot_config(bot_config)
        if validation_result.is_err():
            # Convert the error to be compatible with the return type
            error = validation_result.unwrap_err()
            return Result.Err(error)

    logger.debug(f"Bot configuration list validation passed for {len(bot_configs)} configurations")
    return Result.Ok(bot_configs)


def check_bot_name_uniqueness(bot_configs: list[BotConfig]) -> Result[None, BotNameConflictError]:
    """Check that all bot names are unique."""
    names = [config.name for config in bot_configs]
    unique_names = set(names)

    if len(names) != len(unique_names):
        # Find duplicates
        seen = set()
        duplicates = []
        for name in names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)

        logger.error(f"Bot name uniqueness check failed: duplicate names {duplicates}")
        return Result.Err(BotNameConflictError(conflicting_names=duplicates, all_bot_names=names))

    logger.debug(f"Bot name uniqueness check passed for {len(names)} bots")
    return Result.Ok(None)


def normalize_bot_configs(
    bot: BotConfig | list[BotConfig],
) -> Result[
    list[BotConfig], TelegramConfigurationError | BotValidationError | BotNameConflictError
]:
    """Normalize bot configurations to a list."""
    if isinstance(bot, BotConfig):
        logger.debug("Normalized single BotConfig to list")
        return Result.Ok([bot])
    elif isinstance(bot, list):
        logger.debug(f"Bot configuration is already a list with {len(bot)} items")
        if not bot:
            return Result.Err(
                TelegramConfigurationError(
                    message="Bot configuration list cannot be empty",
                    configuration_type="bot_configs",
                    expected_type="non-empty list[BotConfig]",
                    received_value=bot,
                )
            )
        return Result.Ok(bot)
    else:
        return Result.Err(
            TelegramConfigurationError(
                message=f"Invalid bot configuration type: {type(bot)}",
                configuration_type="bot_configs",
                expected_type="BotConfig | list[BotConfig]",
                received_value=bot,
            )
        )


def validate_and_normalize_bot_configs(
    bot: BotConfig | list[BotConfig],
) -> Result[
    list[BotConfig], TelegramConfigurationError | BotValidationError | BotNameConflictError
]:
    """Comprehensive validation and normalization of bot configurations."""
    logger.info("Starting comprehensive bot configuration validation")

    # Normalize to list
    normalize_result = normalize_bot_configs(bot)
    if normalize_result.is_err():
        return normalize_result

    bot_configs = normalize_result.unwrap()

    # Validate list
    list_validation_result = validate_bot_config_list(bot_configs)
    if list_validation_result.is_err():
        return list_validation_result

    # Check name uniqueness
    uniqueness_result = check_bot_name_uniqueness(bot_configs)
    if uniqueness_result.is_err():
        # Convert Result[None, Error] to Result[list[BotConfig], Error]
        error = uniqueness_result.unwrap_err()
        return Result.Err(error)

    logger.info(
        f"Bot configuration validation completed successfully for bots: {[bot.name for bot in bot_configs]}"
    )
    return Result.Ok(bot_configs)
