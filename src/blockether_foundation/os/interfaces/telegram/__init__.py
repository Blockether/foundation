"""Telegram interface for Agno AgentOS with explicit multi-bot support."""

from .errors import (
    BotNameConflictError,
    BotValidationError,
    TelegramConfigurationError,
    TelegramInterfaceError,
)
from .models import BotConfig
from .telegram import Telegram

__all__ = [
    "BotConfig",
    "Telegram",
    "TelegramInterfaceError",
    "TelegramConfigurationError",
    "BotValidationError",
    "BotNameConflictError",
]
