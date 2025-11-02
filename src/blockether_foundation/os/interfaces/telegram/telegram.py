"""Telegram interface implementation for Agno AgentOS with explicit multi-bot support."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from agno.agent import Agent
from agno.os.interfaces.base import BaseInterface
from agno.team import Team
from agno.utils.log import logger
from agno.workflow import Workflow
from fastapi import APIRouter

from .errors import (
    BotNameConflictError,
    BotValidationError,
    TelegramConfigurationError,
)
from .handlers import attach_routes
from .models import BotConfig
from .validation import validate_and_normalize_bot_configs


class Telegram(BaseInterface):
    """Telegram interface for Agno AgentOS with multi-bot support."""

    type = "telegram"
    version = "1.0.0"

    def __init__(
        self,
        executor: Agent | Team | Workflow,
        bot_configs: list[BotConfig],
        tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Telegram interface.

        Args:
            executor: Agent, Team, or Workflow to handle messages
            bot_configs: List of bot configurations (must have at least one)
            tags: List of tags for the interface

        Raises:
            TelegramConfigurationError: If configuration is invalid
            BotValidationError: If bot configuration is invalid
            BotNameConflictError: If bot names are not unique
        """
        logger.info(f"Initializing Telegram interface with {len(bot_configs)} bot(s)")

        # Validate bot configurations
        validation_result = validate_and_normalize_bot_configs(bot_configs)
        if validation_result.is_err():
            error = validation_result.unwrap_err()
            logger.error(f"Failed to validate bot configurations: {error}")
            raise error

        self.bot_configs = validation_result.unwrap()
        self.tags = list(tags) if tags else ["telegram"]

        # Store the executor in the appropriate BaseInterface property
        if isinstance(executor, Agent):
            self.agent = executor
            self.team = None
            self.workflow = None
            logger.info(f"Telegram interface configured with Agent executor")
        elif isinstance(executor, Team):
            self.agent = None
            self.team = executor
            self.workflow = None
            logger.info(f"Telegram interface configured with Team executor")
        elif isinstance(executor, Workflow):
            self.agent = None
            self.team = None
            self.workflow = executor
            logger.info(f"Telegram interface configured with Workflow executor")
        else:
            self.agent = None
            self.team = None
            self.workflow = None
            logger.warning(f"Telegram interface configured with no executor")

        bot_names = [config.name for config in self.bot_configs]
        logger.info(f"Telegram interface initialized successfully for bots: {bot_names}")

    def get_router(self, use_async: bool = True, **kwargs: Any) -> APIRouter:
        """Get FastAPI router for Telegram webhook handling."""
        logger.info("Creating FastAPI router for Telegram interface")

        # Create main router for the interface
        main_router = APIRouter(prefix="/telegram", tags=list(self.tags))

        # Create individual routers for each bot configuration
        for bot_config in self.bot_configs:
            logger.info(f"Creating router for bot: {bot_config.name}")
            bot_router = APIRouter(prefix=f"/{bot_config.name}", tags=[f"telegram-{bot_config.name}"])

            attach_routes(
                router=bot_router,
                executor=self.agent or self.team or self.workflow,
                bot_config=bot_config,
            )

            # Include bot router in main router
            main_router.include_router(bot_router)

        # Add overview endpoint for all bots
        @main_router.get("/status")
        async def interface_status() -> dict[str, Any]:
            """Get overview status of all configured bots."""
            logger.debug("Interface status requested")
            return {
                "status": "active",
                "interface_version": self.version,
                "bot_count": len(self.bot_configs),
                "bots": [
                    {
                        "name": config.name,
                        "webhook_url": config.webhook_url,
                        "has_webhook_secret": bool(config.webhook_secret),
                        "max_concurrent_updates": config.max_concurrent_updates,
                        "executor_timeout": config.executor_timeout,
                        "has_access_restrictions": bool(config.allowlist_user_ids) or bool(config.denylist_user_ids)
                    }
                    for config in self.bot_configs
                ],
                "executor_type": type(self.agent or self.team or self.workflow).__name__ if (self.agent or self.team or self.workflow) else None,
                "timestamp": datetime.utcnow().isoformat()
            }

        logger.info(f"FastAPI router created with {len(self.bot_configs)} bot endpoints")
        return main_router