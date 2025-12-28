"""Telegram interface implementation for Agno AgentOS with explicit multi-bot support."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, override

from agno.agent import Agent
from agno.os.interfaces.base import BaseInterface
from agno.team import Team
from agno.utils.log import log_debug, log_error, log_info, log_warning, logger  # type: ignore
from agno.workflow import Workflow
from fastapi import APIRouter

from ...runtime.aws_lambda.background_tasks import (
    LambdaBackgroundTaskExtension,
)
from .handlers import BackgroundTaskScheduler, attach_routes
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
        super().__init__(**kwargs)
        self.background_task_scheduler = self._maybe_create_lambda_extension()
        log_info(f"Initializing Telegram interface with {len(bot_configs)} bot(s)")

        # Validate bot configurations
        validation_result = validate_and_normalize_bot_configs(bot_configs)
        if validation_result.is_err():
            error = validation_result.unwrap_err()
            log_error(f"Failed to validate bot configurations: {error}")
            raise error

        self.bot_configs = validation_result.unwrap()
        self.tags = list(tags) if tags else ["telegram"]
        self._configure_executor(executor)

        bot_names = [config.name for config in self.bot_configs]
        log_info(f"Telegram interface initialized successfully for bots: {bot_names}")

    @override
    def get_router(self, use_async: bool = True, **kwargs: Any) -> APIRouter:
        """Get FastAPI router for Telegram webhook handling.

        Args:
            use_async: Deprecated and ignored. Use bot_config.use_async_executor instead.
            **kwargs: Additional keyword arguments
        """
        if use_async is not True:
            logger.warning(
                "use_async parameter is deprecated. Use bot_config.use_async_executor instead."
            )
        """Get FastAPI router for Telegram webhook handling."""
        log_info("Creating FastAPI router for Telegram interface")

        # Create main router for the interface
        main_router = APIRouter(prefix="/telegram", tags=list(self.tags))

        # Create individual routers for each bot configuration
        for bot_config in self.bot_configs:
            log_info(f"Creating router for bot: {bot_config.name}")
            bot_router = APIRouter(
                prefix=f"/{bot_config.name}", tags=[f"telegram-{bot_config.name}"]
            )

            attach_routes(
                router=bot_router,
                executor=self.executor,
                bot_config=bot_config,
                task_scheduler=self.background_task_scheduler,
                use_async_executor=bot_config.use_async_executor,
            )

            main_router.include_router(bot_router)

        log_info(f"FastAPI router created with {len(self.bot_configs)} bot endpoints")
        return main_router

    @property
    def executor(self) -> Agent | Team | Workflow | None:
        """Return the configured executor instance, if any."""
        return self.agent or self.team or self.workflow  # type: ignore

    def _configure_executor(self, executor: Agent | Team | Workflow | None) -> None:
        """Store executor on the correct interface attribute and log the outcome."""
        self.agent = None
        self.team = None
        self.workflow = None

        if isinstance(executor, Agent):
            self.agent = executor
            log_info("Telegram interface configured with Agent executor")
        elif isinstance(executor, Team):
            self.team = executor
            log_info("Telegram interface configured with Team executor")
        elif isinstance(executor, Workflow):
            self.workflow = executor
            log_info("Telegram interface configured with Workflow executor")
        else:
            log_warning("Telegram interface configured with no executor")

    def _maybe_create_lambda_extension(self) -> BackgroundTaskScheduler | None:
        """Instantiate Lambda extension when running inside AWS Lambda."""

        runtime_api = os.getenv("AWS_LAMBDA_RUNTIME_API")
        if not runtime_api:
            log_debug("AWS_LAMBDA_RUNTIME_API not set; skipping Lambda background task scheduler")
            return None

        try:
            extension = LambdaBackgroundTaskExtension()
        except Exception as exc:
            log_warning(f"Failed to start Lambda background task extension: {exc}")
            return None

        log_info("Lambda background task extension enabled for Telegram interface")
        return extension
