"""Minimal webhook handlers for Telegram interface."""

from datetime import datetime
import json
from typing import Any, Dict

from agno.agent import Agent
from agno.team import Team
from agno.utils.log import logger
from agno.workflow import Workflow
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from .models import BotConfig, HealthResponse, StatusResponse, Update, WebhookResponse

# Constants
MAX_WEBHOOK_SIZE = 1024 * 1024  # 1MB
EXECUTOR_TIMEOUT = 30  # seconds


class StructuredLogger:
    """Simple structured logging utility using Agno's logger."""

    @staticmethod
    def log_webhook_received(update_id: int, user_id: int | None) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "webhook_received",
                    "timestamp": datetime.utcnow().isoformat(),
                    "update_id": update_id,
                    "user_id": user_id,
                }
            )
        )

    @staticmethod
    def log_webhook_error(error_detail: str, status_code: int = 400) -> None:
        logger.error(
            json.dumps(
                {
                    "event": "webhook_error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error_detail": error_detail,
                    "status_code": status_code,
                }
            )
        )

    @staticmethod
    def log_executor_start(update_id: int, user_id: int, executor_type: str) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "executor_start",
                    "timestamp": datetime.utcnow().isoformat(),
                    "update_id": update_id,
                    "user_id": user_id,
                    "executor_type": executor_type,
                }
            )
        )

    @staticmethod
    def log_executor_complete(update_id: int, duration_ms: int, success: bool) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "executor_complete",
                    "timestamp": datetime.utcnow().isoformat(),
                    "update_id": update_id,
                    "duration_ms": duration_ms,
                    "success": success,
                }
            )
        )

    @staticmethod
    def log_executor_error(update_id: int, error_type: str, error_message: str) -> None:
        logger.error(
            json.dumps(
                {
                    "event": "executor_error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "update_id": update_id,
                    "error_type": error_type,
                    "error_message": error_message,
                }
            )
        )


def attach_routes(
    router: APIRouter, executor: Agent | Team | Workflow | None, bot_config: BotConfig
) -> APIRouter:
    """Attach minimal Telegram webhook routes to the router."""
    logger.info(f"Attaching routes for bot: {bot_config.name}")

    @router.post("/webhook", response_model=WebhookResponse)
    async def webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        x_telegram_bot_api_secret_token: str | None = Header(
            None, alias="X-Telegram-Bot-Api-Secret-Token"
        ),
    ) -> WebhookResponse:
        """Handle Telegram webhook updates."""
        start_time = datetime.utcnow()
        logger.debug(f"Webhook received for bot {bot_config.name}")

        try:
            # 1. Verify webhook signature if configured
            if hasattr(bot_config, "webhook_secret") and bot_config.webhook_secret:
                if x_telegram_bot_api_secret_token != bot_config.webhook_secret:
                    StructuredLogger.log_webhook_error("Unauthorized webhook request", 401)
                    raise HTTPException(status_code=401, detail="Unauthorized")

            # 2. Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_WEBHOOK_SIZE:
                StructuredLogger.log_webhook_error(
                    f"Request too large: {content_length} bytes", 413
                )
                raise HTTPException(status_code=413, detail="Request too large")

            # 3. Parse and validate update from Telegram
            try:
                update_data = await request.json()
                update = Update(**update_data)
            except Exception as validation_error:
                StructuredLogger.log_webhook_error(
                    f"Invalid update format: {str(validation_error)}", 400
                )
                raise HTTPException(status_code=400, detail="Invalid update format")

            # 4. Basic validation and logging
            user_id = extract_user_id(update)
            StructuredLogger.log_webhook_received(update.update_id, user_id)

            # 5. Process the update in background
            background_tasks.add_task(
                process_update_async, update=update, executor=executor, bot_config=bot_config
            )

            logger.debug(f"Webhook for bot {bot_config.name} queued for processing")
            return WebhookResponse(
                status="ok",
                update_id=update.update_id,
                processed_at=start_time.isoformat(),
            )

        except HTTPException:
            raise
        except Exception as e:
            StructuredLogger.log_webhook_error(f"Unexpected webhook error: {str(e)}", 500)
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get("/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        """Get simple bot status."""
        logger.debug(f"Status requested for bot {bot_config.name}")
        return StatusResponse(
            status="active",
            bot_name=bot_config.name,
            timestamp=datetime.utcnow().isoformat(),
        )

    @router.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Simple health check."""
        logger.debug(f"Health check requested for bot {bot_config.name}")
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat()
        )

    logger.info(f"Routes attached successfully for bot {bot_config.name}")
    return router


def extract_user_id(update: Update) -> int | None:
    """Extract user ID from update."""
    if update.message and "from" in update.message:
        user_id = update.message["from"].get("id")
        return int(user_id) if user_id is not None else None
    elif update.callback_query and "from" in update.callback_query:
        user_id = update.callback_query["from"].get("id")
        return int(user_id) if user_id is not None else None
    return None


def is_user_allowed(user_id: int, bot_config: BotConfig) -> bool:
    """Check if user is allowed based on allowlist and denylist logic."""
    # Convert user_id to string for comparison since bot configs store string IDs
    user_id_str = str(user_id)

    # Check denylist first (takes precedence)
    if bot_config.denylist_user_ids and user_id_str in bot_config.denylist_user_ids:
        return False

    # Check allowlist
    if bot_config.allowlist_user_ids and len(bot_config.allowlist_user_ids) > 0:
        return user_id_str in bot_config.allowlist_user_ids

    # If no allowlist or empty allowlist, allow all
    return True


def get_access_denied_reason(user_id: int, bot_config: BotConfig) -> str:
    """Get the reason why access was denied for logging."""
    # Convert user_id to string for comparison
    user_id_str = str(user_id)

    # Check denylist first
    if bot_config.denylist_user_ids and user_id_str in bot_config.denylist_user_ids:
        return f"User {user_id} is in denylist"

    # Check allowlist
    if bot_config.allowlist_user_ids and len(bot_config.allowlist_user_ids) > 0:
        if user_id_str not in bot_config.allowlist_user_ids:
            return f"User {user_id} not in allowlist"

    return f"User {user_id} access denied"


def format_message_for_executor(update: Update) -> str:
    """Format a Telegram update for the executor."""
    if update.message:
        user_info = update.message.get("from", {})
        chat_info = update.message.get("chat", {})
        message_text = update.message.get("text", "") or update.message.get("caption", "")

        user_display = user_info.get("first_name", "Unknown")
        user_id = user_info.get("id", 0)
        chat_id = chat_info.get("id", 0)
        chat_type = chat_info.get("type", "unknown")

        if message_text:
            return f"User {user_display} (ID: {user_id}) sent message in {chat_type} chat {chat_id}: {message_text}"
        else:
            return f"User {user_display} (ID: {user_id}) sent non-text message in {chat_type} chat {chat_id}"

    elif update.callback_query:
        user_info = update.callback_query.get("from", {})
        callback_data = update.callback_query.get("data", "")

        user_display = user_info.get("first_name", "Unknown")
        user_id = user_info.get("id", 0)

        return f"User {user_display} (ID: {user_id}) pressed button: {callback_data}"

    return f"Received update {update.update_id} with unsupported format"


async def process_update_async(
    update: Update, executor: Agent | Team | Workflow | None, bot_config: BotConfig
) -> None:
    """Process a Telegram update asynchronously."""
    import asyncio

    start_time = datetime.utcnow()
    user_id = extract_user_id(update)

    if not user_id:
        StructuredLogger.log_executor_error(
            update.update_id, "no_user_id", "Could not extract user ID"
        )
        return

    try:
        # Validate user access with allowlist/denylist logic
        if not is_user_allowed(user_id, bot_config):
            reason = get_access_denied_reason(user_id, bot_config)
            StructuredLogger.log_executor_error(update.update_id, "access_denied", reason)
            return

        # Log executor start
        executor_type = type(executor).__name__ if executor else "None"
        StructuredLogger.log_executor_start(update.update_id, user_id, executor_type)

        # Format message for executor
        formatted_message = format_message_for_executor(update)

        # Send to executor if available with timeout
        if executor:
            try:
                # Use configured timeout
                timeout = bot_config.executor_timeout or EXECUTOR_TIMEOUT

                # Use asyncio.wait_for to prevent blocking indefinitely
                await asyncio.wait_for(
                    asyncio.to_thread(executor.run, formatted_message), timeout=timeout
                )

                # Log successful completion
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                StructuredLogger.log_executor_complete(update.update_id, duration_ms, True)

            except asyncio.TimeoutError:
                StructuredLogger.log_executor_error(
                    update.update_id, "timeout", f"Executor timed out after {timeout} seconds"
                )

            except Exception as executor_error:
                StructuredLogger.log_executor_error(
                    update.update_id, "executor_error", str(executor_error)
                )
        else:
            # No executor configured
            StructuredLogger.log_executor_error(
                update.update_id, "no_executor", "No executor configured to handle the update"
            )

    except Exception as processing_error:
        # Log processing errors
        StructuredLogger.log_executor_error(
            update.update_id, "processing_error", str(processing_error)
        )