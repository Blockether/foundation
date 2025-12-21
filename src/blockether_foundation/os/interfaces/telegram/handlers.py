# ignore-development
"""Minimal webhook handlers for Telegram interface."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from agno.agent import Agent
from agno.team import Team
from agno.utils.log import logger
from agno.workflow import Workflow
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from blockether_foundation.audio import AudioTranscriber

from .models import BotConfig, HealthResponse, Update, WebhookResponse

# Type aliases for agno executor return types
ExecutorResult = str | Any

# Constants
MAX_WEBHOOK_SIZE = 1024 * 1024  # 1MB
EXECUTOR_TIMEOUT = 30  # seconds
TELEGRAM_API_BASE_URL = "https://api.telegram.org"
TELEGRAM_MAX_MESSAGE_LENGTH = 4000


@runtime_checkable
class BackgroundTaskScheduler(Protocol):
    """Protocol describing objects capable of scheduling synchronous tasks."""

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Schedule a callable for execution."""
        ...

    def done(self) -> None:
        """Signal that the current invoke/request cycle has completed."""
        ...


# public-api
def attach_routes(
    router: APIRouter,
    executor: Agent | Team | Workflow | None,
    bot_config: BotConfig,
    task_scheduler: BackgroundTaskScheduler | None = None,
    scheduler_managed_by_middleware: bool = False,
) -> APIRouter:
    # ignore-development
    """Attach minimal Telegram webhook routes to the router."""
    logger.info(f"Attaching routes for bot: {bot_config.name}")  # type: ignore[arg-type]

    @router.post("/webhook", response_model=WebhookResponse)
    async def webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        x_telegram_bot_api_secret_token: str | None = None,
    ) -> WebhookResponse:
        """Handle Telegram webhook updates."""
        start_time = datetime.now(UTC)
        logger.debug(f"Webhook received for bot {bot_config.name}")  # type: ignore[arg-type]

        try:
            # 1. Get webhook signature header if configured
            if bot_config.webhook_secret:
                # Get the header from request
                x_telegram_bot_api_secret_token = request.headers.get(
                    "X-Telegram-Bot-Api-Secret-Token"
                )
                if x_telegram_bot_api_secret_token != bot_config.webhook_secret:
                    logger.error("Webhook error (status=401): Unauthorized webhook request")
                    raise HTTPException(status_code=401, detail="Unauthorized")

            # 2. Check request size
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_WEBHOOK_SIZE:
                logger.error(
                    f"Webhook error (status=413): Request too large: {content_length} bytes"
                )
                raise HTTPException(status_code=413, detail="Request too large")

            # 3. Parse and validate update from Telegram
            try:
                update_data = await request.json()
                update = Update(**update_data)
            except Exception as validation_error:
                logger.error(
                    f"Webhook error (status=400): Invalid update format: {str(validation_error)}"
                )
                raise HTTPException(status_code=400, detail="Invalid update format") from None

            # 4. Basic validation and logging
            user_id = extract_user_id(update)
            logger.info(f"Webhook received: update_id={update.update_id}, user_id={user_id}")  # type: ignore[arg-type]

            # 5. Process the update in background
            _schedule_update_processing(
                update=update,
                executor=executor,
                bot_config=bot_config,
                background_tasks=background_tasks,
                task_scheduler=task_scheduler,
            )

            logger.debug(f"Webhook for bot {bot_config.name} queued for processing")  # type: ignore[arg-type]
            return WebhookResponse(
                status="ok",
                update_id=update.update_id,
                processed_at=start_time.isoformat(),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Webhook error (status=500): Unexpected webhook error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error") from None
        finally:
            if not scheduler_managed_by_middleware:
                _notify_scheduler_done(task_scheduler, background_tasks)

    @router.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        logger.debug(f"Health check requested for bot {bot_config.name}")  # type: ignore[arg-type]
        try:
            return HealthResponse(status="healthy", timestamp=datetime.now(UTC).isoformat())
        finally:
            if not scheduler_managed_by_middleware:
                _notify_scheduler_done(task_scheduler, None)

    logger.info(f"Routes attached successfully for bot {bot_config.name}")  # type: ignore[arg-type]
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


def extract_chat_id(update: Update) -> int | None:
    """Extract chat ID from update for sending replies."""

    if update.message and "chat" in update.message:
        chat_id = update.message["chat"].get("id")
        return int(chat_id) if chat_id is not None else None

    if update.callback_query:
        message = update.callback_query.get("message", {})
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        return int(chat_id) if chat_id is not None else None

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


def format_message_for_executor(update: Update, transcription: str | None = None) -> str:
    """Format a Telegram update for the executor."""
    if update.message:
        user_info = update.message.get("from", {})
        chat_info = update.message.get("chat", {})
        message_text = update.message.get("text", "") or update.message.get("caption", "")

        # Use transcription if available
        if transcription:
            message_text = f"{message_text}\n[Audio Transcription]: {transcription}".strip()

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

    start_time = datetime.now(UTC)
    user_id = extract_user_id(update)

    if not user_id:
        logger.error(
            f"Executor error: update_id={update.update_id}, error_type=no_user_id, error=Could not extract user ID"
        )
        return

    try:
        # Validate user access with allowlist/denylist logic
        if not is_user_allowed(user_id, bot_config):
            reason = get_access_denied_reason(user_id, bot_config)
            logger.error(
                f"Executor error: update_id={update.update_id}, error_type=access_denied, error={reason}"
            )
            return

        # Log executor start
        executor_type = type(executor).__name__ if executor else "None"
        logger.info(  # type: ignore[arg-type]
            f"Executor start: update_id={update.update_id}, user_id={user_id}, executor_type={executor_type}"
        )

        # Check for audio/voice and transcribe if needed
        transcription = await _transcribe_audio_if_present(update, bot_config.token)

        # Format message for executor
        formatted_message = format_message_for_executor(update, transcription)
        chat_id = extract_chat_id(update)

        # Configure timeout
        timeout = bot_config.executor_timeout or EXECUTOR_TIMEOUT

        # Send to executor if available with timeout
        if executor:
            try:
                # Use asyncio.wait_for to prevent blocking indefinitely
                # Wrap the sync executor.run call in a lambda for asyncio.to_thread
                result: ExecutorResult = await asyncio.wait_for(
                    asyncio.to_thread(lambda: executor.run(formatted_message)),  # type: ignore[arg-type]
                    timeout=timeout,
                )

                # Log successful completion
                duration_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
                logger.info(  # type: ignore[arg-type]
                    f"Executor complete: update_id={update.update_id}, duration_ms={duration_ms}, success=True"
                )

                reply_text = _extract_executor_reply_text(result)
                if chat_id is None:
                    logger.error(
                        "Executor complete but chat_id missing; unable to deliver Telegram response"
                    )
                elif not reply_text:
                    logger.warning(
                        f"Executor complete but reply text empty; skipping Telegram send for chat_id={chat_id}"
                    )
                else:
                    await _send_telegram_message(bot_config.token, chat_id, reply_text)

            except TimeoutError:
                logger.error(
                    f"Executor error: update_id={update.update_id}, error_type=timeout, error=Executor timed out after {timeout} seconds"
                )

            except Exception as executor_error:
                logger.error(
                    f"Executor error: update_id={update.update_id}, error_type=executor_error, error={str(executor_error)}"
                )
        else:
            # No executor configured
            logger.error(
                f"Executor error: update_id={update.update_id}, error_type=no_executor, error=No executor configured to handle the update"
            )

    except Exception as processing_error:
        # Log processing errors
        logger.error(
            f"Executor error: update_id={update.update_id}, error_type=processing_error, error={str(processing_error)}"
        )


def _schedule_update_processing(
    *,
    update: Update,
    executor: Agent | Team | Workflow | None,
    bot_config: BotConfig,
    background_tasks: BackgroundTasks,
    task_scheduler: BackgroundTaskScheduler | None,
) -> None:
    """Dispatch task via configured scheduler or FastAPI background tasks."""

    if task_scheduler:
        task_scheduler.add_task(_run_process_update_sync, update, executor, bot_config)
    else:
        background_tasks.add_task(
            process_update_async, update=update, executor=executor, bot_config=bot_config
        )


def _run_process_update_sync(
    update: Update, executor: Agent | Team | Workflow | None, bot_config: BotConfig
) -> None:
    """Execute the async update processor inside a fresh event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(
            process_update_async(update=update, executor=executor, bot_config=bot_config)
        )
    else:
        asyncio.run(process_update_async(update=update, executor=executor, bot_config=bot_config))


def _notify_scheduler_done(
    task_scheduler: BackgroundTaskScheduler | None,
    background_tasks: BackgroundTasks | None,
) -> None:
    """Invoke the scheduler's done callback, optionally via FastAPI background tasks."""

    if not task_scheduler:
        return

    def _safe_done() -> None:
        try:
            task_scheduler.done()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to signal background scheduler completion: %s", exc)

    if background_tasks:
        background_tasks.add_task(_safe_done)
    else:
        _safe_done()


def _extract_executor_reply_text(response: Any) -> str | None:
    """Coerce the executor response into a Telegram-friendly string."""

    if response is None:
        return None

    if isinstance(response, str):
        text = response.strip()
        return text or None

    # Define protocols for objects with content or text attributes
    @runtime_checkable
    class HasContent(Protocol):
        content: Any

    @runtime_checkable
    class HasText(Protocol):
        text: Any

    # Check for content attribute using isinstance
    if isinstance(response, HasContent):
        content = response.content
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        elif isinstance(content, list):
            parts: list[str] = [str(item).strip() for item in content if str(item).strip()]  # type: ignore[arg-type]
            if parts:
                return "\n\n".join(parts)

    # Check for text attribute using isinstance
    if isinstance(response, HasText):
        text_attr = response.text
        if isinstance(text_attr, str) and text_attr.strip():
            return text_attr.strip()

    text = str(response).strip()
    return text or None


def _force_split_text(text: str, max_length: int) -> list[str]:
    return [text[i : i + max_length].strip() for i in range(0, len(text), max_length)]


def _split_message_for_telegram(
    text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH
) -> list[str]:
    """Split text into Telegram-safe chunks while keeping paragraphs intact when possible."""

    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= max_length:
        return [cleaned]

    parts: list[str] = []
    current = ""
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    segments = paragraphs if paragraphs else [cleaned]

    for segment in segments:
        working = segment

        if len(working) > max_length:
            if current:
                parts.append(current.strip())
                current = ""
            parts.extend(_force_split_text(working, max_length))
            continue

        if not current:
            current = working
            continue

        candidate = f"{current}\n\n{working}"
        if len(candidate) <= max_length:
            current = candidate
        else:
            parts.append(current.strip())
            current = working

    if current.strip():
        parts.append(current.strip())

    normalized: list[str] = []
    for part in parts:
        if len(part) <= max_length:
            normalized.append(part)
        else:
            normalized.extend(_force_split_text(part, max_length))

    return [part for part in normalized if part]


async def _send_telegram_message(token: str, chat_id: int, text: str) -> None:
    """Send a message back to Telegram using the Bot API."""

    parts = _split_message_for_telegram(text)
    if not parts:
        logger.warning("Skipping Telegram reply: empty response text")
        return

    url = f"{TELEGRAM_API_BASE_URL}/bot{token}/sendMessage"
    headers = {"Content-Type": "application/json"}

    total_parts = len(parts)
    for index, part in enumerate(parts, start=1):
        payload: dict[str, Any] = {"chat_id": chat_id, "text": part}
        try:
            # Create request with timeout
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
            )

            # Send request with timeout
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status >= 400:
                    raise urllib.error.HTTPError(
                        url, response.status, response.reason, response.headers, None
                    )

            if total_parts == 1:
                logger.info(f"Sent Telegram reply to chat_id={chat_id}")  # type: ignore[arg-type]
            else:
                logger.info(  # type: ignore[arg-type]
                    f"Sent Telegram reply part {index}/{total_parts} to chat_id={chat_id}"
                )
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            logger.error(f"Failed to send Telegram reply to chat_id={chat_id}: {exc}")
            break


async def _download_file_from_telegram(token: str, file_id: str) -> bytes | None:
    """Download a file from Telegram and return the content as bytes."""
    try:
        # 1. Get file path info
        get_file_url = f"{TELEGRAM_API_BASE_URL}/bot{token}/getFile"
        payload = json.dumps({"file_id": file_id}).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        req = urllib.request.Request(get_file_url, data=payload, headers=headers, method="POST")

        def _get_file_info():
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))

        data = await asyncio.to_thread(_get_file_info)

        if not data.get("ok"):
            logger.error(f"Failed to get file info: {data}")
            return None

        file_path = data["result"]["file_path"]
        download_url = f"{TELEGRAM_API_BASE_URL}/file/bot{token}/{file_path}"

        # 2. Download file to memory
        def _download():
            with urllib.request.urlopen(download_url, timeout=30) as response:
                return response.read()

        return await asyncio.to_thread(_download)

    except Exception as e:
        logger.error(f"Error downloading file {file_id}: {e}")
        return None


async def _transcribe_audio_if_present(update: Update, token: str) -> str | None:
    """Check for audio in update and transcribe if present."""
    if not update.message:
        return None

    file_id = None

    # Check for voice note
    if "voice" in update.message:
        file_id = update.message["voice"].get("file_id")
        logger.info(f"Processing voice message: {file_id}")  # type: ignore[arg-type]

    # Check for audio file
    elif "audio" in update.message:
        file_id = update.message["audio"].get("file_id")
        logger.info(f"Processing audio message: {file_id}")  # type: ignore[arg-type]

    if not file_id:
        return None

    audio_data = await _download_file_from_telegram(token, file_id)
    if not audio_data:
        return None

    try:
        transcription = await AudioTranscriber.get_instance().transcribe(audio_data)
        if transcription:
            logger.info(f"Transcription successful: {transcription.text[:50]}...")  # type: ignore[arg-type]
            return transcription.text
        return None
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None
