"""Tests for Telegram webhook handler utilities."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, cast
from unittest.mock import Mock

import pytest
from agno.agent import Agent
from agno.team import Team
from agno.workflow import Workflow
from fastapi import APIRouter, BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from blockether_foundation.asr.common import AudioTranscriberProtocol
from blockether_foundation.os.interfaces.telegram import handlers
from blockether_foundation.os.interfaces.telegram.models import BotConfig, Update

# Test Constants
TEST_UPDATE_ID = 1
TEST_USER_ID = 123
TEST_TIMEOUT = 0.5
TEST_TOKEN_LENGTH = 32
TEST_MESSAGE_USER_ID = 42
TEST_CALLBACK_USER_ID = 99
TEST_EMPTY_UPDATE_ID = 3
TEST_MESSAGE_UPDATE_ID = 1
TEST_CALLBACK_UPDATE_ID = 2
TEST_ALLOWLIST_USER_ID = "1"
TEST_DENYLIST_USER_ID = "2"
TEST_OTHER_USER_ID = 3
TEST_ALLOWLIST_USER_ID_2 = "5"
TEST_DENYLIST_USER_ID_2 = "7"
TEST_OTHER_USER_ID_2 = 6
TEST_OTHER_USER_ID_3 = 9
TEST_ALICE_ID = 10
# HTTP Status Constants
HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
HTTP_PAYLOAD_TOO_LARGE = 413
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_SERVER_ERROR = 500
# Split test constants
EXPECTED_SPLIT_PARTS = 3
MAX_MESSAGE_LENGTH = 3000


def _make_update() -> Update:
    return Update(update_id=TEST_UPDATE_ID, message={"from": {"id": TEST_USER_ID}})


def _make_bot_config() -> BotConfig:
    return BotConfig(name="test-bot", token="A" * TEST_TOKEN_LENGTH)


def _capture_error_logs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []
    original_error = handlers.logger.error

    def capture(message: str, *args: Any, **kwargs: Any) -> None:
        formatted = message % args if args else message
        messages.append(str(formatted))
        original_error(message, *args, **kwargs)

    monkeypatch.setattr(handlers.logger, "error", capture)
    return messages


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_process_update_sync_on_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    event = asyncio.Event()

    async def fake_process_update_async(
        update: Update,
        executor: Agent | Team | Workflow | None,
        bot_config: BotConfig,
        audio_transcriber: AudioTranscriberProtocol | None = None,
        use_async_executor: bool = True,
    ) -> None:
        assert update.update_id == TEST_UPDATE_ID
        assert bot_config.name == "test-bot"
        event.set()

    monkeypatch.setattr(handlers, "process_update_async", fake_process_update_async)

    handlers._run_process_update_sync(_make_update(), executor=None, bot_config=_make_bot_config())  # type: ignore[attr-defined]

    await asyncio.wait_for(event.wait(), timeout=TEST_TIMEOUT)


@pytest.mark.unit
def test_run_process_update_sync_without_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def fake_process_update_async(
        update: Update,
        executor: Agent | Team | Workflow | None,
        bot_config: BotConfig,
        audio_transcriber: AudioTranscriberProtocol | None = None,
        use_async_executor: bool = True,
    ) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(handlers, "process_update_async", fake_process_update_async)

    handlers._run_process_update_sync(_make_update(), executor=None, bot_config=_make_bot_config())  # type: ignore[attr-defined]

    assert called


@pytest.mark.unit
def test_extract_user_id_variants() -> None:
    message_update = Update(
        update_id=TEST_MESSAGE_UPDATE_ID, message={"from": {"id": TEST_MESSAGE_USER_ID}}
    )
    callback_update = Update(
        update_id=TEST_CALLBACK_UPDATE_ID,
        callback_query={"from": {"id": str(TEST_CALLBACK_USER_ID)}},
    )
    empty_update = Update(update_id=TEST_EMPTY_UPDATE_ID)

    assert handlers.extract_user_id(message_update) == TEST_MESSAGE_USER_ID
    assert handlers.extract_user_id(callback_update) == TEST_CALLBACK_USER_ID
    assert handlers.extract_user_id(empty_update) is None


@pytest.mark.unit
def test_is_user_allowed_and_denied() -> None:
    config = BotConfig(
        name="test",
        token="T",
        allowlist_user_ids=[TEST_ALLOWLIST_USER_ID],
        denylist_user_ids=[TEST_DENYLIST_USER_ID],
    )

    assert handlers.is_user_allowed(int(TEST_ALLOWLIST_USER_ID), config) is True
    assert handlers.is_user_allowed(int(TEST_DENYLIST_USER_ID), config) is False
    assert handlers.is_user_allowed(TEST_OTHER_USER_ID, config) is False


@pytest.mark.unit
def test_get_access_denied_reason() -> None:
    config = BotConfig(name="test", token="T", allowlist_user_ids=["5"], denylist_user_ids=["7"])

    assert handlers.get_access_denied_reason(7, config) == "User 7 is in denylist"
    assert handlers.get_access_denied_reason(6, config) == "User 6 not in allowlist"
    assert (
        handlers.get_access_denied_reason(9, BotConfig(name="test", token="T"))
        == "User 9 access denied"
    )


@pytest.mark.unit
def test_format_message_variants() -> None:
    message = Update(
        update_id=1,
        message={
            "from": {"first_name": "Alice", "id": 10},
            "chat": {"id": 5, "type": "private"},
            "text": "hello",
        },
    )
    assert "Alice" in handlers.format_message_for_executor(message)

    no_text = Update(
        update_id=2,
        message={
            "from": {"first_name": "Bob", "id": 11},
            "chat": {"id": 6, "type": "group"},
        },
    )
    assert "non-text" in handlers.format_message_for_executor(no_text)

    callback = Update(
        update_id=3, callback_query={"from": {"first_name": "Eve", "id": 12}, "data": "btn"}
    )
    assert "button" in handlers.format_message_for_executor(callback)

    unsupported_update = Update(update_id=4)
    assert "unsupported" in handlers.format_message_for_executor(unsupported_update)


class DummyExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, message: str, **kwargs: Any) -> None:
        self.calls.append(message)

    async def arun(self, message: str, **kwargs: Any) -> None:
        self.calls.append(message)


class ErrorRaisingExecutor:
    def __init__(self, error: Exception) -> None:
        self.calls: list[str] = []
        self.error = error

    def run(self, message: str, **kwargs: Any) -> None:
        self.calls.append(message)
        raise self.error

    async def arun(self, message: str, **kwargs: Any) -> None:
        self.calls.append(message)
        raise self.error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_update_async_success(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = DummyExecutor()

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> None:
        func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    update = Update(
        update_id=1,
        message={"from": {"id": 77}, "chat": {"id": 1, "type": "private"}, "text": "hi"},
    )
    await handlers.process_update_async(update, cast(Agent, executor), _make_bot_config())

    assert executor.calls and "hi" in executor.calls[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_update_async_denied_user(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = _capture_error_logs(monkeypatch)
    bot_config = BotConfig(name="test", token="T", allowlist_user_ids=["1"])
    update = Update(update_id=1, message={"from": {"id": 2}})

    await handlers.process_update_async(update, cast(Agent, DummyExecutor()), bot_config)

    assert any("access_denied" in message for message in messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_update_async_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _capture_error_logs(monkeypatch)
    executor = DummyExecutor()

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> None:
        func(*args, **kwargs)

    timeout_error = TimeoutError("boom")

    async def fake_wait_for(coro: Awaitable[Any], timeout: float) -> Any:
        await coro
        raise timeout_error

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    update = Update(update_id=1, message={"from": {"id": 1}})
    await handlers.process_update_async(update, cast(Agent, executor), _make_bot_config())

    assert any("timeout" in message for message in messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_update_async_executor_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _capture_error_logs(monkeypatch)
    executor = ErrorRaisingExecutor(ValueError("boom"))

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> None:
        func(*args, **kwargs)

    async def fake_wait_for(coro: Awaitable[Any], timeout: float) -> Any:
        await coro

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    update = Update(update_id=1, message={"from": {"id": 1}})
    await handlers.process_update_async(update, cast(Agent, executor), _make_bot_config())

    assert any("executor_error" in message for message in messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_update_async_without_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = _capture_error_logs(monkeypatch)
    update = Update(update_id=1, message={"from": {"id": 1}})
    await handlers.process_update_async(update, None, _make_bot_config())
    assert any("no_executor" in message for message in messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_update_async_missing_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = _capture_error_logs(monkeypatch)
    update = Update(update_id=42)

    await handlers.process_update_async(update, cast(Agent, DummyExecutor()), _make_bot_config())

    assert any("no_user_id" in message for message in messages)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_update_async_processing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = _capture_error_logs(monkeypatch)
    processing_error = RuntimeError("explode")

    def raise_processing_error(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN001
        raise processing_error

    monkeypatch.setattr(handlers, "is_user_allowed", raise_processing_error)

    update = Update(update_id=7, message={"from": {"id": 1}})
    await handlers.process_update_async(update, cast(Agent, DummyExecutor()), _make_bot_config())

    assert any("processing_error" in message for message in messages)


class DummyScheduler:
    def __init__(self) -> None:
        self.tasks: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = []
        self.done_calls = 0

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.tasks.append((func, args, kwargs))

    def done(self) -> None:
        self.done_calls += 1

    @property
    def is_enabled(self) -> bool:
        return True


@pytest.mark.unit
def test_schedule_update_processing_with_scheduler() -> None:
    scheduler = DummyScheduler()
    update = _make_update()
    handlers._schedule_update_processing(  # type: ignore[attr-defined]
        update=update,
        executor=None,
        bot_config=_make_bot_config(),
        background_tasks=BackgroundTasks(),
        task_scheduler=scheduler,
    )
    assert scheduler.tasks
    func, args, kwargs = scheduler.tasks[0]
    assert func is handlers._run_process_update_sync  # type: ignore[attr-defined]


@pytest.mark.unit
def test_schedule_update_processing_without_scheduler() -> None:
    background_tasks = BackgroundTasks()
    handlers._schedule_update_processing(  # type: ignore[attr-defined]
        update=_make_update(),
        executor=None,
        bot_config=_make_bot_config(),
        background_tasks=background_tasks,
        task_scheduler=None,
    )
    assert background_tasks.tasks


@pytest.mark.unit
def test_notify_scheduler_done_with_background_tasks() -> None:
    scheduler = DummyScheduler()
    background_tasks = BackgroundTasks()
    handlers._notify_scheduler_done(scheduler, background_tasks)  # type: ignore[attr-defined]
    assert len(background_tasks.tasks) == 1


@pytest.mark.unit
def test_notify_scheduler_done_direct_call() -> None:
    scheduler = DummyScheduler()
    handlers._notify_scheduler_done(scheduler, None)  # type: ignore[attr-defined]
    assert scheduler.done_calls == 1


@pytest.mark.unit
def test_split_message_for_telegram_handles_long_text() -> None:
    long_paragraph = "A" * 2500
    text = f"{long_paragraph}\n\n{long_paragraph}\n\n{long_paragraph}"

    parts = handlers._split_message_for_telegram(text, max_length=MAX_MESSAGE_LENGTH)  # type: ignore[attr-defined]

    assert len(parts) == EXPECTED_SPLIT_PARTS
    assert all(len(part) <= MAX_MESSAGE_LENGTH for part in parts)
    recombined = "".join(part.replace("\n\n", "") for part in parts)
    assert recombined == text.replace("\n\n", "")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_send_telegram_message_splits_and_sends_multiple_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_payloads: list[str] = []

    class FakeResponse:
        def __init__(self):
            self.status = HTTP_OK
            self.reason = "OK"
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> bool:
            return False

    def fake_urlopen(request: Mock, timeout: float | None = None) -> Mock:
        # Extract the payload from the request data
        data = json.loads(request.data.decode("utf-8"))
        sent_payloads.append(data["text"])
        return FakeResponse()  # type: ignore[return-value]

    monkeypatch.setattr(
        handlers,
        "_split_message_for_telegram",
        lambda text: ["part1", "part2"],  # type: ignore[attr-defined]
    )
    monkeypatch.setattr(handlers.urllib.request, "urlopen", fake_urlopen)

    await handlers._send_telegram_message("token", 123, "ignored")  # type: ignore[attr-defined]

    assert sent_payloads == ["part1", "part2"]


def _build_test_app(
    bot_config: BotConfig, monkeypatch: pytest.MonkeyPatch | None = None
) -> TestClient:
    router = APIRouter(prefix=f"/telegram/{bot_config.name}")

    app = FastAPI()
    app.include_router(handlers.attach_routes(router, executor=None, bot_config=bot_config))
    return TestClient(app)


def _build_test_app_with_mock_processing(
    bot_config: BotConfig, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, dict[str, Any]]:
    """Helper to build test app with mocked processing for tracking calls."""
    called: dict[str, Any] = {}

    def fake_schedule(**kwargs: Any) -> None:
        called["update_id"] = kwargs["update"].update_id

    monkeypatch.setattr(handlers, "_schedule_update_processing", fake_schedule)
    client = _build_test_app(bot_config)

    return client, called


@pytest.mark.unit
def test_webhook_success(monkeypatch: pytest.MonkeyPatch) -> None:
    bot_config = BotConfig(name="bot", token="T", webhook_secret="secret")
    client, called = _build_test_app_with_mock_processing(bot_config, monkeypatch)

    response = client.post(
        f"/telegram/{bot_config.name}/webhook",
        json={"update_id": 10, "message": {"from": {"id": 5}}},
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
    )

    assert response.status_code == HTTP_OK
    assert called["update_id"] == 10


@pytest.mark.unit
def test_webhook_rejects_bad_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_test_app(BotConfig(name="bot", token="T", webhook_secret="secret"), monkeypatch)
    response = client.post(
        "/telegram/bot/webhook",
        json={"update_id": 1, "message": {"from": {"id": 1}}},
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
    )
    assert response.status_code == HTTP_UNAUTHORIZED


@pytest.mark.unit
def test_webhook_rejects_large_request(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_test_app(BotConfig(name="bot", token="T"), monkeypatch)
    response = client.post(
        "/telegram/bot/webhook",
        json={"update_id": 1, "message": {"from": {"id": 1}}},
        headers={"Content-Length": str(handlers.MAX_WEBHOOK_SIZE + 1)},
    )
    assert response.status_code == HTTP_PAYLOAD_TOO_LARGE


@pytest.mark.unit
def test_webhook_invalid_update(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_test_app(BotConfig(name="bot", token="T"), monkeypatch)
    response = client.post(
        "/telegram/bot/webhook",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == HTTP_BAD_REQUEST


@pytest.mark.unit
def test_webhook_handles_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_error = RuntimeError("boom")

    def boom(**kwargs: Any) -> None:  # noqa: ANN003
        raise runtime_error

    monkeypatch.setattr(handlers, "_schedule_update_processing", boom)
    client = _build_test_app(BotConfig(name="bot", token="T"))
    response = client.post(
        "/telegram/bot/webhook",
        json={"update_id": 1, "message": {"from": {"id": 1}}},
    )
    assert response.status_code == HTTP_INTERNAL_SERVER_ERROR


@pytest.mark.unit
def test_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_test_app(BotConfig(name="bot", token="T"), monkeypatch)
    response = client.get("/telegram/bot/health")
    assert response.status_code == HTTP_OK
    assert response.json()["status"] == "healthy"


@pytest.mark.unit
def test_extract_chat_id_from_callback_query() -> None:
    """Test extracting chat_id from callback query updates."""
    TEST_CHAT_ID = 456
    update = Update(update_id=1, callback_query={"message": {"chat": {"id": TEST_CHAT_ID}}})
    assert handlers.extract_chat_id(update) == TEST_CHAT_ID


@pytest.mark.unit
def test_extract_chat_id_from_callback_query_without_message() -> None:
    """Test extracting chat_id from callback query without message."""
    update = Update(update_id=1, callback_query={})
    assert handlers.extract_chat_id(update) is None


@pytest.mark.unit
def test_extract_chat_id_from_callback_query_without_chat() -> None:
    """Test extracting chat_id from callback query without chat."""
    update = Update(update_id=1, callback_query={"message": {}})
    assert handlers.extract_chat_id(update) is None


@pytest.mark.unit
def test_extract_executor_reply_text_empty_string() -> None:
    """Test _extract_executor_reply_text with empty string response."""
    response = ""
    assert handlers._extract_executor_reply_text(response) is None  # type: ignore[attr-defined]


@pytest.mark.unit
def test_extract_executor_reply_text_whitespace_only() -> None:
    """Test _extract_executor_reply_text with whitespace-only string."""
    response = "   \n\t  "
    assert handlers._extract_executor_reply_text(response) is None  # type: ignore[attr-defined]


@pytest.mark.unit
def test_extract_executor_reply_text_with_content_list() -> None:
    """Test _extract_executor_reply_text with content list."""

    class MockResponse:
        content = ["item1", "item2", "item3"]

    response = MockResponse()
    result = handlers._extract_executor_reply_text(response)  # type: ignore[attr-defined]
    assert result == "item1\n\nitem2\n\nitem3"


@pytest.mark.unit
def test_extract_executor_reply_text_with_text_attribute() -> None:
    """Test _extract_executor_reply_text with text attribute."""

    class MockResponse:
        content = None
        text = "Some text content"

    response = MockResponse()
    result = handlers._extract_executor_reply_text(response)  # type: ignore[attr-defined]
    assert result == "Some text content"


@pytest.mark.unit
def test_force_split_text_no_breakpoints() -> None:
    """Test _force_split_text with text that has no natural breakpoints."""
    long_word = "a" * 50
    expected_parts = 3
    expected_total_length = 50
    max_split_length = 20
    result = handlers._force_split_text(long_word, max_split_length)  # type: ignore[attr-defined]
    assert len(result) == expected_parts
    assert sum(len(part) for part in result) == expected_total_length
    assert all(len(part) <= max_split_length for part in result)


@pytest.mark.unit
def test_split_message_normalization_whitespace() -> None:
    """Test _split_message_for_telegram normalizes whitespace."""
    text = "  Line1\n\n\nLine2\n\nLine3  "
    result = handlers._split_message_for_telegram(text, 1000)  # type: ignore[attr-defined]
    assert len(result) == 1
    # The function strips outer whitespace and splits by double newlines, preserving empty paragraphs
    assert result[0] == "Line1\n\n\nLine2\n\nLine3"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_send_telegram_message_empty_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _send_telegram_message with empty parts after splitting."""

    def mock_split_empty(text: str, max_length: int | None = None) -> list[str]:
        return []

    monkeypatch.setattr(handlers, "_split_message_for_telegram", mock_split_empty)

    # This should not raise any errors even when split returns empty list
    await handlers._send_telegram_message("test_token", 123, "test")  # type: ignore[attr-defined]

    # Test passes if no exception is raised
    assert True


@pytest.mark.unit
def test_extract_executor_reply_text_from_none() -> None:
    """Test _extract_executor_reply_text with None response."""
    assert handlers._extract_executor_reply_text(None) is None  # type: ignore[attr-defined]


@pytest.mark.unit
def test_bot_config_webhook_url_property() -> None:
    """Test BotConfig webhook_url property (covers line 22 in models.py)."""
    config = BotConfig(
        name="my_test_bot",
        token="A" * 46,
        webhook_secret="test_secret",
    )

    # Test webhook_url property
    assert config.webhook_url == "/telegram/my_test_bot/webhook"

    # Test with special characters in name
    config.name = "bot-123_test"
    assert config.webhook_url == "/telegram/bot-123_test/webhook"
