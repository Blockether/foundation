"""Tests for Telegram webhook handler utilities."""

import asyncio
import json
from typing import Any, cast

import pytest
from agno.agent import Agent
from fastapi import APIRouter, BackgroundTasks, FastAPI
from fastapi.testclient import TestClient

from blockether_foundation.os.interfaces.telegram import handlers
from blockether_foundation.os.interfaces.telegram.models import BotConfig, Update


def _make_update() -> Update:
    return Update(update_id=1, message={"from": {"id": 123}})


def _make_bot_config() -> BotConfig:
    return BotConfig(name="test-bot", token="A" * 32)


def _capture_error_logs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []
    original_error = handlers.logger.error

    def capture(message: str, *args: Any, **kwargs: Any) -> None:
        formatted = message % args if args else message
        messages.append(str(formatted))
        original_error(message, *args, **kwargs)

    monkeypatch.setattr(handlers.logger, "error", capture)
    return messages


@pytest.mark.asyncio
async def test_run_process_update_sync_on_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    event = asyncio.Event()

    async def fake_process_update_async(update: Update, executor, bot_config: BotConfig) -> None:
        assert update.update_id == 1
        assert bot_config.name == "test-bot"
        event.set()

    monkeypatch.setattr(handlers, "process_update_async", fake_process_update_async)

    # Call from within an active event loop; should schedule task without calling asyncio.run
    handlers._run_process_update_sync(_make_update(), executor=None, bot_config=_make_bot_config())

    await asyncio.wait_for(event.wait(), timeout=0.5)


def test_run_process_update_sync_without_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def fake_process_update_async(update: Update, executor, bot_config: BotConfig) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(handlers, "process_update_async", fake_process_update_async)

    handlers._run_process_update_sync(_make_update(), executor=None, bot_config=_make_bot_config())

    assert called


def test_extract_user_id_variants() -> None:
    message_update = Update(update_id=1, message={"from": {"id": 42}})
    callback_update = Update(update_id=2, callback_query={"from": {"id": "99"}})
    empty_update = Update(update_id=3)

    assert handlers.extract_user_id(message_update) == 42
    assert handlers.extract_user_id(callback_update) == 99
    assert handlers.extract_user_id(empty_update) is None


def test_is_user_allowed_and_denied() -> None:
    config = BotConfig(name="test", token="T", allowlist_user_ids=["1"], denylist_user_ids=["2"])

    assert handlers.is_user_allowed(1, config) is True
    assert handlers.is_user_allowed(2, config) is False
    assert handlers.is_user_allowed(3, config) is False


def test_get_access_denied_reason() -> None:
    config = BotConfig(name="test", token="T", allowlist_user_ids=["5"], denylist_user_ids=["7"])

    assert handlers.get_access_denied_reason(7, config) == "User 7 is in denylist"
    assert handlers.get_access_denied_reason(6, config) == "User 6 not in allowlist"
    assert (
        handlers.get_access_denied_reason(9, BotConfig(name="test", token="T"))
        == "User 9 access denied"
    )


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

    fallback = Update(update_id=4)
    assert "unsupported" in handlers.format_message_for_executor(fallback)


class DummyExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.should_raise: Exception | None = None

    def run(self, message: str) -> None:
        if self.should_raise:
            raise self.should_raise
        self.calls.append(message)


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


@pytest.mark.asyncio
async def test_process_update_async_denied_user(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = _capture_error_logs(monkeypatch)
    bot_config = BotConfig(name="test", token="T", allowlist_user_ids=["1"])
    update = Update(update_id=1, message={"from": {"id": 2}})

    await handlers.process_update_async(update, cast(Agent, DummyExecutor()), bot_config)

    assert any("access_denied" in message for message in messages)


@pytest.mark.asyncio
async def test_process_update_async_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _capture_error_logs(monkeypatch)
    executor = DummyExecutor()

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> None:
        func(*args, **kwargs)

    async def fake_wait_for(coro, timeout):  # type: ignore[override]
        await coro
        raise TimeoutError("boom")

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    update = Update(update_id=1, message={"from": {"id": 1}})
    await handlers.process_update_async(update, cast(Agent, executor), _make_bot_config())

    assert any("timeout" in message for message in messages)


@pytest.mark.asyncio
async def test_process_update_async_executor_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = _capture_error_logs(monkeypatch)
    executor = DummyExecutor()
    executor.should_raise = ValueError("boom")

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> None:
        func(*args, **kwargs)

    async def fake_wait_for(coro, timeout):  # type: ignore[override]
        await coro

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    update = Update(update_id=1, message={"from": {"id": 1}})
    await handlers.process_update_async(update, cast(Agent, executor), _make_bot_config())

    assert any("executor_error" in message for message in messages)


@pytest.mark.asyncio
async def test_process_update_async_without_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = _capture_error_logs(monkeypatch)
    update = Update(update_id=1, message={"from": {"id": 1}})
    await handlers.process_update_async(update, None, _make_bot_config())
    assert any("no_executor" in message for message in messages)


@pytest.mark.asyncio
async def test_process_update_async_missing_user_id(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = _capture_error_logs(monkeypatch)
    update = Update(update_id=42)

    await handlers.process_update_async(update, cast(Agent, DummyExecutor()), _make_bot_config())

    assert any("no_user_id" in message for message in messages)


@pytest.mark.asyncio
async def test_process_update_async_processing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = _capture_error_logs(monkeypatch)

    def raise_processing_error(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("explode")

    monkeypatch.setattr(handlers, "is_user_allowed", raise_processing_error)

    update = Update(update_id=7, message={"from": {"id": 1}})
    await handlers.process_update_async(update, cast(Agent, DummyExecutor()), _make_bot_config())

    assert any("processing_error" in message for message in messages)


class DummyScheduler:
    def __init__(self) -> None:
        self.tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self.done_calls = 0

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))

    def done(self) -> None:
        self.done_calls += 1

    @property
    def is_enabled(self) -> bool:
        return True


def test_schedule_update_processing_with_scheduler() -> None:
    scheduler = DummyScheduler()
    update = _make_update()
    handlers._schedule_update_processing(
        update=update,
        executor=None,
        bot_config=_make_bot_config(),
        background_tasks=BackgroundTasks(),
        task_scheduler=scheduler,
    )
    assert scheduler.tasks
    func, args, kwargs = scheduler.tasks[0]
    assert func is handlers._run_process_update_sync


def test_schedule_update_processing_without_scheduler() -> None:
    background_tasks = BackgroundTasks()
    handlers._schedule_update_processing(
        update=_make_update(),
        executor=None,
        bot_config=_make_bot_config(),
        background_tasks=background_tasks,
        task_scheduler=None,
    )
    assert background_tasks.tasks


def test_notify_scheduler_done_with_background_tasks() -> None:
    scheduler = DummyScheduler()
    background_tasks = BackgroundTasks()
    handlers._notify_scheduler_done(scheduler, background_tasks)
    assert len(background_tasks.tasks) == 1


def test_notify_scheduler_done_direct_call() -> None:
    scheduler = DummyScheduler()
    handlers._notify_scheduler_done(scheduler, None)
    assert scheduler.done_calls == 1


def test_split_message_for_telegram_handles_long_text() -> None:
    long_paragraph = "A" * 2500
    text = f"{long_paragraph}\n\n{long_paragraph}\n\n{long_paragraph}"

    parts = handlers._split_message_for_telegram(text, max_length=3000)

    assert len(parts) == 3
    assert all(len(part) <= 3000 for part in parts)
    recombined = "".join(part.replace("\n\n", "") for part in parts)
    assert recombined == text.replace("\n\n", "")


@pytest.mark.asyncio
async def test_send_telegram_message_splits_and_sends_multiple_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_payloads: list[str] = []
    sleep_calls: list[float] = []

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> None:
        func(*args, **kwargs)

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    class DummyResponse:
        def __enter__(self):  # noqa: D401
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: D401
            return False

    def fake_urlopen(request, timeout=10):  # type: ignore[override]
        payload = json.loads(request.data.decode("utf-8"))
        sent_payloads.append(payload["text"])
        return DummyResponse()

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        handlers, "_split_message_for_telegram", lambda text, max_length=4000: ["part1", "part2"]
    )
    monkeypatch.setattr(handlers.urllib_request, "urlopen", fake_urlopen)

    await handlers._send_telegram_message("token", 123, "ignored")

    assert sent_payloads == ["part1", "part2"]
    assert len(sleep_calls) == 1


def _build_test_app(
    bot_config: BotConfig, monkeypatch: pytest.MonkeyPatch | None = None
) -> TestClient:
    router = APIRouter(prefix=f"/telegram/{bot_config.name}")

    if monkeypatch is not None:
        monkeypatch.setattr(handlers, "_schedule_update_processing", lambda **kwargs: None)

    app = FastAPI()
    app.include_router(handlers.attach_routes(router, executor=None, bot_config=bot_config))
    return TestClient(app)


def test_webhook_success(monkeypatch: pytest.MonkeyPatch) -> None:
    bot_config = BotConfig(name="bot", token="T", webhook_secret="secret")
    called = {}

    def fake_schedule(**kwargs):
        called["update_id"] = kwargs["update"].update_id

    monkeypatch.setattr(handlers, "_schedule_update_processing", fake_schedule)
    client = _build_test_app(bot_config)

    response = client.post(
        f"/telegram/{bot_config.name}/webhook",
        json={"update_id": 10, "message": {"from": {"id": 5}}},
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
    )

    assert response.status_code == 200
    assert called["update_id"] == 10


def test_webhook_rejects_bad_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_test_app(BotConfig(name="bot", token="T", webhook_secret="secret"), monkeypatch)
    response = client.post(
        "/telegram/bot/webhook",
        json={"update_id": 1, "message": {"from": {"id": 1}}},
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
    )
    assert response.status_code == 401


def test_webhook_rejects_large_request(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_test_app(BotConfig(name="bot", token="T"), monkeypatch)
    response = client.post(
        "/telegram/bot/webhook",
        json={"update_id": 1, "message": {"from": {"id": 1}}},
        headers={"Content-Length": str(handlers.MAX_WEBHOOK_SIZE + 1)},
    )
    assert response.status_code == 413


def test_webhook_invalid_update(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_test_app(BotConfig(name="bot", token="T"), monkeypatch)
    response = client.post(
        "/telegram/bot/webhook",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_webhook_handles_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(handlers, "_schedule_update_processing", boom)
    client = _build_test_app(BotConfig(name="bot", token="T"))
    response = client.post(
        "/telegram/bot/webhook",
        json={"update_id": 1, "message": {"from": {"id": 1}}},
    )
    assert response.status_code == 500


def test_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _build_test_app(BotConfig(name="bot", token="T"), monkeypatch)
    response = client.get("/telegram/bot/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_extract_chat_id_from_callback_query() -> None:
    """Test extracting chat_id from callback query updates."""
    update = Update(
        update_id=1,
        callback_query={
            "message": {
                "chat": {"id": 456}
            }
        }
    )
    assert handlers.extract_chat_id(update) == 456


def test_extract_chat_id_from_callback_query_without_message() -> None:
    """Test extracting chat_id from callback query without message."""
    update = Update(
        update_id=1,
        callback_query={}
    )
    assert handlers.extract_chat_id(update) is None


def test_extract_chat_id_from_callback_query_without_chat() -> None:
    """Test extracting chat_id from callback query without chat."""
    update = Update(
        update_id=1,
        callback_query={
            "message": {}
        }
    )
    assert handlers.extract_chat_id(update) is None


def test_extract_executor_reply_text_empty_string() -> None:
    """Test _extract_executor_reply_text with empty string response."""
    response = ""
    assert handlers._extract_executor_reply_text(response) is None


def test_extract_executor_reply_text_whitespace_only() -> None:
    """Test _extract_executor_reply_text with whitespace-only string."""
    response = "   \n\t  "
    assert handlers._extract_executor_reply_text(response) is None


def test_extract_executor_reply_text_with_content_list() -> None:
    """Test _extract_executor_reply_text with content list."""
    class MockResponse:
        content = ["item1", "item2", "item3"]

    response = MockResponse()
    result = handlers._extract_executor_reply_text(response)
    assert result == "item1\n\nitem2\n\nitem3"


def test_extract_executor_reply_text_with_text_attribute() -> None:
    """Test _extract_executor_reply_text with text attribute."""
    class MockResponse:
        content = None
        text = "Some text content"

    response = MockResponse()
    result = handlers._extract_executor_reply_text(response)
    assert result == "Some text content"


def test_force_split_text_no_breakpoints() -> None:
    """Test _force_split_text with text that has no natural breakpoints."""
    long_word = "a" * 50
    result = handlers._force_split_text(long_word, 20)
    assert len(result) == 3
    assert sum(len(part) for part in result) == 50
    assert all(len(part) <= 20 for part in result)


def test_split_message_normalization_whitespace() -> None:
    """Test _split_message_for_telegram normalizes whitespace."""
    text = "  Line1\n\n\nLine2\n\nLine3  "
    result = handlers._split_message_for_telegram(text, 1000)
    assert len(result) == 1
    # The function strips outer whitespace and splits by double newlines, preserving empty paragraphs
    assert result[0] == "Line1\n\n\nLine2\n\nLine3"


def test_send_telegram_message_empty_parts() -> None:
    """Test _send_telegram_message with empty parts after splitting."""
    def mock_split_empty(text, max_length=None):
        return []

    import asyncio
    with pytest.MonkeyPatch().context() as m:
        m.setattr(handlers, "_split_message_for_telegram", mock_split_empty)

        async def test_send():
            await handlers._send_telegram_message("test_token", 123, "test")

        asyncio.run(test_send())  # Should not raise any errors


def test_extract_executor_reply_text_from_none() -> None:
    """Test _extract_executor_reply_text with None response."""
    assert handlers._extract_executor_reply_text(None) is None
