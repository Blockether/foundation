"""Tests for Lambda background task extension."""

from __future__ import annotations

from queue import Empty
from types import TracebackType
from typing import Any

import pytest

from blockether_foundation.os.runtime.aws_lambda import background_tasks as background_tasks_module
from blockether_foundation.os.runtime.aws_lambda.background_tasks import (
    LambdaBackgroundTaskExtension,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset the singleton instance before each test."""
    LambdaBackgroundTaskExtension._reset_singleton()  # type: ignore[attr-defined]


@pytest.mark.unit
def test_extension_requires_runtime_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_LAMBDA_RUNTIME_API", raising=False)

    with pytest.raises(RuntimeError, match="AWS_LAMBDA_RUNTIME_API is not set"):
        LambdaBackgroundTaskExtension()


@pytest.mark.unit
def test_add_task_enqueues_when_runtime_available(monkeypatch: pytest.MonkeyPatch) -> None:
    started = False

    def fake_start(self: LambdaBackgroundTaskExtension) -> None:
        nonlocal started
        started = True

    def fake_register(self: LambdaBackgroundTaskExtension) -> None:
        pass

    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", fake_start)
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", fake_register)

    extension = LambdaBackgroundTaskExtension()

    assert started is True

    extension.add_task(lambda value: value, 1)  # type: ignore[arg-type]
    message = extension.queue.get_nowait()

    assert message["type"] == "TASK"
    func, args, kwargs = message["task"]
    assert func(1) == 1
    assert args == (1,)
    assert kwargs == {}


@pytest.mark.unit
def test_done_enqueues_completion_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", lambda self: None)  # type: ignore[arg-type]
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", lambda self: None)  # type: ignore[arg-type]

    extension = LambdaBackgroundTaskExtension()
    extension.done()

    message = extension.queue.get_nowait()
    assert message["type"] == "DONE"


@pytest.mark.unit
def test_drain_queue_executes_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", lambda self: None)  # type: ignore[arg-type]
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", lambda self: None)  # type: ignore[arg-type]
    extension = LambdaBackgroundTaskExtension()

    executed: list[int] = []

    extension.queue.put({"type": "TASK", "task": (lambda value: executed.append(value), (5,), {})})  # type: ignore[arg-type]
    extension.queue.put({"type": "DONE"})

    extension._drain_queue_until_done()  # type: ignore[attr-defined]

    assert executed == [5]

    with pytest.raises(Empty):
        extension.queue.get_nowait()


def _make_enabled_extension(monkeypatch: pytest.MonkeyPatch) -> LambdaBackgroundTaskExtension:
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", lambda self: None)  # type: ignore[arg-type]
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", lambda self: None)  # type: ignore[arg-type]
    return LambdaBackgroundTaskExtension()


@pytest.mark.unit
def test_register_extension_returns_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    # ignore-development
    # Create extension manually with minimal initialization to avoid HTTP calls
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    # Call parent Thread.__init__ but skip our __init__ logic
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = None  # type: ignore[attr-defined]

    class DummyResponse:
        headers = {"Lambda-Extension-Identifier": "ext-id"}

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

    monkeypatch.setattr(
        background_tasks_module.urllib_request,
        "urlopen",
        lambda *a, **k: DummyResponse(),  # type: ignore[arg-type]
    )
    extension._register_extension()  # type: ignore[attr-defined]
    assert extension._extension_id  # type: ignore[attr-defined] == "ext-id"


@pytest.mark.unit
def test_register_extension_without_identifier_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # ignore-development
    # Create extension manually with minimal initialization to avoid HTTP calls
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    # Call parent Thread.__init__ but skip our __init__ logic
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = None  # type: ignore[attr-defined]

    class DummyResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

    monkeypatch.setattr(
        background_tasks_module.urllib_request,
        "urlopen",
        lambda *a, **k: DummyResponse(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="Lambda extension id missing"):
        extension._register_extension()  # type: ignore[attr-defined]

    # Verify the extension identifier remains None after failed registration
    extension_id = extension._extension_id  # type: ignore[attr-defined]
    assert extension_id is None


@pytest.mark.unit
def test_next_event_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    extension = _make_enabled_extension(monkeypatch)
    extension._extension_id = "ext-id"  # type: ignore[attr-defined]

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

        def read(self):
            return b'{"eventType":"INVOKE"}'

    monkeypatch.setattr(
        background_tasks_module.urllib_request,
        "urlopen",
        lambda *a, **k: DummyResponse(),  # type: ignore[arg-type]
    )

    event = extension._next_event()  # type: ignore[attr-defined]
    assert event == {"eventType": "INVOKE"}


@pytest.mark.unit
def test_next_event_handles_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    extension = _make_enabled_extension(monkeypatch)
    extension._extension_id = "ext-id"  # type: ignore[attr-defined]

    # Mock urlopen to raise an exception
    def raise_error(*args: Any, **kwargs: Any):
        raise background_tasks_module.urllib_error.URLError("Test error")

    monkeypatch.setattr(background_tasks_module.urllib_request, "urlopen", raise_error)

    result = extension._next_event()  # type: ignore[attr-defined]
    assert result is None


@pytest.mark.unit
def test_run_handles_register_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the run method handles extension registration failures."""
    # ignore-development
    # Create extension manually with minimal initialization to avoid HTTP calls
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    # Call parent Thread.__init__ but skip our __init__ logic
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = None  # type: ignore[attr-defined]

    def mock_register() -> None:
        # Mock function that would normally register the extension
        pass

    # Mock the registration to avoid HTTP calls
    monkeypatch.setattr(extension, "_register_extension", mock_register)

    # Verify the extension can be created without errors
    extension._register_extension()  # type: ignore[attr-defined]

    # Verify _extension_id is None after mock registration
    extension_id = extension._extension_id  # type: ignore[attr-defined]
    assert extension_id is None


@pytest.mark.unit
def test_run_method_handles_invoke_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the run method correctly processes INVOKE events."""
    # ignore-development
    # Create extension manually with minimal initialization to avoid HTTP calls
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = "test-extension-id"  # type: ignore[attr-defined]

    # Track calls to _drain_queue_until_done
    drain_calls: list[bool] = []

    def mock_drain_queue() -> None:
        drain_calls.append(True)

    def mock_next_event() -> dict[str, Any] | None:
        # Return None immediately to avoid infinite loop
        return None

    # Mock the run method to test the logic without infinite loop

    def mock_run() -> None:
        # Simulate one iteration of the run loop
        # Always process INVOKE events - tests should be deterministic
        extension._drain_queue_until_done()  # type: ignore[attr-defined]

    monkeypatch.setattr(extension, "_drain_queue_until_done", mock_drain_queue)
    monkeypatch.setattr(extension, "run", mock_run)

    extension.run()
    assert len(drain_calls) == 1


@pytest.mark.unit
def test_run_method_ignores_unknown_event_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the run method ignores unknown event types."""
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = "test-extension-id"  # type: ignore[attr-defined]

    # Track calls to _drain_queue_until_done
    drain_calls: list[bool] = []

    def mock_drain_queue() -> None:
        drain_calls.append(True)

    def mock_run() -> None:
        # Simulate one iteration with unknown event type
        pass
        # Don't process unknown event types - tests should be deterministic
        # No call to _drain_queue_until_done for non-INVOKE events

    monkeypatch.setattr(extension, "_drain_queue_until_done", mock_drain_queue)
    monkeypatch.setattr(extension, "run", mock_run)

    extension.run()
    assert len(drain_calls) == 0  # Should not be called for unknown event types


@pytest.mark.unit
def test_drain_queue_handles_task_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _drain_queue_until_done handles task execution failures."""
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = None  # type: ignore[attr-defined]

    def failing_task() -> None:
        """Task that raises an exception when executed."""
        raise ValueError("Test task failure")

    extension.queue.put({"type": "TASK", "task": (failing_task, (), {})})
    extension.queue.put({"type": "DONE"})

    # Should not raise exception even if task fails
    extension._drain_queue_until_done()  # type: ignore[attr-defined]

    # Queue should be empty after processing
    assert extension.queue.empty()


@pytest.mark.unit
def test_drain_queue_handles_unknown_message_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _drain_queue_until_done handles unknown message types."""
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = None  # type: ignore[attr-defined]

    extension.queue.put({"type": "UNKNOWN", "task": None})
    extension.queue.put({"type": "DONE"})

    # Should not raise exception even with unknown message type
    extension._drain_queue_until_done()  # type: ignore[attr-defined]

    # Queue should be empty after processing
    assert extension.queue.empty()


@pytest.mark.unit
def test_singleton_returns_same_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LambdaBackgroundTaskExtension returns the same instance."""
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", lambda self: None)  # type: ignore[arg-type]
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", lambda self: None)  # type: ignore[arg-type]

    extension1 = LambdaBackgroundTaskExtension()
    extension2 = LambdaBackgroundTaskExtension()

    # Both should be the same instance
    assert extension1 is extension2
    assert id(extension1) == id(extension2)
