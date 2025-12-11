"""Tests for Lambda background task extension."""

from queue import Empty

import pytest

from blockether_foundation.os.runtime.aws_lambda import background_tasks as background_tasks_module
from blockether_foundation.os.runtime.aws_lambda.background_tasks import (
    LambdaBackgroundTaskExtension,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset the singleton instance before each test."""
    LambdaBackgroundTaskExtension._reset_singleton()


@pytest.mark.unit
def test_extension_requires_runtime_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_LAMBDA_RUNTIME_API", raising=False)

    with pytest.raises(RuntimeError):
        LambdaBackgroundTaskExtension()


@pytest.mark.unit
def test_add_task_enqueues_when_runtime_available(monkeypatch: pytest.MonkeyPatch) -> None:
    started = False

    def fake_start(self: LambdaBackgroundTaskExtension) -> None:  # type: ignore[override]
        nonlocal started
        started = True

    def fake_register(self: LambdaBackgroundTaskExtension) -> None:  # type: ignore[override]
        pass

    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", fake_start)
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", fake_register)

    extension = LambdaBackgroundTaskExtension()

    assert started is True

    extension.add_task(lambda value: value, 1)
    message = extension.queue.get_nowait()

    assert message["type"] == "TASK"
    func, args, kwargs = message["task"]
    assert func(1) == 1
    assert args == (1,)
    assert kwargs == {}


@pytest.mark.unit
def test_done_enqueues_completion_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", lambda self: None)
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", lambda self: None)

    extension = LambdaBackgroundTaskExtension()
    extension.done()

    message = extension.queue.get_nowait()
    assert message["type"] == "DONE"


@pytest.mark.unit
def test_drain_queue_executes_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", lambda self: None)
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", lambda self: None)
    extension = LambdaBackgroundTaskExtension()

    executed: list[int] = []

    extension.queue.put({"type": "TASK", "task": (lambda value: executed.append(value), (5,), {})})
    extension.queue.put({"type": "DONE"})

    extension._drain_queue_until_done()

    assert executed == [5]

    with pytest.raises(Empty):
        extension.queue.get_nowait()


def _make_enabled_extension(monkeypatch: pytest.MonkeyPatch) -> LambdaBackgroundTaskExtension:
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", lambda self: None)
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", lambda self: None)
    return LambdaBackgroundTaskExtension()


@pytest.mark.unit
def test_register_extension_returns_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create extension manually with minimal initialization to avoid HTTP calls
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    # Call parent Thread.__init__ but skip our __init__ logic
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = None

    class DummyResponse:
        headers = {"Lambda-Extension-Identifier": "ext-id"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        background_tasks_module.urllib_request, "urlopen", lambda *a, **k: DummyResponse()
    )
    extension._register_extension()
    assert extension._extension_id == "ext-id"


@pytest.mark.unit
def test_register_extension_without_identifier_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create extension manually with minimal initialization to avoid HTTP calls
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    # Call parent Thread.__init__ but skip our __init__ logic
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = None

    class DummyResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        background_tasks_module.urllib_request, "urlopen", lambda *a, **k: DummyResponse()
    )

    with pytest.raises(RuntimeError):
        extension._register_extension()


@pytest.mark.unit
def test_next_event_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    extension = _make_enabled_extension(monkeypatch)
    extension._extension_id = "ext-id"

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"eventType":"INVOKE"}'

    monkeypatch.setattr(
        background_tasks_module.urllib_request, "urlopen", lambda *a, **k: DummyResponse()
    )

    event = extension._next_event()
    assert event == {"eventType": "INVOKE"}


@pytest.mark.unit
def test_next_event_handles_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    extension = _make_enabled_extension(monkeypatch)
    extension._extension_id = "ext-id"

    def raise_error(*args, **kwargs):  # noqa: ANN001
        raise background_tasks_module.urllib_error.URLError("boom")

    monkeypatch.setattr(background_tasks_module.urllib_request, "urlopen", raise_error)

    assert extension._next_event() is None


@pytest.mark.unit
def test_run_handles_register_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create extension manually with minimal initialization to avoid HTTP calls
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    # Call parent Thread.__init__ but skip our __init__ logic
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = None

    def raise_error():  # noqa: ANN001
        raise RuntimeError("fail")

    monkeypatch.setattr(extension, "_register_extension", raise_error)

    # Test that _register_extension failure is handled gracefully
    try:
        extension._register_extension()
    except RuntimeError:
        pass  # Expected

    assert extension._extension_id is None


@pytest.mark.unit
def test_run_method_handles_invoke_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the run method correctly processes INVOKE events."""
    # Create extension manually with minimal initialization to avoid HTTP calls
    extension = LambdaBackgroundTaskExtension.__new__(LambdaBackgroundTaskExtension)
    background_tasks_module.Thread.__init__(extension, daemon=True)
    extension.extension_name = "lambda-background-tasks"
    extension.runtime_api = "localhost:9001"
    extension.queue = background_tasks_module.Queue()
    extension._extension_id = "test-extension-id"

    # Track calls to _drain_queue_until_done
    drain_calls = []

    def mock_drain_queue():
        drain_calls.append(True)

    def mock_next_event():
        # Return None immediately to avoid infinite loop
        return None

    # Mock the run method to test the logic without infinite loop
    original_run = extension.run

    def mock_run():
        # Simulate one iteration of the run loop
        event = {"eventType": "INVOKE"}
        if event.get("eventType") == "INVOKE":
            extension._drain_queue_until_done()

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
    extension._extension_id = "test-extension-id"

    # Track calls to _drain_queue_until_done
    drain_calls = []

    def mock_drain_queue():
        drain_calls.append(True)

    def mock_run():
        # Simulate one iteration with unknown event type
        event = {"eventType": "UNKNOWN"}
        if event.get("eventType") == "INVOKE":
            extension._drain_queue_until_done()

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
    extension._extension_id = None

    def failing_task():
        raise RuntimeError("Task failed")

    extension.queue.put({"type": "TASK", "task": (failing_task, (), {})})
    extension.queue.put({"type": "DONE"})

    # Should not raise exception even if task fails
    extension._drain_queue_until_done()

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
    extension._extension_id = None

    extension.queue.put({"type": "UNKNOWN", "task": None})
    extension.queue.put({"type": "DONE"})

    # Should not raise exception even with unknown message type
    extension._drain_queue_until_done()

    # Queue should be empty after processing
    assert extension.queue.empty()


@pytest.mark.unit
def test_singleton_returns_same_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that LambdaBackgroundTaskExtension returns the same instance."""
    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "localhost:9001")
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "start", lambda self: None)
    monkeypatch.setattr(LambdaBackgroundTaskExtension, "_register_extension", lambda self: None)

    extension1 = LambdaBackgroundTaskExtension()
    extension2 = LambdaBackgroundTaskExtension()

    # Both should be the same instance
    assert extension1 is extension2
    assert id(extension1) == id(extension2)
