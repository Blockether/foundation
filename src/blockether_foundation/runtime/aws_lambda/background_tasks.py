"""AWS Lambda background task extension utilities."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from queue import Queue
from threading import Thread
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from agno.utils.log import logger


class LambdaBackgroundTaskExtension(Thread):
    """Lightweight Lambda extension that drains queued tasks after each invoke."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.extension_name = "lambda-background-tasks"
        self.runtime_api = os.environ.get("AWS_LAMBDA_RUNTIME_API")
        if not self.runtime_api:
            raise RuntimeError(
                "AWS_LAMBDA_RUNTIME_API is not set. LambdaBackgroundTaskExtension must run inside AWS Lambda."
            )

        self.queue: Queue[dict[str, Any]] = Queue()
        self._extension_id: str | None = None
        logger.info("Starting Lambda background task extension thread")
        self._register_extension()

        self.start()

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Add a callable to the processing queue for deferred execution."""

        self.queue.put({"type": "TASK", "task": (func, args, kwargs)})

    def done(self) -> None:
        """Signal completion of the current invoke cycle."""

        self.queue.put({"type": "DONE"})

    def run(self) -> None:  # type: ignore[override]
        while True:
            event = self._next_event()
            if not event:
                continue

            if event.get("eventType") == "INVOKE":
                self._drain_queue_until_done()

    def _register_extension(self) -> None:
        if not self._extension_id:
            payload = json.dumps({"events": ["INVOKE"]}).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Lambda-Extension-Name": self.extension_name,
            }
            req = urllib_request.Request(
                f"http://{self.runtime_api}/2020-01-01/extension/register",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=5) as resp:  # type: ignore[arg-type]
                extension_id = resp.headers.get("Lambda-Extension-Identifier")
                if not extension_id:
                    raise RuntimeError("Lambda extension id missing from register response")
                self._extension_id = extension_id

    def _next_event(self) -> dict[str, Any] | None:
        if not self._extension_id:
            logger.debug("No extension ID available, skipping event polling")
            return None

        headers = {"Lambda-Extension-Identifier": self._extension_id}
        req = urllib_request.Request(
            f"http://{self.runtime_api}/2020-01-01/extension/event/next",
            headers=headers,  # type: ignore
            method="GET",
        )
        try:
            with urllib_request.urlopen(req, timeout=None) as resp:  # type: ignore[arg-type]
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except urllib_error.URLError as exc:  # pragma: no cover - network failure
            logger.warning(f"Lambda extension event loop error: {exc}")
            return None

    def _drain_queue_until_done(self) -> None:
        while True:
            message = self.queue.get()
            message_type = message.get("type")

            if message_type == "TASK":
                func, args, kwargs = message["task"]
                try:
                    func(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - task failure logging
                    logger.error(f"Background task failed: {exc}")
            elif message_type == "DONE":
                break
            else:
                logger.warning(
                    f"Unknown message type in Lambda background task queue: {message_type}"
                )
