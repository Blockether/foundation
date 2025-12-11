"""Tests for Telegram interface routing configuration."""

from typing import cast

import pytest
from agno.agent import Agent
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blockether_foundation.os.interfaces.telegram import telegram as telegram_module
from blockether_foundation.os.interfaces.telegram.errors import BotValidationError
from blockether_foundation.os.interfaces.telegram.models import BotConfig
from blockether_foundation.os.interfaces.telegram.telegram import Telegram
from blockether_foundation.result import Result


@pytest.mark.unit
def test_get_router_exposes_bot_routes() -> None:
    bot_config = BotConfig(name="blockether-bot", token="A" * 32)
    telegram_interface = Telegram(executor=cast(Agent, None), bot_configs=[bot_config])

    router = telegram_interface.get_router()
    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)
    response = client.get("/telegram/blockether-bot/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["timestamp"]


@pytest.mark.unit
def test_configure_executor_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        pass

    class FakeTeam:
        pass

    class FakeWorkflow:
        pass

    monkeypatch.setattr(telegram_module, "Agent", FakeAgent)
    monkeypatch.setattr(telegram_module, "Team", FakeTeam)
    monkeypatch.setattr(telegram_module, "Workflow", FakeWorkflow)

    config = BotConfig(name="bot", token="A" * 32)
    interface = telegram_module.Telegram(executor=cast(Agent, None), bot_configs=[config])

    interface._configure_executor(cast(Agent, FakeAgent()))
    assert interface.agent is not None

    interface._configure_executor(cast(telegram_module.Team, FakeTeam()))
    assert interface.team is not None

    interface._configure_executor(cast(telegram_module.Workflow, FakeWorkflow()))
    assert interface.workflow is not None


@pytest.mark.unit
def test_lambda_extension_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyExtension:
        def __init__(self) -> None:
            self.is_enabled = True

    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "runtime")
    monkeypatch.setattr(telegram_module, "LambdaBackgroundTaskExtension", lambda: DummyExtension())

    interface = telegram_module.Telegram(
        executor=cast(Agent, None), bot_configs=[BotConfig(name="bot", token="A" * 32)]
    )
    assert isinstance(interface.background_task_scheduler, DummyExtension)


@pytest.mark.unit
def test_lambda_extension_not_created_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_extension_error():  # noqa: ANN001
        raise RuntimeError("Extension disabled")

    monkeypatch.setenv("AWS_LAMBDA_RUNTIME_API", "runtime")
    monkeypatch.setattr(telegram_module, "LambdaBackgroundTaskExtension", raise_extension_error)

    interface = telegram_module.Telegram(
        executor=cast(Agent, None), bot_configs=[BotConfig(name="bot", token="A" * 32)]
    )
    assert interface.background_task_scheduler is None


@pytest.mark.unit
def test_telegram_initialization_raises_on_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = BotValidationError("bot", ["bad"], {"name": "bot"})

    def fake_validate(bot_configs):  # noqa: ANN001
        return Result.Err(error)

    monkeypatch.setattr(telegram_module, "validate_and_normalize_bot_configs", fake_validate)

    with pytest.raises(BotValidationError):
        telegram_module.Telegram(
            executor=cast(Agent, None), bot_configs=[BotConfig(name="bot", token="A" * 32)]
        )
