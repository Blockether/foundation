"""Tests for Telegram interface routing configuration."""

from typing import cast

import pytest
from agno.agent import Agent
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blockether_foundation.os.interfaces.telegram import telegram as telegram_module
from blockether_foundation.os.interfaces.telegram.errors import (
    BotValidationError,
    BotValidationErrorDetails,
)
from blockether_foundation.os.interfaces.telegram.models import BotConfig
from blockether_foundation.os.interfaces.telegram.telegram import Telegram
from blockether_foundation.result import Result

# HTTP status code constants
HTTP_OK = 200


@pytest.mark.unit
def test_get_router_exposes_bot_routes() -> None:
    bot_config = BotConfig(name="blockether-bot", token="A" * 32)
    telegram_interface = Telegram(executor=cast(Agent, None), bot_configs=[bot_config])

    router = telegram_interface.get_router()
    app = FastAPI()
    app.include_router(router)

    client = TestClient(app)
    response = client.get("/telegram/blockether-bot/health")

    assert response.status_code == HTTP_OK
    body = response.json()
    assert body["status"] == "healthy"
    assert body["timestamp"]


@pytest.mark.unit
def test_configure_executor_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        def __init__(self, name: str = "test_agent") -> None:
            self.name = name

    class FakeTeam:
        def __init__(self, name: str = "test_team") -> None:
            self.name = name

    class FakeWorkflow:
        def __init__(self, name: str = "test_workflow") -> None:
            self.name = name

    monkeypatch.setattr(telegram_module, "Agent", FakeAgent)
    monkeypatch.setattr(telegram_module, "Team", FakeTeam)
    monkeypatch.setattr(telegram_module, "Workflow", FakeWorkflow)

    config = BotConfig(name="bot", token="A" * 32)

    # Test Agent configuration
    fake_agent = FakeAgent("agent_test")
    interface_agent = telegram_module.Telegram(
        executor=cast(Agent, fake_agent), bot_configs=[config]
    )
    assert interface_agent.agent is not None
    assert interface_agent.agent == fake_agent
    assert interface_agent.agent.name == "agent_test"

    # Test Team configuration
    fake_team = FakeTeam("team_test")
    interface_team = telegram_module.Telegram(
        executor=cast(telegram_module.Team, fake_team), bot_configs=[config]
    )
    assert interface_team.team is not None
    assert interface_team.team == fake_team
    assert interface_team.team.name == "team_test"

    # Test Workflow configuration
    fake_workflow = FakeWorkflow("workflow_test")
    interface_workflow = telegram_module.Telegram(
        executor=cast(telegram_module.Workflow, fake_workflow), bot_configs=[config]
    )
    assert interface_workflow.workflow is not None
    assert interface_workflow.workflow == fake_workflow
    assert interface_workflow.workflow.name == "workflow_test"


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
    # This function intentionally raises an error to simulate a disabled extension
    # The error is expected and handled gracefully by the Telegram initialization
    def raise_extension_error() -> None:
        # Extension disabled - this error is expected
        raise RuntimeError("Extension disabled") from None  # pragma: no cover

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

    def fake_validate(bot_configs: list[BotConfig]) -> Result[list[BotConfig], BotValidationError]:
        return Result.Err(error)  # type: ignore[return-value]

    monkeypatch.setattr(telegram_module, "validate_and_normalize_bot_configs", fake_validate)

    # Verify that the specific BotValidationError is raised with expected properties
    with pytest.raises(BotValidationError) as exc_info:
        telegram_module.Telegram(
            executor=cast(Agent, None), bot_configs=[BotConfig(name="bot", token="A" * 32)]
        )

    # Assert the exception details
    captured_error = exc_info.value
    assert captured_error.bot_name == "bot"
    # Access validation errors through the details object
    assert captured_error.details is not None
    details = cast(BotValidationErrorDetails, captured_error.details)
    assert details.validation_errors == ["bad"]
    # Access provided config through the details object
    assert details.provided_config == {"name": "bot"}
    error_str = str(captured_error)
    assert "Bot configuration validation failed for 'bot'" in error_str
