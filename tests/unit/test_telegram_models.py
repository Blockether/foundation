"""Tests for Telegram model utilities."""

from blockether_foundation.os.interfaces.telegram.models import BotConfig


def test_bot_config_webhook_url() -> None:
    config = BotConfig(name="alpha-bot", token="token")
    assert config.webhook_url == "/telegram/alpha-bot/webhook"
