"""Minimal data models for Telegram interface - only what's needed."""

from typing import Any, Literal

from pydantic import BaseModel


class BotConfig(BaseModel):
    """Configuration for a Telegram bot."""
    name: str
    token: str
    webhook_secret: str | None = None
    max_concurrent_updates: int = 10
    executor_timeout: int = 30
    allowlist_user_ids: list[str] = []
    denylist_user_ids: list[str] = []
    enable_debug_mode: bool = False

    @property
    def webhook_url(self) -> str:
        """Get the webhook URL for this bot configuration."""
        return f"/telegram/{self.name}/webhook"


class Update(BaseModel):
    """Simplified Telegram update model for basic validation."""
    update_id: int
    message: dict[str, Any] | None = None
    callback_query: dict[str, Any] | None = None


class WebhookResponse(BaseModel):
    """Response for webhook processing confirmation."""
    status: Literal["ok", "error"]
    update_id: int
    processed_at: str


class StatusResponse(BaseModel):
    """Simple status response for a bot."""
    status: Literal["active", "inactive", "error"]
    bot_name: str
    timestamp: str


class HealthResponse(BaseModel):
    """Simple health check response."""
    status: Literal["healthy", "unhealthy"]
    timestamp: str