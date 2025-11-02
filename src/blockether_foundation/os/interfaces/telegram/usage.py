"""Clean usage examples for Telegram interface with no backward compatibility."""

from fastapi import FastAPI
from agno.agent import Agent

from blockether_foundation.os.interfaces.telegram import BotConfig, Telegram


# Create FastAPI application
app = FastAPI()


# Example 1: Single Bot Configuration
def single_bot_example() -> Telegram:
    """Example with a single bot configuration."""

    # Create bot configuration
    bot_config = BotConfig(
        name="my-assistant",
        token="YOUR_BOT_TOKEN_HERE",
        webhook_secret="your-webhook-secret",
        allowlist_user_ids=["123456789"],  # Only specific users (as strings)
        max_concurrent_updates=5,
        executor_timeout=30
    )

    # Create AI agent
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful AI assistant."
    )

    # Create and mount the Telegram interface
    telegram_interface = Telegram(
        executor=agent,
        bot_configs=[bot_config]  # Always a list, even for single bot
    )

    app.include_router(telegram_interface.get_router())
    return telegram_interface


# Example 2: Multiple Bots Configuration
def multiple_bots_example() -> Telegram:
    """Example with multiple bot configurations."""

    # Create multiple bot configurations
    bot_configs = [
        BotConfig(
            name="general-assistant",
            token="GENERAL_BOT_TOKEN",
            webhook_secret="general-secret",
            allowlist_user_ids=["123456789", "987654321"],  # Team members (as strings)
            max_concurrent_updates=10,
            executor_timeout=45
        ),
        BotConfig(
            name="support-bot",
            token="SUPPORT_BOT_TOKEN",
            webhook_secret="support-secret",
            # No allowlist = public bot
            denylist_user_ids=["111111111"],  # Block problematic user (as string)
            max_concurrent_updates=20,
            executor_timeout=60
        ),
        BotConfig(
            name="admin-bot",
            token="ADMIN_BOT_TOKEN",
            webhook_secret="admin-secret",
            allowlist_user_ids=["555555555"],  # Only admin (as string)
            max_concurrent_updates=3,
            executor_timeout=120,
            enable_debug_mode=True
        )
    ]

    # Create AI agent
    agent = Agent(
        name="Multi-Purpose Assistant",
        instructions="You are a helpful AI assistant."
    )

    # Create interface with multiple bots
    telegram_interface = Telegram(
        executor=agent,
        bot_configs=bot_configs  # List of BotConfig objects
    )

    app.include_router(telegram_interface.get_router())
    return telegram_interface


# Example 3: Different Executors per Bot Group
def multiple_executors_example() -> list[Telegram]:
    """Example with different executors for different bot groups."""

    # Create different agents for different purposes
    general_agent = Agent(name="General", instructions="General assistance")
    support_agent = Agent(name="Support", instructions="Customer support")
    admin_agent = Agent(name="Admin", instructions="Administrative tasks")

    # Create separate interfaces for different bot groups
    interfaces = []

    # General and support bots (same executor)
    general_support_bots = [
        BotConfig(
            name="general-assistant",
            token="GENERAL_BOT_TOKEN",
            webhook_secret="general-secret",
            max_concurrent_updates=10
        ),
        BotConfig(
            name="support-bot",
            token="SUPPORT_BOT_TOKEN",
            webhook_secret="support-secret",
            max_concurrent_updates=15
        )
    ]

    general_support_interface = Telegram(
        executor=general_agent,
        bot_configs=general_support_bots
    )
    interfaces.append(general_support_interface)

    # Admin bot (different executor)
    admin_bots = [
        BotConfig(
            name="admin-bot",
            token="ADMIN_BOT_TOKEN",
            webhook_secret="admin-secret",
            allowlist_user_ids=["555555555"],  # Only admin (as string)
            max_concurrent_updates=3
        )
    ]

    admin_interface = Telegram(
        executor=admin_agent,
        bot_configs=admin_bots
    )
    interfaces.append(admin_interface)

    # Mount all interfaces
    for interface in interfaces:
        app.include_router(interface.get_router())

    return interfaces


# Webhook Setup Examples
def webhook_setup_example() -> list[BotConfig]:
    """Example of how to set up webhooks for multiple bots."""

    bot_configs = [
        BotConfig(name="general-assistant", token="GENERAL_BOT_TOKEN"),
        BotConfig(name="support-bot", token="SUPPORT_BOT_TOKEN"),
        BotConfig(name="admin-bot", token="ADMIN_BOT_TOKEN")
    ]

    print("Webhook setup commands:")
    print("=" * 50)

    for bot_config in bot_configs:
        webhook_url = f"https://your-domain.com{bot_config.webhook_url}"
        curl_command = f'''curl -X POST "https://api.telegram.org/bot{bot_config.token}/setWebhook" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "url": "{webhook_url}",
       "secret_token": "{bot_config.webhook_secret}"
     }}'\n'''
        print(f"Bot: {bot_config.name}")
        print(curl_command)

    return bot_configs


# Usage
if __name__ == "__main__":
    import uvicorn

    # Create interfaces
    single_bot = single_bot_example()
    multiple_bots = multiple_bots_example()
    multiple_executors = multiple_executors_example()

    # Print webhook setup information
    webhook_setup_example()

    # Run the application
    print("\nStarting FastAPI application with Telegram interface...")
    print("Available endpoints:")
    print("- Single bot: /telegram/my-assistant/webhook")
    print("- General assistant: /telegram/general-assistant/webhook")
    print("- Support bot: /telegram/support-bot/webhook")
    print("- Admin bot: /telegram/admin-bot/webhook")
    print("- Interface status: /telegram/status")
    print("- Individual bot status: /telegram/{bot-name}/status")
    print("- Health check: /telegram/{bot-name}/health")

    uvicorn.run(app, host="0.0.0.0", port=8000)