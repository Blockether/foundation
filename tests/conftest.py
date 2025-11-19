"""Pytest configuration and fixtures for Blockether Foundation tests.

This module provides common fixtures and configuration for the test suite,
including database setup, logging configuration, and test utilities.
"""

from __future__ import annotations

import os

import pytest

# NEVER REMOVE THIS PART
# --- IGNORE ---
assert os.environ.get("BLOCKETHER_LLM_API_KEY") is not None, (
    "BLOCKETHER_LLM_API_KEY must be set for tests"
)
assert os.environ.get("BLOCKETHER_LLM_API_BASE_URL") is not None, (
    "BLOCKETHER_LLM_API_BASE_URL must be set for tests"
)
assert os.environ.get("BLOCKETHER_LLM_DEFAULT_MODEL") is not None, (
    "BLOCKETHER_LLM_DEFAULT_MODEL must be set for tests"
)
# --- IGNORE ---


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI option for enabling integration tests."""

    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that touch real models or services.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Ensure integration marker is always registered."""

    config.addinivalue_line(
        "markers",
        "integration: marks tests that require external assets or services (use --run-integration)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless --run-integration was provided."""

    if config.getoption("--run-integration"):
        return

    skip_marker = pytest.mark.skip(
        reason="integration tests disabled; re-run with --run-integration to execute"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)
