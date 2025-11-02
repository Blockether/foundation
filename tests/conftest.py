"""Pytest configuration and fixtures for Blockether Foundation tests.

This module provides common fixtures and configuration for the test suite,
including database setup, logging configuration, and test utilities.
"""

from __future__ import annotations

import os

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
