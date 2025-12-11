"""Integration test configuration and fixtures."""

from __future__ import annotations

import scenario

scenario.configure(default_model="openai/gpt-4o", verbose=True, debug=True, headless=True)
