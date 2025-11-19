"""Task-specific implementations for benchmark evaluations.

This module contains task-specific environments and evaluation logic for
different benchmark tasks.
"""

from typing import Optional

from .simple_qa import SimpleQAEnvironment


def get_task_environment(task_name: str):
    """Get the environment implementation for a specific task."""
    environments = {
        "simple_qa": SimpleQAEnvironment,
    }
    return environments.get(task_name)


__all__ = ["get_task_environment", "SimpleQAEnvironment"]
