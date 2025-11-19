"""Benchmark system for evaluating ACE framework performance.

This package provides a comprehensive benchmark system that enables standardized
evaluation across diverse tasks through pluggable components, inspired by
lm-evaluation-harness.

Core Components:
- BenchmarkConfig: Configuration structure from YAML/TOML files
- DataLoader: Abstract interface for data sources
- BenchmarkEnvironment: Base class for task-specific evaluation logic
- BenchmarkTaskManager: Central coordinator for discovery and orchestration
"""

from .base import (
    BenchmarkConfig,
    BenchmarkEnvironment,
    DataLoader,
    MetricConfig,
    TaskMetadata,
)
from .manager import BenchmarkTaskManager
from .sample import Sample

__all__ = [
    "BenchmarkConfig",
    "BenchmarkEnvironment",
    "DataLoader",
    "MetricConfig",
    "TaskMetadata",
    "BenchmarkTaskManager",
    "Sample",
]
