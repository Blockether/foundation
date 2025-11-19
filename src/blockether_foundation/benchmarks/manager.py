"""Benchmark task manager for discovery and orchestration.

This module provides the central coordinator that handles configuration discovery,
benchmark instantiation, and execution orchestration.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BenchmarkConfig, BenchmarkEnvironment, ConfigLoader, DataLoader
from .loaders import get_loader

logger = logging.getLogger(__name__)


class BenchmarkTaskManager:
    """Central coordinator for benchmark task discovery and management."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize the benchmark task manager.

        Args:
            config_dir: Directory containing benchmark task configurations.
                       Defaults to benchmarks/tasks/ relative to this package.
        """
        if config_dir is None:
            # Default to the tasks directory within this package
            config_dir = Path(__file__).parent / "tasks"

        self.config_dir = Path(config_dir)
        self._configs: dict[str, BenchmarkConfig] = {}
        self._environments: dict[str, type[BenchmarkEnvironment]] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Load all benchmark configurations from the config directory."""
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            return

        config_files = ConfigLoader.discover_configs(self.config_dir)
        logger.info(f"Found {len(config_files)} configuration files")

        for config_file in config_files:
            try:
                config = ConfigLoader.load_from_file(config_file)
                self._configs[config.task] = config
                logger.debug(f"Loaded configuration for task: {config.task}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_file}: {e}")

    def register_environment(
        self, task_name: str, environment_class: type[BenchmarkEnvironment]
    ) -> None:
        """Register a custom environment class for a specific task."""
        self._environments[task_name] = environment_class
        logger.debug(f"Registered environment for task: {task_name}")

    def list_tasks(self) -> list[str]:
        """List all available benchmark tasks."""
        return list(self._configs.keys())

    def get_config(self, task_name: str) -> BenchmarkConfig | None:
        """Get configuration for a specific task."""
        return self._configs.get(task_name)

    def get_environment(self, task_name: str) -> BenchmarkEnvironment | None:
        """Get the environment instance for a specific task."""
        config = self.get_config(task_name)
        if not config:
            return None

        # Check for custom environment
        if task_name in self._environments:
            return self._environments[task_name]()

        # Try to import default environment
        try:
            from .tasks import get_task_environment

            env_class = get_task_environment(task_name)
            return env_class() if env_class else None
        except ImportError:
            logger.error(f"No environment found for task: {task_name}")
            return None

    def get_loader(self, config: BenchmarkConfig) -> DataLoader | None:
        """Get the appropriate data loader for a configuration."""
        source = config.data.get("source")
        if not source:
            logger.error(f"No data source specified for task: {config.task}")
            return None

        try:
            return get_loader(source)
        except ValueError as e:
            logger.error(f"Unknown data source '{source}' for task {config.task}: {e}")
            return None

    def validate_task(self, task_name: str) -> bool:
        """Validate that a task is properly configured and ready for execution."""
        config = self.get_config(task_name)
        if not config:
            logger.error(f"Configuration not found for task: {task_name}")
            return False

        # Validate data loader
        loader = self.get_loader(config)
        if not loader:
            logger.error(f"No data loader available for task: {task_name}")
            return False

        if not loader.validate(config):
            logger.error(f"Data validation failed for task: {task_name}")
            return False

        # Validate environment
        environment = self.get_environment(task_name)
        if not environment:
            logger.error(f"No environment available for task: {task_name}")
            return False

        return True

    def load_samples(self, task_name: str) -> list:
        """Load all samples for a given task."""
        config = self.get_config(task_name)
        if not config:
            raise ValueError(f"Configuration not found for task: {task_name}")

        loader = self.get_loader(config)
        if not loader:
            raise ValueError(f"No data loader available for task: {task_name}")

        return loader.load(config)

    def get_task_info(self) -> dict[str, dict]:
        """Get summary information for all available tasks."""
        info = {}
        for task_name, config in self._configs.items():
            info[task_name] = {
                "version": config.version,
                "description": config.metadata.description,
                "domain": config.metadata.domain,
                "task_type": config.metadata.task_type,
                "evaluation_type": config.metadata.evaluation_type,
                "data_source": config.data.get("source"),
                "metrics": [m.name for m in config.metrics],
                "has_environment": task_name in self._environments
                or self._has_default_environment(task_name),
            }
        return info

    def _has_default_environment(self, task_name: str) -> bool:
        """Check if a task has a default environment implementation."""
        try:
            from .tasks import get_task_environment

            env = get_task_environment(task_name)
            return env is not None
        except ImportError:
            return False
