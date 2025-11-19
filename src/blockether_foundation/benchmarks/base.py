"""Base classes for the benchmark system.

This module defines the foundational interfaces and configuration structures
that enable pluggable, extensible benchmarking across diverse tasks and data sources.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class MetricConfig:
    """Configuration for evaluation metrics."""

    name: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("Metric weight must be non-negative")


class TaskMetadata(BaseModel):
    """Metadata describing the benchmark task."""

    description: str = Field(description="Human-readable description of the task")
    domain: str = Field(description="Domain category of the task")
    task_type: str = Field(description="Type of task (e.g., qa, classification, etc.)")
    evaluation_type: str = Field(
        description="Type of evaluation (e.g., text_match, numeric, etc.)"
    )

    class Config:
        extra = "allow"  # Allow additional fields


class BenchmarkConfig(BaseModel):
    """Configuration structure for benchmark tasks."""

    task: str = Field(description="Unique identifier for the benchmark task")
    version: str = Field(description="Version of the benchmark configuration")

    # Data configuration
    data: dict[str, Any] = Field(
        description="Data source configuration (source, dataset_path, split, limit, etc.)"
    )

    # Preprocessing configuration
    preprocessing: dict[str, Any] = Field(
        description="Data preprocessing settings (templates, field mappings, etc.)"
    )

    # Metrics configuration
    metrics: list[MetricConfig] = Field(
        default_factory=list,
        description="List of metrics to compute with their weights",
    )

    # Task metadata
    metadata: TaskMetadata = Field(description="Task metadata and classification")

    class Config:
        arbitrary_types_allowed = True


class DataLoader(abc.ABC):
    """Abstract interface for data sources."""

    @abc.abstractmethod
    def load(self, config: BenchmarkConfig) -> list[Sample]:
        """Load data from the source according to configuration."""
        pass

    @abc.abstractmethod
    def validate(self, config: BenchmarkConfig) -> bool:
        """Validate that the data source is accessible and properly configured."""
        pass


class BenchmarkEnvironment(abc.ABC):
    """Base class for task-specific evaluation logic."""

    @abc.abstractmethod
    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        ground_truth: str,
        metrics: list[MetricConfig],
    ) -> dict[str, float]:
        """Evaluate a prediction against ground truth using specified metrics."""
        pass

    @abc.abstractmethod
    def format_input(self, sample: Sample) -> str:
        """Format a sample for model input."""
        pass

    @abc.abstractmethod
    def extract_ground_truth(self, sample: Sample) -> str:
        """Extract the ground truth answer from a sample."""
        pass


class Sample(BaseModel):
    """Unified data representation for benchmark samples."""

    id: str = Field(description="Unique identifier for the sample")
    question: str = Field(description="The question or task description")
    context: str | None = Field(
        default=None, description="Optional context information"
    )
    ground_truth: str = Field(description="The expected answer or output")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional sample metadata"
    )

    # Source information
    source_dataset: str | None = Field(
        default=None, description="Name of the source dataset"
    )
    source_split: str | None = Field(
        default=None, description="Split within the source dataset"
    )

    # Evaluation status
    split_type: str | None = Field(
        default=None, description="Train/test split assignment"
    )

    class Config:
        arbitrary_types_allowed = True


# Cache management utilities
def get_cache_dir() -> Path:
    """Get the default cache directory for benchmark data."""
    cache_dir = Path.home() / ".cache" / "ace_benchmarks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_cache() -> None:
    """Clear all cached benchmark data."""
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)


class ConfigLoader:
    """Utility class for loading benchmark configurations."""

    @staticmethod
    def load_from_file(config_path: Path) -> BenchmarkConfig:
        """Load configuration from YAML or TOML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        suffix = config_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            import yaml

            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        elif suffix == ".toml":
            import tomllib

            with open(config_path, "rb") as f:
                config_data = tomllib.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {suffix}")

        # Convert metric configs to MetricConfig objects
        if "metrics" in config_data:
            config_data["metrics"] = [
                MetricConfig(**metric) if isinstance(metric, dict) else metric
                for metric in config_data["metrics"]
            ]

        # Convert metadata to TaskMetadata object
        if "metadata" in config_data:
            config_data["metadata"] = TaskMetadata(**config_data["metadata"])

        # Convert version to string if it's a number
        if "version" in config_data and not isinstance(config_data["version"], str):
            config_data["version"] = str(config_data["version"])

        return BenchmarkConfig(**config_data)

    @staticmethod
    def discover_configs(config_dir: Path) -> list[Path]:
        """Discover all configuration files in the given directory."""
        if not config_dir.exists():
            return []

        config_files = []
        for config_path in config_dir.rglob("*"):
            if config_path.is_file() and config_path.suffix.lower() in [
                ".yaml",
                ".yml",
                ".toml",
            ]:
                config_files.append(config_path)

        return sorted(config_files)
