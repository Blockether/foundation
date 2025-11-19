"""HuggingFace dataset loader with streaming support.

This module provides a data loader for HuggingFace datasets with support for
streaming, caching, and efficient large dataset handling.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "HuggingFace datasets library is required. Install with: pip install datasets"
    )

from ..base import BenchmarkConfig, DataLoader, Sample, get_cache_dir

logger = logging.getLogger(__name__)


class HuggingFaceLoader(DataLoader):
    """Data loader for HuggingFace datasets with streaming and caching support."""

    def __init__(self, cache_dir: str | None = None):
        """Initialize the HuggingFace loader.

        Args:
            cache_dir: Custom cache directory. If None, uses default location.
        """
        self.cache_dir = cache_dir or str(get_cache_dir() / "huggingface")

    def load(self, config: BenchmarkConfig) -> list[Sample]:
        """Load data from HuggingFace dataset according to configuration."""
        dataset_path = config.data.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path is required for HuggingFace data source")

        split = config.data.get("split", "train")
        limit = config.data.get("limit")
        streaming = config.data.get("streaming", True)

        logger.info(
            f"Loading dataset: {dataset_path}, split: {split}, "
            f"limit: {limit}, streaming: {streaming}"
        )

        try:
            # Load dataset with streaming support for large datasets
            dataset = load_dataset(
                dataset_path,
                split=split,
                streaming=streaming,
                cache_dir=self.cache_dir,
            )

            # Convert to list if streaming and limit is specified
            if streaming:
                if limit:
                    dataset = list(dataset.take(limit))  # type: ignore
                else:
                    dataset = list(dataset)  # Convert entire streaming dataset
            else:
                # Non-streaming: slice if limit is specified
                if limit:
                    dataset = dataset.select(range(min(limit, len(dataset))))  # type: ignore
                dataset = list(dataset)

            logger.info(f"Loaded {len(dataset)} samples from {dataset_path}")

            return self._convert_to_samples(dataset, config)

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            raise

    def validate(self, config: BenchmarkConfig) -> bool:
        """Validate that the dataset is accessible and properly configured."""
        dataset_path = config.data.get("dataset_path")
        if not dataset_path:
            logger.error("dataset_path is required for HuggingFace data source")
            return False

        try:
            # Try to load dataset info without downloading
            dataset_info = load_dataset(
                dataset_path,
                split=config.data.get("split", "train"),
                streaming=True,
                cache_dir=self.cache_dir,
            )

            # Try to get first sample to validate structure
            first_sample = next(iter(dataset_info))
            logger.debug(f"Dataset validation successful for {dataset_path}")
            return True

        except Exception as e:
            logger.error(f"Dataset validation failed for {dataset_path}: {e}")
            return False

    def _convert_to_samples(
        self, dataset: list[dict[str, Any]], config: BenchmarkConfig
    ) -> list[Sample]:
        """Convert raw dataset entries to Sample objects."""
        samples = []
        preprocessing = config.preprocessing

        question_field = preprocessing.get("input_field", "question")
        context_field = preprocessing.get("context_field")
        ground_truth_field = preprocessing.get("ground_truth_field", "answer")
        question_template = preprocessing.get("question_template")

        for idx, data in enumerate(dataset):
            try:
                # Extract fields using configured field names
                question = self._extract_field(data, question_field)
                ground_truth = self._extract_field(data, ground_truth_field)
                context = (
                    self._extract_field(data, context_field) if context_field else None
                )

                # Apply question template if provided
                if question_template:
                    if context:
                        question = question_template.format(
                            context=context, question=question
                        )
                    else:
                        question = question_template.format(question=question)

                # Create sample ID
                sample_id = f"{config.task}_{idx:04d}"

                sample = Sample(
                    id=sample_id,
                    question=question,
                    context=context,
                    ground_truth=ground_truth,
                    source_dataset=config.data.get("dataset_path"),
                    source_split=config.data.get("split", "train"),
                    metadata={"raw_data": data, "index": idx},
                )

                samples.append(sample)

            except Exception as e:
                logger.warning(f"Failed to convert sample {idx}: {e}")
                continue

        logger.debug(f"Successfully converted {len(samples)} samples")
        return samples

    def _extract_field(self, data: dict[str, Any], field_path: str) -> str:
        """Extract a field from data, supporting nested paths."""
        if not field_path:
            return ""

        # Handle nested field paths (e.g., "answer.text")
        keys = field_path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif isinstance(value, list) and key.isdigit():
                idx = int(key)
                if 0 <= idx < len(value):
                    value = value[idx]
                else:
                    return ""
            else:
                return ""

        # Convert to string, handling various data types
        if isinstance(value, list):
            # For SQuAD-style answers, take the first answer instead of joining all
            return str(value[0]) if value else ""
        elif isinstance(value, dict):
            return str(value)
        else:
            return str(value)

    def get_dataset_info(self, dataset_path: str) -> dict[str, Any]:
        """Get information about a dataset without loading it."""
        try:
            dataset_info = load_dataset(dataset_path, split="train", streaming=True)
            first_sample = next(iter(dataset_info))

            return {
                "dataset_path": dataset_path,
                "sample_keys": list(first_sample.keys()),  # type: ignore
                "sample_types": {k: type(v).__name__ for k, v in first_sample.items()},  # type: ignore
            }

        except Exception as e:
            logger.error(f"Failed to get dataset info for {dataset_path}: {e}")
            return {}
