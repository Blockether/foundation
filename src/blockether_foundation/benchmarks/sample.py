"""Sample data representation for benchmark evaluation.

This module defines the unified Sample class that provides a consistent interface
for different types of benchmark tasks and data sources.
"""

from __future__ import annotations

from typing import Any

from .base import Sample as BaseSample


class Sample(BaseSample):
    """Enhanced Sample class with additional utility methods."""

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to dictionary representation."""
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "split_type": self.split_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        """Create Sample from dictionary representation."""
        return cls(**data)

    def format_for_display(self) -> str:
        """Format sample for human-readable display."""
        result = f"Question: {self.question}\n"
        if self.context:
            result += f"Context: {self.context}\n"
        result += f"Ground Truth: {self.ground_truth}\n"
        if self.source_dataset:
            result += f"Source: {self.source_dataset}/{self.source_split}\n"
        return result

    def clone(self, **updates: Any) -> Sample:
        """Create a copy of the sample with optional updates."""
        data = self.to_dict()
        data.update(updates)
        return self.from_dict(data)
