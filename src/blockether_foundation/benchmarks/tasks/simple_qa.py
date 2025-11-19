"""Simple Question Answering environment for SQuAD-based evaluation.

This module implements the evaluation environment for the simple_qa benchmark task,
which uses SQuAD dataset for context-based question answering evaluation.
"""

from __future__ import annotations

import logging

import numpy as np

from ..base import BenchmarkEnvironment, MetricConfig, Sample
from ..metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class SimpleQAEnvironment(BenchmarkEnvironment):
    """Environment for evaluating question answering on SQuAD-style tasks."""

    def evaluate(
        self,
        sample: Sample,
        prediction: str,
        ground_truth: str,
        metrics: list[MetricConfig],
    ) -> dict[str, float]:
        """Evaluate QA prediction against ground truth."""
        # Handle SQuAD-style answers (list of possible answers)
        if isinstance(ground_truth, list) and ground_truth:
            # Use the first answer as primary ground truth
            ground_truth = ground_truth[0]

        # Compute all requested metrics
        metric_names = [metric.name for metric in metrics]
        results = compute_all_metrics(metric_names, prediction, ground_truth)

        # Generate feedback based on performance
        feedback = self._generate_feedback(prediction, ground_truth, results)
        results["feedback"] = feedback  # type: ignore

        return results

    def format_input(self, sample: Sample) -> str:
        """Format sample for model input (already formatted in preprocessing)."""
        return sample.question

    def extract_ground_truth(self, sample: Sample) -> str:
        """Extract ground truth answer from sample."""
        return sample.ground_truth

    def _generate_feedback(
        self, prediction: str, ground_truth: str, metrics: dict[str, float]
    ) -> str:
        """Generate human-readable feedback based on evaluation results."""
        exact_match = metrics.get("exact_match", 0.0)
        f1_score = metrics.get("f1", 0.0)

        if exact_match >= 0.9:
            return "Perfect match. Answer is correct and well-formatted."
        elif f1_score >= 0.7:
            return "Good performance. Answer contains key information but may need formatting improvements."
        elif f1_score >= 0.3:
            return "Partial match. Answer contains some relevant information but may be incomplete or inaccurate."
        else:
            return "Low performance. Answer may be incorrect, incomplete, or poorly formatted."

    def get_evaluation_summary(
        self, results: list[dict[str, float]]
    ) -> dict[str, float]:
        """Generate summary statistics for a batch of evaluations."""
        if not results:
            return {}

        # Calculate mean scores for each metric
        summary = {}
        metric_names = set()
        for result in results:
            metric_names.update(result.keys())

        # Exclude feedback from numeric calculations
        metric_names.discard("feedback")

        for metric_name in metric_names:
            values = [
                result.get(metric_name, 0.0)
                for result in results
                if metric_name in result
            ]
            if values:
                summary[f"{metric_name}_mean"] = float(np.mean(values))
                summary[f"{metric_name}_min"] = float(np.min(values))
                summary[f"{metric_name}_max"] = float(np.max(values))

        return summary
