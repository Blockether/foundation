"""Metrics computation utilities for benchmark evaluation.

This module provides implementations of common evaluation metrics used
for benchmarking language models and QA systems.
"""

from __future__ import annotations

import re


class ExactMatchMetric:
    """Exact match evaluation metric."""

    @staticmethod
    def compute(prediction: str, ground_truth: str) -> float:
        """Compute exact match score between prediction and ground truth."""
        # Normalize strings: strip whitespace and lowercase
        pred = str(prediction).strip().lower()
        truth = str(ground_truth).strip().lower()

        # For QA tasks, also check if the ground truth is contained in the prediction
        # This handles cases where the model gives a longer answer that contains the correct answer
        if truth in pred:
            return 1.0

        return float(pred == truth)


def _normalize_text(text: str) -> list[str]:
    """Normalize text and return list of tokens."""
    # Convert to lowercase and remove punctuation
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = text.split()
    return tokens


class F1Metric:
    """F1 score evaluation metric for QA tasks."""

    @staticmethod
    def compute(prediction: str, ground_truth: str) -> float:
        """Compute F1 score between prediction and ground truth."""
        # Tokenize and normalize
        pred_tokens = _normalize_text(prediction)
        truth_tokens = _normalize_text(ground_truth)

        if not pred_tokens and not truth_tokens:
            return 1.0

        if not pred_tokens or not truth_tokens:
            return 0.0

        # Compute precision, recall, F1
        common_tokens = set(pred_tokens) & set(truth_tokens)
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(truth_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1


class AccuracyMetric:
    """Simple accuracy metric for classification-style tasks."""

    @staticmethod
    def compute(prediction: str, ground_truth: str) -> float:
        """Compute accuracy score."""
        pred = str(prediction).strip().lower()
        truth = str(ground_truth).strip().lower()

        # For QA tasks, also check if the ground truth is contained in the prediction
        if truth in pred:
            return 1.0

        return float(pred == truth)


# Registry of available metrics
METRICS = {
    "exact_match": ExactMatchMetric,
    "f1": F1Metric,
    "accuracy": AccuracyMetric,
}


def compute_metric(metric_name: str, prediction: str, ground_truth: str) -> float:
    """Compute a specific metric between prediction and ground truth."""
    if metric_name not in METRICS:
        raise ValueError(
            f"Unknown metric: {metric_name}. Available: {list(METRICS.keys())}"
        )

    metric_class = METRICS[metric_name]
    return metric_class.compute(prediction, ground_truth)


def compute_all_metrics(
    metric_names: list[str], prediction: str, ground_truth: str
) -> dict[str, float]:
    """Compute multiple metrics and return results as a dictionary."""
    results = {}
    for metric_name in metric_names:
        try:
            results[metric_name] = compute_metric(metric_name, prediction, ground_truth)
        except ValueError as e:
            print(f"Warning: Failed to compute {metric_name}: {e}")
            results[metric_name] = 0.0

    return results


def compute_weighted_score(
    metrics_results: dict[str, float], metric_weights: dict[str, float]
) -> float:
    """Compute weighted score from metric results."""
    total_weight = sum(metric_weights.values())
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(
        metrics_results.get(metric, 0.0) * weight
        for metric, weight in metric_weights.items()
    )

    return weighted_sum / total_weight
