#!/usr/bin/env python3
"""ACE Benchmark Runner

Main script for running benchmarks on the ACE framework.
Supports multiple evaluation modes: baseline, offline learning, online learning, and comparison.

Usage:
    python run_benchmark.py --task simple_qa --mode baseline
    python run_benchmark.py --task simple_qa --mode offline --epochs 3
    python run_benchmark.py --task simple_qa --mode compare
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb

from blockether_foundation.ace.program import AceProgram, model
from blockether_foundation.benchmarks import BenchmarkTaskManager, Sample

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Main benchmark execution engine."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize the benchmark runner.

        Args:
            output_dir: Directory to save benchmark results.
        """
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.task_manager = BenchmarkTaskManager()

        # Initialize ACE program
        self.ace_program = AceProgram(
            generator_model=model,
            premade_playbook=None,
            last_history_messages=0,  # Disable history for clean evaluation
            enable_consensus=False,  # Disable consensus for faster evaluation
            playbook_file_path=None,  # Don't save playbook during evaluation
        )

    def run_benchmark(
        self,
        task_name: str,
        mode: str,
        split_ratio: float = 0.8,
        epochs: int = 1,
        limit: int | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run benchmark with specified configuration.

        Args:
            task_name: Name of the benchmark task
            mode: Evaluation mode (baseline, offline, online, compare)
            split_ratio: Train/test split ratio for offline learning
            epochs: Number of training epochs for offline learning
            limit: Maximum number of samples to evaluate
            seed: Random seed for reproducibility

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Starting benchmark: {task_name} in {mode} mode")

        # Set random seed for reproducibility
        random.seed(seed)

        # Validate and load task
        if not self.task_manager.validate_task(task_name):
            raise ValueError(f"Task validation failed: {task_name}")

        samples = self.task_manager.load_samples(task_name)
        logger.info(f"Loaded {len(samples)} samples for task: {task_name}")

        # Apply limit if specified
        if limit:
            samples = samples[:limit]
            logger.info(f"Limited to {len(samples)} samples")

        # Get environment and config
        environment = self.task_manager.get_environment(task_name)
        config = self.task_manager.get_config(task_name)

        if not environment or not config:
            raise ValueError(
                f"Failed to load environment or config for task: {task_name}"
            )

        # Run evaluation based on mode
        if mode == "baseline":
            results = self._run_baseline(samples, environment, config)
        elif mode == "offline":
            results = self._run_offline(
                samples, environment, config, split_ratio, epochs
            )
        elif mode == "online":
            results = self._run_online(samples, environment, config)
        elif mode == "compare":
            results = self._run_compare(
                samples, environment, config, split_ratio, epochs
            )
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")

        # Add metadata
        results["metadata"] = {
            "task": task_name,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "samples_evaluated": len(samples),
            "split_ratio": split_ratio,
            "epochs": epochs,
            "seed": seed,
            "config": (
                config.model_dump() if hasattr(config, "model_dump") else str(config)
            ),
        }

        # Save results
        self._save_results(results, task_name, mode)

        logger.info(f"Benchmark completed: {task_name} in {mode} mode")
        return results

    def _run_baseline(
        self, samples: list[Sample], environment, config
    ) -> dict[str, Any]:
        """Run baseline evaluation (no learning)."""
        logger.info("Running baseline evaluation")

        results = {
            "mode": "baseline",
            "samples": [],
            "summary": {},
        }

        for i, sample in enumerate(samples):
            logger.debug(f"Evaluating sample {i+1}/{len(samples)}")

            # Generate response using ACE
            formatted_input = environment.format_input(sample)
            response = self._generate_response(formatted_input)

            # Evaluate response
            ground_truth = environment.extract_ground_truth(sample)
            metrics = environment.evaluate(
                sample, response, ground_truth, config.metrics
            )

            result = {
                "sample_id": sample.id,
                "question": sample.question,
                "prediction": response,
                "ground_truth": ground_truth,
                "metrics": metrics,
                "feedback": metrics.get("feedback", ""),
            }
            results["samples"].append(result)

        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results["samples"], config.metrics)

        return results

    def _run_offline(
        self,
        samples: list[Sample],
        environment,
        config,
        split_ratio: float,
        epochs: int,
    ) -> dict[str, Any]:
        """Run offline learning evaluation."""
        logger.info(f"Running offline learning with {epochs} epochs")

        # Split data
        train_size = int(len(samples) * split_ratio)
        train_samples = samples[:train_size]
        test_samples = samples[train_size:]

        logger.info(
            f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}"
        )

        results = {
            "mode": "offline_train_test_split",
            "split_ratio": split_ratio,
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "epochs": epochs,
            "train_results": [],
            "test_results": [],
            "train_summary": {},
            "test_summary": {},
            "overfitting_gap": {},
        }

        # Training phase
        logger.info("Starting training phase")
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            for i, sample in enumerate(train_samples):
                logger.debug(f"Training on sample {i+1}/{len(train_samples)}")

                formatted_input = environment.format_input(sample)
                response = self._generate_response(formatted_input)

                ground_truth = environment.extract_ground_truth(sample)
                metrics = environment.evaluate(
                    sample, response, ground_truth, config.metrics
                )

                # Learning happens through ACE's internal mechanism
                result = {
                    "sample_id": f"{sample.id}_epoch_{epoch+1}",
                    "question": sample.question,
                    "prediction": response,
                    "ground_truth": ground_truth,
                    "metrics": metrics,
                    "feedback": metrics.get("feedback", ""),
                    "epoch": epoch + 1,
                    "split": "train",
                }
                results["train_results"].append(result)

        # Testing phase
        logger.info("Starting testing phase")
        for i, sample in enumerate(test_samples):
            logger.debug(f"Testing on sample {i+1}/{len(test_samples)}")

            formatted_input = environment.format_input(sample)
            response = self._generate_response(formatted_input)

            ground_truth = environment.extract_ground_truth(sample)
            metrics = environment.evaluate(
                sample, response, ground_truth, config.metrics
            )

            result = {
                "sample_id": sample.id,
                "question": sample.question,
                "prediction": response,
                "ground_truth": ground_truth,
                "metrics": metrics,
                "feedback": metrics.get("feedback", ""),
                "split": "test",
            }
            results["test_results"].append(result)

        # Calculate summaries
        results["train_summary"] = self._calculate_summary(
            results["train_results"], config.metrics
        )
        results["test_summary"] = self._calculate_summary(
            results["test_results"], config.metrics
        )

        # Calculate overfitting gap
        results["overfitting_gap"] = self._calculate_overfitting_gap(
            results["train_summary"], results["test_summary"]
        )

        # Combined summary (test performance is the main result)
        results["summary"] = results["test_summary"]

        return results

    def _run_online(self, samples: list[Sample], environment, config) -> dict[str, Any]:
        """Run online learning evaluation."""
        logger.info("Running online learning (sequential)")

        results = {
            "mode": "online",
            "samples": [],
            "summary": {},
        }

        for i, sample in enumerate(samples):
            logger.debug(f"Processing sample {i+1}/{len(samples)}")

            formatted_input = environment.format_input(sample)
            response = self._generate_response(formatted_input)

            ground_truth = environment.extract_ground_truth(sample)
            metrics = environment.evaluate(
                sample, response, ground_truth, config.metrics
            )

            result = {
                "sample_id": sample.id,
                "question": sample.question,
                "prediction": response,
                "ground_truth": ground_truth,
                "metrics": metrics,
                "feedback": metrics.get("feedback", ""),
            }
            results["samples"].append(result)

            # Learning happens continuously through ACE's internal mechanism

        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results["samples"], config.metrics)

        return results

    def _run_compare(
        self,
        samples: list[Sample],
        environment,
        config,
        split_ratio: float,
        epochs: int,
    ) -> dict[str, Any]:
        """Run comparison between baseline and offline learning."""
        logger.info("Running comparison: baseline vs offline learning")

        # Run baseline
        logger.info("Running baseline evaluation...")
        baseline_results = self._run_baseline(samples, environment, config)

        # Run offline learning
        logger.info("Running offline learning evaluation...")
        offline_results = self._run_offline(
            samples, environment, config, split_ratio, epochs
        )

        # Combine results
        results = {
            "mode": "comparison",
            "baseline": baseline_results,
            "offline": offline_results,
            "comparison": self._compare_results(baseline_results, offline_results),
            "summary": offline_results["summary"],  # Use offline as primary result
        }

        return results

    def _generate_response(self, input_text: str) -> str:
        """Generate response using ACE program."""
        try:
            # Create an agent with the ACE program as a pre-hook
            agent = Agent(
                model=self.ace_program.generator_model,
                pre_hooks=[self.ace_program.pre_hook()],
                debug_mode=False,  # Disable debug for cleaner benchmark output
                db=InMemoryDb(),  # Use in-memory database for evaluation
            )

            # Run the agent with the input text
            response = agent.run(input_text, stream=False)

            # Extract and return the content
            return response.content if response.content else ""

        except Exception as e:
            logger.error(f"Error generating response with ACE program: {e}")
            # Return a fallback response in case of errors
            return f"Error: Unable to generate response - {str(e)}"

    def _calculate_summary(self, sample_results: list[dict], metrics) -> dict[str, Any]:
        """Calculate summary statistics from sample results."""
        if not sample_results:
            return {}

        summary = {}
        metric_names = [m.name for m in metrics]

        for metric_name in metric_names:
            values = []
            for result in sample_results:
                if "metrics" in result and metric_name in result["metrics"]:
                    values.append(result["metrics"][metric_name])

            if values:
                import numpy as np

                summary[f"{metric_name}_mean"] = float(np.mean(values))
                summary[f"{metric_name}_min"] = float(np.min(values))
                summary[f"{metric_name}_max"] = float(np.max(values))
                summary[f"{metric_name}_std"] = float(np.std(values))

        return summary

    def _calculate_overfitting_gap(
        self, train_summary: dict[str, Any], test_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate overfitting gap between train and test performance."""
        gap = {}
        for key in train_summary:
            if key in test_summary:
                gap[key] = train_summary[key] - test_summary[key]
        return gap

    def _compare_results(
        self, baseline_results: dict, offline_results: dict
    ) -> dict[str, Any]:
        """Compare baseline and offline learning results."""
        comparison = {
            "baseline_better": [],
            "offline_better": [],
            "similar": [],
        }

        baseline_summary = baseline_results.get("summary", {})
        offline_summary = offline_results.get("summary", {})

        for metric in baseline_summary:
            if metric.endswith("_mean"):
                baseline_val = baseline_summary[metric]
                offline_val = offline_summary.get(metric, 0)

                diff = offline_val - baseline_val
                if diff > 0.05:  # 5% improvement threshold
                    comparison["offline_better"].append(
                        {
                            "metric": metric,
                            "improvement": diff,
                            "baseline": baseline_val,
                            "offline": offline_val,
                        }
                    )
                elif diff < -0.05:
                    comparison["baseline_better"].append(
                        {
                            "metric": metric,
                            "improvement": abs(diff),
                            "baseline": baseline_val,
                            "offline": offline_val,
                        }
                    )
                else:
                    comparison["similar"].append(
                        {
                            "metric": metric,
                            "baseline": baseline_val,
                            "offline": offline_val,
                        }
                    )

        return comparison

    def _save_results(self, results: dict[str, Any], task_name: str, mode: str):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_name}_{mode}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ACE benchmarks")
    parser.add_argument("--task", help="Benchmark task name")
    parser.add_argument(
        "--mode",
        choices=["baseline", "offline", "online", "compare"],
        default="baseline",
        help="Evaluation mode",
    )
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument(
        "--split-ratio", type=float, default=0.8, help="Train/test split ratio"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--list-tasks", action="store_true", help="List available tasks"
    )

    args = parser.parse_args()

    if args.list_tasks:
        task_manager = BenchmarkTaskManager()
        tasks = task_manager.list_tasks()
        print("Available tasks:")
        for task in tasks:
            print(f"  - {task}")
        return

    if not args.task:
        parser.error("--task is required unless --list-tasks is specified")

    # Initialize runner
    runner = BenchmarkRunner(
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    print("Starting benchmark execution...")
    print(f"Task: {args.task}, Mode: {args.mode}, Limit: {args.limit}")

    try:
        # Run benchmark
        results = runner.run_benchmark(
            task_name=args.task,
            mode=args.mode,
            split_ratio=args.split_ratio,
            epochs=args.epochs,
            limit=args.limit,
            seed=args.seed,
        )

        # Print summary
        print("\nBenchmark completed successfully!")
        print(f"Task: {args.task}")
        print(f"Mode: {args.mode}")
        print(
            f"Samples evaluated: {results.get('metadata', {}).get('samples_evaluated', 0)}"
        )

        summary = results.get("summary", {})
        if summary:
            print("\nSummary Statistics:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
