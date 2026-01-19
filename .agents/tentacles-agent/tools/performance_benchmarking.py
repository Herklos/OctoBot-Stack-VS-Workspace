"""
Performance Benchmarking Framework for OctoBot Tentacles Agent

This module provides performance benchmarking capabilities including baseline measurement,
regression testing, and automated performance comparison for tentacle testing operations.
"""

import json
import time
import statistics
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Individual benchmark measurement result"""

    operation: str
    timestamp: datetime
    duration_seconds: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline data for regression testing"""

    operation: str
    baseline_duration_seconds: float
    baseline_memory_mb: Optional[float] = None
    baseline_cpu_percent: Optional[float] = None
    tolerance_percent: float = 10.0  # Allow 10% degradation by default
    sample_count: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RegressionReport:
    """Regression analysis report"""

    operation: str
    baseline: PerformanceBaseline
    current_result: BenchmarkResult
    duration_regression_percent: float
    memory_regression_percent: Optional[float] = None
    cpu_regression_percent: Optional[float] = None
    is_regression: bool = False
    recommendations: List[str] = field(default_factory=list)


class PerformanceBenchmarker:
    """Performance benchmarking and regression testing framework"""

    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = baseline_file
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.current_results: List[BenchmarkResult] = []
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load performance baselines from file"""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, "r") as f:
                    data = json.load(f)
                    for key, baseline_data in data.items():
                        baseline_data["created_at"] = datetime.fromisoformat(
                            baseline_data["created_at"]
                        )
                        baseline_data["updated_at"] = datetime.fromisoformat(
                            baseline_data["updated_at"]
                        )
                        self.baselines[key] = PerformanceBaseline(**baseline_data)
                logger.info(f"Loaded {len(self.baselines)} performance baselines")
            except Exception as e:
                logger.warning(f"Failed to load baselines: {e}")

    def _save_baselines(self) -> None:
        """Save performance baselines to file"""
        try:
            data = {}
            for key, baseline in self.baselines.items():
                baseline_dict = {
                    "operation": baseline.operation,
                    "baseline_duration_seconds": baseline.baseline_duration_seconds,
                    "baseline_memory_mb": baseline.baseline_memory_mb,
                    "baseline_cpu_percent": baseline.baseline_cpu_percent,
                    "tolerance_percent": baseline.tolerance_percent,
                    "sample_count": baseline.sample_count,
                    "created_at": baseline.created_at.isoformat(),
                    "updated_at": baseline.updated_at.isoformat(),
                }
                data[key] = baseline_dict

            with open(self.baseline_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def benchmark_operation(
        self, operation_name: str, func: Callable, *args, **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a single operation

        Args:
            operation_name: Name of the operation being benchmarked
            func: Function to benchmark
            *args, **kwargs: Arguments to pass to the function

        Returns:
            BenchmarkResult with performance metrics
        """
        start_time = time.time()

        # Get initial resource usage
        process = None
        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            initial_cpu = process.cpu_percent(interval=0.1)
        except ImportError:
            initial_memory = None
            initial_cpu = None

        # Execute the operation
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Benchmark operation {operation_name} failed: {e}")
            raise

        end_time = time.time()
        duration = end_time - start_time

        # Get final resource usage
        try:
            if process is not None and initial_memory is not None:
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = final_memory - initial_memory
            else:
                memory_usage = None

            if process is not None and initial_cpu is not None:
                final_cpu = process.cpu_percent(interval=0.1)
                cpu_usage = final_cpu
            else:
                cpu_usage = None
        except:
            memory_usage = None
            cpu_usage = None

        benchmark_result = BenchmarkResult(
            operation=operation_name,
            timestamp=datetime.now(),
            duration_seconds=duration,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            metadata={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
        )

        self.current_results.append(benchmark_result)
        logger.info(
            f"Benchmarked {operation_name}: {duration:.2f}s, "
            f"Memory: {memory_usage:.1f} MB, CPU: {cpu_usage:.1f}%"
            if memory_usage and cpu_usage
            else f"Benchmarked {operation_name}: {duration:.2f}s"
        )

        return benchmark_result

    def establish_baseline(
        self,
        operation_name: str,
        results: List[BenchmarkResult],
        tolerance_percent: float = 10.0,
    ) -> PerformanceBaseline:
        """
        Establish a performance baseline from multiple benchmark results

        Args:
            operation_name: Name of the operation
            results: List of benchmark results to average
            tolerance_percent: Acceptable performance degradation percentage

        Returns:
            PerformanceBaseline for regression testing
        """
        if not results:
            raise ValueError("Cannot establish baseline without benchmark results")

        durations = [r.duration_seconds for r in results]
        memory_usages = [
            r.memory_usage_mb for r in results if r.memory_usage_mb is not None
        ]
        cpu_usages = [
            r.cpu_usage_percent for r in results if r.cpu_usage_percent is not None
        ]

        baseline = PerformanceBaseline(
            operation=operation_name,
            baseline_duration_seconds=statistics.mean(durations),
            baseline_memory_mb=statistics.mean(memory_usages)
            if memory_usages
            else None,
            baseline_cpu_percent=statistics.mean(cpu_usages) if cpu_usages else None,
            tolerance_percent=tolerance_percent,
            sample_count=len(results),
        )

        self.baselines[operation_name] = baseline
        self._save_baselines()

        logger.info(
            f"Established baseline for {operation_name}: "
            f"Duration: {baseline.baseline_duration_seconds:.2f}s, "
            f"Memory: {baseline.baseline_memory_mb:.1f} MB, "
            f"CPU: {baseline.baseline_cpu_percent:.1f}%"
            if baseline.baseline_cpu_percent
            else f"Established baseline for {operation_name}: Duration: {baseline.baseline_duration_seconds:.2f}s"
        )

        return baseline

    def check_regression(self, result: BenchmarkResult) -> Optional[RegressionReport]:
        """
        Check if a benchmark result shows performance regression

        Args:
            result: Benchmark result to check

        Returns:
            RegressionReport if regression detected, None otherwise
        """
        if result.operation not in self.baselines:
            return None

        baseline = self.baselines[result.operation]

        # Calculate regression percentages
        if baseline.baseline_duration_seconds == 0:
            duration_regression = float("inf") if result.duration_seconds > 0 else 0.0
        else:
            duration_regression = (
                (result.duration_seconds - baseline.baseline_duration_seconds)
                / baseline.baseline_duration_seconds
            ) * 100

        memory_regression = None
        if (
            result.memory_usage_mb is not None
            and baseline.baseline_memory_mb is not None
            and baseline.baseline_memory_mb != 0
        ):
            memory_regression = (
                (result.memory_usage_mb - baseline.baseline_memory_mb)
                / baseline.baseline_memory_mb
            ) * 100

        cpu_regression = None
        if (
            result.cpu_usage_percent is not None
            and baseline.baseline_cpu_percent is not None
            and baseline.baseline_cpu_percent != 0
        ):
            cpu_regression = (
                (result.cpu_usage_percent - baseline.baseline_cpu_percent)
                / baseline.baseline_cpu_percent
            ) * 100

        # Check if any metric exceeds tolerance
        is_regression = (
            abs(duration_regression) > baseline.tolerance_percent
            or (
                memory_regression is not None
                and abs(memory_regression) > baseline.tolerance_percent
            )
            or (
                cpu_regression is not None
                and abs(cpu_regression) > baseline.tolerance_percent
            )
        )

        if is_regression:
            recommendations = []
            if abs(duration_regression) > baseline.tolerance_percent:
                recommendations.append(
                    f"Duration regression: {duration_regression:.1f}% - consider optimizing execution time"
                )
            if (
                memory_regression
                and abs(memory_regression) > baseline.tolerance_percent
            ):
                recommendations.append(
                    f"Memory regression: {memory_regression:.1f}% - check for memory leaks"
                )
            if cpu_regression and abs(cpu_regression) > baseline.tolerance_percent:
                recommendations.append(
                    f"CPU regression: {cpu_regression:.1f}% - review CPU-intensive operations"
                )

            report = RegressionReport(
                operation=result.operation,
                baseline=baseline,
                current_result=result,
                duration_regression_percent=duration_regression,
                memory_regression_percent=memory_regression,
                cpu_regression_percent=cpu_regression,
                is_regression=True,
                recommendations=recommendations,
            )

            duration_str = (
                f"{duration_regression:.1f}"
                if duration_regression != float("inf")
                else "∞"
            )
            memory_str = (
                f"{memory_regression:.1f}" if memory_regression is not None else "0.0"
            )
            cpu_str = f"{cpu_regression:.1f}" if cpu_regression is not None else "0.0"

            logger.warning(
                f"Performance regression detected in {result.operation}: "
                f"duration +{duration_str}%, memory +{memory_str}%, CPU +{cpu_str}%"
            )
            return report

        return None

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of all benchmark results and regressions

        Returns:
            Dictionary with performance summary data
        """
        regressions = []
        for result in self.current_results:
            regression = self.check_regression(result)
            if regression:
                regressions.append(regression)

        summary = {
            "total_operations": len(self.current_results),
            "total_baselines": len(self.baselines),
            "regressions_detected": len(regressions),
            "regressions": [
                {
                    "operation": r.operation,
                    "duration_regression": r.duration_regression_percent,
                    "memory_regression": r.memory_regression_percent,
                    "cpu_regression": r.cpu_regression_percent,
                    "recommendations": r.recommendations,
                }
                for r in regressions
            ],
            "operations_summary": [
                {
                    "operation": r.operation,
                    "duration": r.duration_seconds,
                    "memory_mb": r.memory_usage_mb,
                    "cpu_percent": r.cpu_usage_percent,
                }
                for r in self.current_results
            ],
        }

        return summary

    def export_results(self, output_file: str, format: str = "json") -> None:
        """
        Export benchmark results to file

        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
        """
        if format == "json":
            data = {
                "timestamp": datetime.now().isoformat(),
                "results": [
                    {
                        "operation": r.operation,
                        "timestamp": r.timestamp.isoformat(),
                        "duration_seconds": r.duration_seconds,
                        "memory_usage_mb": r.memory_usage_mb,
                        "cpu_usage_percent": r.cpu_usage_percent,
                        "metadata": r.metadata,
                    }
                    for r in self.current_results
                ],
                "summary": self.get_performance_summary(),
            }

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            import csv

            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "operation",
                        "timestamp",
                        "duration_seconds",
                        "memory_usage_mb",
                        "cpu_usage_percent",
                    ]
                )

                for result in self.current_results:
                    writer.writerow(
                        [
                            result.operation,
                            result.timestamp.isoformat(),
                            result.duration_seconds,
                            result.memory_usage_mb or "",
                            result.cpu_usage_percent or "",
                        ]
                    )

        logger.info(f"Exported benchmark results to {output_file} in {format} format")
