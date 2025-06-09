"""
Evaluator module for computing metrics on agent trajectories.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from flotorch_eval.agent_eval.core.schemas import EvaluationResult, Trajectory
from flotorch_eval.agent_eval.metrics.base import BaseMetric, MetricResult


class Evaluator:
    """Orchestrates the evaluation of agent trajectories using multiple metrics."""

    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        """
        Initialize evaluator with metrics.

        Args:
            metrics: List of metric instances to use for evaluation
        """
        self.metrics = metrics or []

    def add_metric(self, metric: BaseMetric) -> None:
        """Add a metric to the evaluator."""
        self.metrics.append(metric)

    def add_metrics(self, metrics: List[BaseMetric]) -> None:
        """Add multiple metrics to the evaluator."""
        self.metrics.extend(metrics)

    async def evaluate(
        self, trajectory: Trajectory, metrics: Optional[List[BaseMetric]] = None
    ) -> EvaluationResult:
        """
        Evaluate a trajectory using the configured metrics.

        Args:
            trajectory: The trajectory to evaluate
            metrics: Optional list of metrics to use instead of configured ones

        Returns:
            EvaluationResult containing scores from all metrics
        """
        metrics_to_use = metrics or self.metrics
        scores = []

        for metric in metrics_to_use:
            result = await metric.compute(trajectory)
            scores.append(result)

        return EvaluationResult(trajectory_id=trajectory.trace_id, scores=scores)
