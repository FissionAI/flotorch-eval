from typing import Optional
from flotorch_eval.agent_eval.metrics.base import BaseMetric, MetricConfig
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.agent_eval.metrics.schema import LatencySummary
from flotorch_eval.common.latency_utils import extract_latency_from_trajectory  # assumed moved here

class LatencyMetric(BaseMetric):
    """Metric to compute latency per step and overall for a given trajectory."""

    requires_llm = False

    @property
    def name(self) -> str:
        return "latency_summary"

    def _setup(self) -> None:
        """
        No specific setup required for latency metric.
        """
        pass

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Compute latency summary across the trajectory steps.

        Args:
            trajectory: The trajectory to evaluate.

        Returns:
            MetricResult with latency summary.
        """
        latency_summary = extract_latency_from_trajectory(trajectory)

        return MetricResult(
            name=self.name,
            score=0.0, 
            details=latency_summary
        )
