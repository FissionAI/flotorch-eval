from typing import Optional
from flotorch_eval.agent_eval.interfaces.base.latency_scoring_interface import LatencyScoringInterface
from flotorch_eval.agent_eval.interfaces.latency.latency_scoring_engine import LatencyScoringEngine
from flotorch_eval.agent_eval.metrics.base import BaseMetric
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory


class LatencyMetric(BaseMetric):
    """
    Evaluates latency using a pluggable latency scoring engine.
    """

    requires_llm = False

    def __init__(self, engine: Optional[LatencyScoringInterface] = None):
        """
        Args:
            engine: A scoring engine that implements LatencyScoringEngine.
        """
        self.engine = engine or LatencyScoringEngine()
        self._setup()

    @property
    def name(self) -> str:
        return "latency_summary"

    def _setup(self) -> None:
        pass  # No additional setup required

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Evaluate latency from the agent trajectory.

        Args:
            trajectory: The trajectory to evaluate

        Returns:
            MetricResult with score and latency details
        """
        result = await self.engine.compute_from_trajectory(trajectory)

        details = {"evaluation_type": "latency_summary", **result.metadata}

        return MetricResult(
            name=self.name,
            score=result.score,
            details=details,
        )
