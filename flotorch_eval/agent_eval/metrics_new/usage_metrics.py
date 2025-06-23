from typing import Optional
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.agent_eval.interfaces.base.usage_estimation_interface import UsageEstimationEngineInterface
from flotorch_eval.agent_eval.interfaces.usage.usage_estimation_engine import UsageEstimationEngine
from flotorch_eval.agent_eval.metrics.base import BaseMetric, MetricConfig


class UsageMetric(BaseMetric):
    """
    Evaluates token and cost usage using a pluggable estimation engine.
    """

    requires_llm = False

    def __init__(
        self,
        config: Optional[MetricConfig] = None,
        engine: Optional[UsageEstimationEngineInterface] = None,
    ):
        """
        Args:
            config: MetricConfig containing AWS region.
            engine: Optional custom usage estimation engine.
        """
        self.config = config or MetricConfig()
        self.engine = engine or UsageEstimationEngine(aws_region=self.config.metric_params.get("aws_region", ""))
        self._setup()

    @property
    def name(self) -> str:
        return "usage_summary"

    def _setup(self) -> None:
        if not self.config or not self.config.metric_params.get("aws_region"):
            raise ValueError("UsageMetric requires 'aws_region' in metric_params")

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Evaluate usage cost and tokens from a trajectory.

        Args:
            trajectory: The agent trajectory

        Returns:
            MetricResult with cost and usage summary
        """
        result = await self.engine.compute_from_trajectory(trajectory)

        details = {"evaluation_type": "usage_summary", **result.metadata}

        return MetricResult(
            name=self.name,
            score=result.score,
            details=details,
        )
