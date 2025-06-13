from typing import Optional
from flotorch_eval.agent_eval.metrics.base import BaseMetric, MetricConfig
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.common.cost_utils import calculate_cost_from_tokens
from flotorch_eval.common.token_utils import extract_token_usage_from_trajectory

class UsageMetric(BaseMetric):
    """Metric to compute cost/token of LLM usage per span and overall."""

    requires_llm = False

    @property
    def name(self) -> str:
        return "usage_summary"

    def _setup(self) -> None:
        """
        Validate required configuration like AWS region.
        """
        if not self.config or not self.config.metric_params.get("aws_region"):
            raise ValueError("CostMetric requires 'aws_region' in metric_params")
        self.aws_region = self.config.metric_params["aws_region"]

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Compute cost estimation for the trajectory using Bedrock pricing.

        Args:
            trajectory: The trajectory to evaluate.

        Returns:
            MetricResult with cost summary.
        """
        token_summary = extract_token_usage_from_trajectory(trajectory)

        cost_summary = calculate_cost_from_tokens(token_summary, aws_region=self.aws_region)

        return MetricResult(
            name=self.name,
            score=0.0,
            details={
                "total_cost": cost_summary.total_cost,
                "average_cost_per_call": cost_summary.average_cost_per_call,
                "cost_breakdown": [
                    {
                        "model": record.model,
                        "input_tokens": record.input_tokens,
                        "output_tokens": record.output_tokens,
                        "cost": record.cost
                    }
                    for record in cost_summary.cost_breakdown
                ]
            }
        )
