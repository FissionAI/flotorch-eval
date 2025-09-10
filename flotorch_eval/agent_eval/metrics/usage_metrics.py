from flotorch_eval.agent_eval.metrics.base import MetricConfig
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.common.cost_utils import calculate_cost_from_tokens
from flotorch_eval.common.token_utils import extract_token_usage_from_trajectory
from flotorch_eval.agent_eval.metrics.base import LLMBaseEval

class UsageMetric(LLMBaseEval):
    """
    Metric to compute cost and token usage of LLM usage per span and overall.

    This metric extracts token usage from the provided trajectory, estimates the cost using AWS pricing,
    and returns a summary including total cost, average cost per call, and a breakdown per model/span.

    Attributes:
        aws_region (str): The AWS region to use for cost calculation, must be provided in metric_params.
    """

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return "usage_summary"

    @property
    def needs_llm(self) -> bool:
        """Indicates whether this metric requires an LLM."""
        return False

    def evaluate(self, trajectory: Trajectory, metric_params: MetricConfig) -> MetricResult:
        """
        Compute cost estimation for the trajectory using AWS pricing.

        Args:
            trajectory (Trajectory): The trajectory to evaluate.
            metric_params (MetricConfig): Metric parameters, must include 'aws_region'.

        Returns:
            MetricResult: The result containing cost summary.

        Raises:
            ValueError: If 'aws_region' is not provided in metric_params.
        """
        if not metric_params or not metric_params.get("aws_region"):
            raise ValueError("CostMetric requires 'aws_region' in metric_params")
            
        self.aws_region = metric_params["aws_region"]
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
