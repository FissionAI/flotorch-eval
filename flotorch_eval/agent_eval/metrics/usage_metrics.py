from flotorch_eval.agent_eval.metrics.base import MetricConfig
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.common.cost_utils import calculate_cost_from_tokens
from flotorch_eval.common.token_utils import extract_token_usage_from_trajectory
from flotorch_eval.agent_eval.metrics.base import LLMBaseEval

class UsageMetric(LLMBaseEval):
    """
    Metric to compute cost and token usage of LLM usage per span and overall.

    This metric extracts token usage from the provided trajectory, estimates the cost using model pricing,
    and returns a summary including total cost, average cost per call, and a breakdown per model/span.
    """

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return "usage_summary"

    @property
    def needs_llm(self) -> bool:
        """Indicates whether this metric requires an LLM."""
        return False

    @property
    def run_async(self) -> bool:
        """Indicates that this metric should run asynchronously."""
        return True

    async def evaluate(self, trajectory: Trajectory, metric_params: MetricConfig = None) -> MetricResult:
        """
        Compute cost estimation for the trajectory using model pricing.

        Args:
            trajectory (Trajectory): The trajectory to evaluate.

        Returns:
            MetricResult: The result containing cost summary.
        """

        token_summary = extract_token_usage_from_trajectory(trajectory)

        cost_summary = await calculate_cost_from_tokens(token_summary)

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
