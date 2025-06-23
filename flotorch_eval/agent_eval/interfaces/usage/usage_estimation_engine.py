from typing import Optional
from flotorch_eval.agent_eval.core.schemas import EvaluationScore, Trajectory
from flotorch_eval.agent_eval.interfaces.base.usage_estimation_interface import UsageEstimationEngineInterface
from flotorch_eval.common.cost_utils import calculate_cost_from_tokens
from flotorch_eval.common.token_utils import extract_token_usage_from_trajectory


class UsageEstimationEngine(UsageEstimationEngineInterface):
    """Engine to estimate usage cost and token statistics."""

    def __init__(self, aws_region: str):
        if not aws_region:
            raise ValueError("UsageEstimationEngine requires a valid AWS region")
        self.aws_region = aws_region

    async def compute_from_trajectory(self, trajectory: Trajectory) -> EvaluationScore:
        """
        Compute usage metrics from the trajectory.

        Args:
            trajectory: The full trajectory with span-level token usage

        Returns:
            EvaluationScore containing cost and token details
        """
        try:
            token_summary = extract_token_usage_from_trajectory(trajectory)
            cost_summary = calculate_cost_from_tokens(token_summary, aws_region=self.aws_region)

            return EvaluationScore(
                score="N/A", 
                metadata={
                    "evaluation_type": "usage_summary",
                    "total_cost": cost_summary.total_cost,
                    "average_cost_per_call": cost_summary.average_cost_per_call,
                    "cost_breakdown": [
                        {
                            "model": record.model,
                            "input_tokens": record.input_tokens,
                            "output_tokens": record.output_tokens,
                            "cost": record.cost,
                        }
                        for record in cost_summary.cost_breakdown
                    ]
                }
            )
        except Exception as e:
            return EvaluationScore(
                score="N/A",
                metadata={"error": f"[UsageEstimationEngine] {str(e)}"}
            )
