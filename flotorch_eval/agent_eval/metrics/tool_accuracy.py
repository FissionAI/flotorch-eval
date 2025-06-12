"""
Tool accuracy evaluation metrics.
"""

from typing import Dict, List, Optional, Union

from flotorch_eval.agent_eval.core.schemas import ToolCall, Trajectory
from flotorch_eval.agent_eval.metrics.base import BaseMetric, MetricResult


class ToolAccuracyMetric(BaseMetric):
    """Measures the accuracy of tool calls in a trajectory."""

    @property
    def name(self) -> str:
        return "tool_accuracy"
    
    def _setup(self) -> None:
        """No setup needed for this metric."""
        pass

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Compute tool accuracy score for a trajectory.

        The score is calculated as the ratio of successful tool calls to total tool calls.
        Tool calls are considered successful if:
        1. They have success=True
        2. They have no error
        3. They have valid outputs

        Args:
            trajectory: The trajectory to evaluate

        Returns:
            MetricResult with the accuracy score and detailed statistics
        """
        tool_calls: List[ToolCall] = []
        for message in trajectory.messages:
            tool_calls.extend(message.tool_calls)

        if not tool_calls:
            return MetricResult(
                name=self.name,
                score=1.0,  # Perfect score if no tool calls (vacuous truth)
                details={
                    "comment": "No tool calls were made in this trajectory.",
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "errors": [],
                },
            )

        successful = 0
        failed = 0
        errors = []

        for tool_call in tool_calls:
            if (
                tool_call.success
                and not tool_call.error
                and tool_call.output is not None
            ):
                successful += 1
            else:
                failed += 1
                if tool_call.error:
                    errors.append(
                        {"tool": tool_call.name, "error": tool_call.error}
                    )

        score = successful / len(tool_calls)

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "comment": "Tool usage statistics calculated successfully.",
                "total_calls": len(tool_calls),
                "successful_calls": successful,
                "failed_calls": failed,
                "errors": errors,
            },
        )
