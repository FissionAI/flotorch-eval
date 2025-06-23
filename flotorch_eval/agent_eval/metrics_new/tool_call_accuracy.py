from typing import Optional
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.agent_eval.metrics.base import BaseMetric
from flotorch_eval.agent_eval.interfaces.base.tool_call_interface import ToolCallScoringEngine
from flotorch_eval.agent_eval.interfaces.ragas.ragas_tool_call import RagasToolCallAccuracyEngine


class ToolCallAccuracyMetric(BaseMetric):
    """
    Evaluates tool call accuracy using a pluggable scoring engine.
    The engine must implement ToolCallScoringEngine.
    """

    requires_llm = False

    def __init__(self, engine: Optional[ToolCallScoringEngine] = None):
        """
        Args:
            engine: A scoring engine that implements ToolCallScoringEngine.
        """
        self.engine = engine or RagasToolCallAccuracyEngine()
        self._setup()

    @property
    def name(self) -> str:
        return "tool_call_accuracy"

    def _setup(self) -> None:
        pass  # No setup needed if engine is passed directly

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Evaluate the tool call accuracy using the provided scoring engine.

        Args:
            trajectory: The agent trajectory

        Returns:
            MetricResult with score and optional details
        """
        result = await self.engine.compute_from_trajectory(trajectory)

        details = {"evaluation_type": "tool_call_accuracy", **result.metadata}

        return MetricResult(
            name=self.name,
            score=result.score,
            details=details,
        )
