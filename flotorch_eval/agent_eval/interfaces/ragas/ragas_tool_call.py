from typing import Optional, List
from ragas.dataset_schema import MultiTurnSample
from ragas.metrics import ToolCallAccuracy
from ragas import messages as r

from flotorch_eval.agent_eval.core.schemas import Trajectory
from flotorch_eval.agent_eval.integrations.ragas_utils import convert_to_ragas_format
from flotorch_eval.agent_eval.interfaces.base.tool_call_interface import ToolCallScoringEngine


class RagasToolCallAccuracyEngine(ToolCallScoringEngine):
    """Ragas-backed implementation of tool call accuracy scoring."""

    def __init__(self):
        self.evaluator = ToolCallAccuracy()

    async def compute_from_trajectory(self, trajectory: Trajectory) -> float:
        """
        Compute score using Ragas directly from the Trajectory.

        Args:
            trajectory: The full trajectory (from spans/messages)

        Returns:
            A score between 0.0 and 1.0
        """
        try:
            messages, references = convert_to_ragas_format(trajectory)

            if not references:
                return 0.0

            sample = MultiTurnSample(
                user_input=messages,
                reference_tool_calls=references
            )

            return await self.evaluator.multi_turn_ascore(sample)

        except Exception as e:
            print(f"[RagasToolCallAccuracyEngine] Error during scoring: {e}")
            return 0.0
