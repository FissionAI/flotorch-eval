from typing import Optional, List
from ragas.dataset_schema import MultiTurnSample
from ragas.metrics import ToolCallAccuracy
from ragas import messages as r

from flotorch_eval.agent_eval.core.schemas import EvaluationScore, Trajectory
from flotorch_eval.agent_eval.integrations.ragas_utils import convert_trajectory_to_ragas_messages
from flotorch_eval.agent_eval.interfaces.base.tool_call_interface import ToolCallScoringEngine


class RagasToolCallAccuracyEngine(ToolCallScoringEngine):
    """Ragas-backed implementation of tool call accuracy scoring."""

    def __init__(self):
        self.evaluator = ToolCallAccuracy()

    async def compute_from_trajectory(self, trajectory: Trajectory) -> EvaluationScore:
        """
        Compute score using Ragas directly from the Trajectory.

        Args:
            trajectory: The full trajectory (from spans/messages)

        Returns:
            EvaluationScore: Contains score and metadata
        """
        try:
            messages, references = convert_trajectory_to_ragas_messages(trajectory)

            if not references:
                return EvaluationScore(
                    score=0.0,
                    metadata={"error": "No reference tool calls found"}
                )

            sample = MultiTurnSample(
                user_input=messages,
                reference_tool_calls=references
            )

            score = await self.evaluator.multi_turn_ascore(sample)

            return EvaluationScore(
                score=score,
                metadata={"evaluation_type": "tool_call_accuracy"}
            )

        except Exception as e:
            return EvaluationScore(
                score=0.0,
                metadata={"error": f"[RagasToolCallAccuracyEngine] {str(e)}"}
            )
