

from flotorch_eval.agent_eval.interfaces.base.latency_scoring_interface import LatencyScoringInterface
from flotorch_eval.common.latency_utils import extract_latency_from_trajectory
from flotorch_eval.agent_eval.core.schemas import EvaluationScore, Trajectory


class LatencyScoringEngine(LatencyScoringInterface):
    """
    Basic latency engine that extracts latency stats from the trajectory.
    """

    async def compute_from_trajectory(self, trajectory: Trajectory) -> EvaluationScore:
        """
        Compute latency summary using simple extraction logic.

        Args:
            trajectory: The trajectory to analyze

        Returns:
            EvaluationScore: Contains latency stats as metadata
        """
        try:
            latency_summary = extract_latency_from_trajectory(trajectory)

            return EvaluationScore(
                score=latency_summary.total_latency_ms,
                metadata=latency_summary.to_dict()
            )
        except Exception as e:
            return EvaluationScore(
                score=0.0,
                metadata={"error": f"[BasicLatencyEngine] {str(e)}"}
            )
