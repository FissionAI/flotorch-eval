from abc import ABC, abstractmethod
from flotorch_eval.agent_eval.core.schemas import EvaluationScore, Trajectory


class UsageEstimationEngineInterface(ABC):
    """
    Abstract base class for usage estimation engines.
    """

    @abstractmethod
    async def compute_from_trajectory(self, trajectory: Trajectory) -> EvaluationScore:
        """
        Compute usage-based evaluation from a trajectory.

        Args:
            trajectory: The full agent trajectory

        Returns:
            EvaluationScore with usage metadata
        """
        pass
