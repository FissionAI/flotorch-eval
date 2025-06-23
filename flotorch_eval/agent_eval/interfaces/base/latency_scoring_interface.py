from abc import ABC, abstractmethod
from flotorch_eval.agent_eval.core.schemas import Trajectory, EvaluationScore

class LatencyScoringInterface(ABC):
    """Abstract base class for latency scoring engines."""

    @abstractmethod
    async def compute_from_trajectory(self, trajectory: Trajectory) -> EvaluationScore:
        """
        Compute latency score and metadata from a given trajectory.

        Args:
            trajectory: The full trajectory

        Returns:
            EvaluationScore containing score and metadata
        """
        pass
