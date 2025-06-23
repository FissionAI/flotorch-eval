from abc import ABC, abstractmethod
from flotorch_eval.agent_eval.core.schemas import Trajectory

class TrajectoryEvalEngine(ABC):
    """
    Abstract interface for trajectory evaluation using LLMs or other logic.
    """

    @abstractmethod
    async def compute_from_trajectory(self, trajectory: Trajectory) -> float:
        """
        Compute a trajectory-level accuracy score.

        Args:
            trajectory: The agent trajectory to evaluate

        Returns:
            A float score (e.g., 0.85)
        """
        pass
