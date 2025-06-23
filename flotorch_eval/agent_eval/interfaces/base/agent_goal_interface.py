from abc import ABC, abstractmethod
from flotorch_eval.agent_eval.core.schemas import Trajectory

class AgentGoalScoringEngine(ABC):
    """Abstract interface for agent goal accuracy scoring."""

    @abstractmethod
    async def compute_from_trajectory(self, trajectory: Trajectory) -> float:
        """
        Compute the agent goal accuracy score directly from the trajectory.

        Args:
            trajectory: The full agent trajectory (spans, messages, etc.)

        Returns:
            A score between 0.0 and 1.0
        """
        pass
