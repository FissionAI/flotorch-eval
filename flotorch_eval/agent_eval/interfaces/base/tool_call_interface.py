from abc import ABC, abstractmethod
from flotorch_eval.agent_eval.core.schemas import Trajectory

class ToolCallScoringEngine(ABC):
    """Abstract interface for tool call accuracy scoring."""

    @abstractmethod
    async def compute_from_trajectory(self, trajectory: Trajectory) -> float:
        """
        Compute the tool call accuracy score directly from the trajectory.

        The engine handles all internal logic (conversion, sample building, scoring).

        Args:
            trajectory: The full agent trajectory (spans, messages, etc.)

        Returns:
            A score between 0.0 and 1.0
        """
        pass
