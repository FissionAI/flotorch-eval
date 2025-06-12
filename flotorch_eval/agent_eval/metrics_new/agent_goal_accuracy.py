from typing import Optional, Any
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.agent_eval.metrics.base import BaseMetric
from flotorch_eval.agent_eval.interfaces.base.agent_goal_interface import AgentGoalScoringEngine
from flotorch_eval.agent_eval.interfaces.ragas.ragas_agent_goal import RagasAgentGoalAccuracyEngine
from flotorch_eval.agent_eval.metrics.base import MetricConfig


class AgentGoalAccuracyMetric(BaseMetric):
    """
    Evaluates agent goal accuracy using a pluggable scoring engine.
    Supports both direct engine injection or default Ragas engine via LLM and config.
    """

    requires_llm = True

    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[MetricConfig] = None,
        engine: Optional[AgentGoalScoringEngine] = None,
    ):
        """
        Args:
            llm: Optional LLM used by Ragas engine (ignored if custom engine is passed)
            config: Optional metric config passed to the Ragas engine
            engine: Optional custom scoring engine
        """
        self.engine = engine or RagasAgentGoalAccuracyEngine(llm=llm, config=config)
        self._setup()

    @property
    def name(self) -> str:
        return "agent_goal_accuracy"

    def _setup(self) -> None:
        pass

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        score = await self.engine.compute_from_trajectory(trajectory)

        return MetricResult(
            name=self.name,
            score=score,
            details={"evaluation_type": "agent_goal_accuracy"},
        )
