from typing import Optional, Any

from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.agent_eval.interfaces.langchain.trajectory_with_llm import LangChainTrajectoryEvalWithLLMEngine
from flotorch_eval.agent_eval.metrics.base import BaseMetric, MetricConfig
from flotorch_eval.agent_eval.interfaces.base.trajectory_eval_interface import TrajectoryEvalEngine



class TrajectoryEvalWithLLMMetric(BaseMetric):
    """
    Evaluates agent trajectory using an LLM as a judge.
    Uses a pluggable evaluation engine.
    """

    requires_llm = True

    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[MetricConfig] = None,
        engine: Optional[TrajectoryEvalEngine] = None,
    ):
        """
        Args:
            llm: Optional LLM to use in default engine
            config: Optional config passed to the engine
            engine: Optional custom trajectory evaluation engine, Mandatory if llm and config not provided
        """
        self.engine = engine or LangChainTrajectoryEvalWithLLMEngine(llm=llm, config=config)
        self._setup()

    @property
    def name(self) -> str:
        return "trajectory_eval_with_llm"

    def _setup(self) -> None:
        pass

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        result = await self.engine.compute_from_trajectory(trajectory)

        return MetricResult(
            name=self.name,
            score=result.score,
            details=result.metadata,
        )
