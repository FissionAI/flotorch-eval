from typing import Optional, Any

from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.agent_eval.metrics.base import BaseMetric, MetricConfig
from flotorch_eval.agent_eval.interfaces.base.trajectory_eval_interface import TrajectoryEvalEngine
from flotorch_eval.agent_eval.interfaces.langchain.trajectory_without_llm import LangChainTrajectoryEvalWithoutLLMEngine


class TrajectoryEvalWithoutLLMMetric(BaseMetric):
    """
    Evaluates agent trajectory using rule-based matching (no LLM).
    Uses a pluggable evaluation engine.
    """

    requires_llm = False

    def __init__(
        self,
        llm: Optional[Any] = None,  # not used, but included for interface symmetry
        config: Optional[MetricConfig] = None,
        engine: Optional[TrajectoryEvalEngine] = None,
    ):
        """
        Args:
            llm: Ignored (kept for consistent interface)
            config: Optional config passed to the engine
            engine: Optional custom trajectory evaluation engine. Required if config not provided.
        """
        self.engine = engine or LangChainTrajectoryEvalWithoutLLMEngine(config=config.metric_params if config else {})
        self._setup()

    @property
    def name(self) -> str:
        return "trajectory_eval_without_llm"

    def _setup(self) -> None:
        pass

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        result = await self.engine.compute_from_trajectory(trajectory)

        details = {
            "evaluation_type": "trajectory_eval_without_llm"
        }
        details.update(result.metadata)

        return MetricResult(
            name=self.name,
            score=result.score,
            details=details,
        )
