import json
from typing import Optional
from flotorch_eval.agent_eval.core.schemas import EvaluationScore, Trajectory
from flotorch_eval.agent_eval.integrations.langchain_utils import convert_trajectory_to_langchain_format
from flotorch_eval.agent_eval.interfaces.base.trajectory_eval_interface import TrajectoryEvalEngine
from flotorch_eval.agent_eval.metrics.base import MetricConfig
from agentevals.trajectory.llm import (
    TRAJECTORY_ACCURACY_PROMPT,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    create_trajectory_llm_as_judge,
)


class LangChainTrajectoryEvalWithLLMEngine(TrajectoryEvalEngine):
    def __init__(self, llm, config: Optional[MetricConfig] = None):
        self.llm = llm
        self.config = config or MetricConfig()
        self._setup()

    def _setup(self):
        metric_params = self.config.metric_params or {}

        self.reference_outputs = metric_params.get("reference_outputs", [])
        model_identifier = metric_params.get("model")

        self.prompt = (
            TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
            if self.reference_outputs
            else TRAJECTORY_ACCURACY_PROMPT
        )

        self.evaluator = create_trajectory_llm_as_judge(
            prompt=self.prompt,
            judge=self.llm,
            model=model_identifier,
        )

    async def compute_from_trajectory(self, trajectory: Trajectory) -> EvaluationScore:
        outputs = convert_trajectory_to_langchain_format(trajectory)

        try:
            if self.reference_outputs:
                result = self.evaluator(
                    outputs=outputs, reference_outputs=self.reference_outputs
                )
            else:
                result = self.evaluator(outputs=outputs)

            score = 1.0 if result.get("score", False) else 0.0

            return EvaluationScore(
                score=score,
                metadata={
                    "comment": str(result.get("comment", "")),
                    "has_reference": bool(self.reference_outputs),
                    "raw_score": bool(result.get("score", False)),
                },
            )

        except Exception as e:
            return EvaluationScore(
                score=0.0,
                metadata={
                    "error": str(e),
                    "has_reference": bool(self.reference_outputs),
                },
            )

    
