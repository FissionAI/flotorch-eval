import json
from typing import Optional
from flotorch_eval.agent_eval.core.schemas import Trajectory
from flotorch_eval.agent_eval.core.schemas import EvaluationScore
from flotorch_eval.agent_eval.interfaces.base.trajectory_eval_interface import TrajectoryEvalEngine
from flotorch_eval.agent_eval.integrations.langchain_utils import convert_trajectory_to_langchain_format
from agentevals.trajectory.match import create_trajectory_match_evaluator


class LangChainTrajectoryEvalWithoutLLMEngine(TrajectoryEvalEngine):
    """
    Rule-based trajectory evaluator using match evaluation logic (no LLM).
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._setup()

    def _setup(self):
        match_modes = self.config or {}

        self.trajectory_match_mode = match_modes.get("trajectory_match_mode", "strict")
        self.tool_args_match_mode = match_modes.get("tool_args_match_mode", "exact")

        if self.trajectory_match_mode not in ("strict", "unordered", "subset", "superset"):
            raise ValueError(f"Invalid trajectory_match_mode: {self.trajectory_match_mode}")

        if self.tool_args_match_mode not in ("exact", "ignore", "subset", "superset"):
            raise ValueError(f"Invalid tool_args_match_mode: {self.tool_args_match_mode}")

        self.evaluator = create_trajectory_match_evaluator(
            trajectory_match_mode=self.trajectory_match_mode,
            tool_args_match_mode=self.tool_args_match_mode,
        )

    async def compute_from_trajectory(self, trajectory: Trajectory) -> EvaluationScore:
        reference_outputs = self.config.get("reference_outputs")

        if not reference_outputs:
            return EvaluationScore(
                score=0.0,
                metadata={"error": "Reference trajectory required for match evaluation"}
            )

        try:
            outputs = convert_trajectory_to_langchain_format(trajectory)

            result = self.evaluator(
                outputs=outputs, reference_outputs=reference_outputs
            )

            score = 1.0 if result.get("score", False) else 0.0

            return EvaluationScore(
                score=score,
                metadata={
                    "trajectory_match_mode": self.trajectory_match_mode,
                    "tool_args_match_mode": self.tool_args_match_mode,
                    "evaluation_details": result,
                }
            )

        except Exception as e:
            return EvaluationScore(
                score=0.0,
                metadata={"error": str(e)}
            )
