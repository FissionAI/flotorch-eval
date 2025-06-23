from typing import Optional

from ragas.dataset_schema import MultiTurnSample
from ragas.metrics import AgentGoalAccuracyWithReference, AgentGoalAccuracyWithoutReference

from flotorch_eval.agent_eval.core.schemas import EvaluationScore, Trajectory
from flotorch_eval.agent_eval.interfaces.base.agent_goal_interface import AgentGoalScoringEngine
from flotorch_eval.agent_eval.integrations.ragas_utils import convert_trajectory_to_ragas_messages
from ragas.llms import LangchainLLMWrapper
from flotorch_eval.agent_eval.metrics.base import MetricConfig


class RagasAgentGoalAccuracyEngine(AgentGoalScoringEngine):
    """Ragas-backed agent goal accuracy scoring engine with/without reference support."""

    def __init__(self, llm: LangchainLLMWrapper, config: Optional[MetricConfig] = None):
        self.llm = llm
        self.config = config or MetricConfig()
        metric_params = self.config.metric_params or {}

        # Choose evaluator based on presence of reference_answer
        if metric_params.get("reference_answer"):
            self.evaluator = AgentGoalAccuracyWithReference()
            self.has_reference = True
        else:
            self.evaluator = AgentGoalAccuracyWithoutReference()
            self.has_reference = False

        if not isinstance(self.llm, LangchainLLMWrapper):
            raise ValueError("LLM must be a LangchainLLMWrapper instance")

        self.evaluator.llm = self.llm

    
    async def compute_from_trajectory(self, trajectory: Trajectory) -> EvaluationScore:
        try:
            ragas_messages, _ = convert_trajectory_to_ragas_messages(trajectory)
            if not ragas_messages:
                return EvaluationScore(score=0.0, metadata={"error": "No user input extracted from trajectory"})

            sample_params = {"user_input": ragas_messages}
            if self.has_reference:
                sample_params["reference"] = self.config.metric_params["reference_answer"]

            sample = MultiTurnSample(**sample_params)
            score = await self.evaluator.multi_turn_ascore(sample)

            return EvaluationScore(
                score=score,
                metadata={
                    "evaluation_type": "agent_goal_with_reference" if self.has_reference else "agent_goal_without_reference"
                },
            )

        except Exception as e:
            return EvaluationScore(score=0.0, metadata={"error": str(e)})
