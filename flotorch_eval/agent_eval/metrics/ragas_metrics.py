"""
Ragas-based evaluation metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import ragas.messages as r
from ragas import evaluate
from ragas.dataset_schema import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
    ToolCallAccuracy,
)

from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.agent_eval.metrics.base import BaseMetric, MetricConfig
from flotorch_eval.agent_eval.integrations.ragas_utils import convert_to_ragas_format


class RagasMetricMixin:
    """Mixin class providing common functionality for Ragas metrics."""

    def _convert_trajectory_to_ragas(
        self, trajectory: Trajectory
    ) -> Tuple[List[r.Message], List[r.ToolCall]]:
        """
        Convert a trajectory to Ragas message format.

        Args:
            trajectory: The trajectory to convert

        Returns:
            Tuple of (ragas_messages, reference_tool_calls)
        """
        ragas_messages = []
        reference_tool_calls = []

        for msg in trajectory.messages:
            if msg.role == "user":
                ragas_messages.append(r.HumanMessage(content=msg.content))

            elif msg.role == "assistant":
                # Convert tool calls to Ragas format
                tool_calls = []
                for tc in msg.tool_calls:
                    ragas_tool_call = r.ToolCall(name=tc.name, args=tc.arguments)
                    tool_calls.append(ragas_tool_call)
                    reference_tool_calls.append(ragas_tool_call)

                ragas_messages.append(
                    r.AIMessage(
                        content=msg.content,
                        tool_calls=tool_calls if tool_calls else None,
                    )
                )

            elif msg.role == "tool":
                ragas_messages.append(r.ToolMessage(content=msg.content))

        return ragas_messages, reference_tool_calls


class ToolCallAccuracyMetric(BaseMetric, RagasMetricMixin):
    """Evaluates the agent's tool call accuracy."""

    requires_llm = False

    @property
    def name(self) -> str:
        return "tool_call_accuracy"

    def _setup(self) -> None:
        """Setup the Ragas tool call accuracy evaluator."""
        self.evaluator = ToolCallAccuracy()

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Compute tool call accuracy score for the trajectory.

        Args:
            trajectory: The trajectory to evaluate

        Returns:
            MetricResult with tool call accuracy score
        """
        # Convert trajectory to Ragas format
        ragas_messages, reference_tool_calls = self._convert_trajectory_to_ragas(
            trajectory
        )

        # Only evaluate if we have reference tool calls
        if not reference_tool_calls:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No tool calls found to evaluate"},
            )

        # Evaluate
        score = await self._evaluate_interaction(
            messages=ragas_messages, reference_tool_calls=reference_tool_calls
        )

        if not score:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "Failed to evaluate interaction"},
            )

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "comment": "Evaluation completed successfully.",
                "evaluation_type": "tool_call_accuracy"},
        )

    async def _evaluate_interaction(
        self,
        messages: List[r.Message],
        reference_tool_calls: Optional[List[r.ToolCall]] = None,
        reference_answer: Optional[str] = None,
    ) -> Optional[float]:
        """Evaluate interaction using Ragas."""
        if not messages:
            return None

        # Create sample with only required parameters
        sample_params = {"user_input": messages}
        if reference_tool_calls:
            sample_params["reference_tool_calls"] = reference_tool_calls
        if reference_answer:
            sample_params["reference"] = reference_answer

        try:
            sample = MultiTurnSample(**sample_params)
            score = await self.evaluator.multi_turn_ascore(sample)
            return score
        except Exception as e:
            print(f"Error evaluating interaction: {e}")
            return None


class AgentGoalAccuracyMetric(BaseMetric, RagasMetricMixin):
    """Evaluates the agent's goal accuracy."""

    requires_llm = True

    @property
    def name(self) -> str:
        return "agent_goal_accuracy"

    def _setup(self) -> None:
        """Setup the Ragas goal accuracy evaluator."""
        metric_params = self.config.metric_params if self.config else {}

        # Determine which evaluator to use based on whether reference is provided
        if metric_params.get("reference_answer"):
            self.evaluator = AgentGoalAccuracyWithReference()
            self.has_reference = True
        else:
            self.evaluator = AgentGoalAccuracyWithoutReference()
            self.has_reference = False

        # Set LLM for evaluator
        if not isinstance(self.llm, LangchainLLMWrapper):
            raise ValueError("LLM must be a LangchainLLMWrapper instance")
        self.evaluator.llm = self.llm

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Compute goal accuracy score for the trajectory.

        Args:
            trajectory: The trajectory to evaluate

        Returns:
            MetricResult with goal accuracy score
        """
        # Convert trajectory to Ragas format
        ragas_messages, _ = self._convert_trajectory_to_ragas(trajectory)

        # Get reference answer if available
        reference_answer = (
            self.config.metric_params.get("reference_answer") if self.config else None
        )

        # Evaluate
        score = await self._evaluate_interaction(
            messages=ragas_messages,
            reference_answer=reference_answer if self.has_reference else None,
        )

        if not score:
            return MetricResult(name=self.name, score=0.0, details={})

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "evaluation_type": (
                    "agent_goal_with_reference"
                    if self.has_reference
                    else "agent_goal_without_reference"
                )
            },
        )

    async def _evaluate_interaction(
        self, messages: List[r.Message], reference_answer: Optional[str] = None
    ) -> Optional[float]:
        """Evaluate interaction using Ragas."""
        if not messages:
            return None

        # Create sample with only required parameters
        sample_params = {"user_input": messages}
        if reference_answer:
            sample_params["reference"] = reference_answer

        try:
            sample = MultiTurnSample(**sample_params)
            score = await self.evaluator.multi_turn_ascore(sample)
            return score
        except Exception as e:
            print(f"Error evaluating interaction: {e}")
            return None
