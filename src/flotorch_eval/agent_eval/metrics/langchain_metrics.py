"""
LangChain-based evaluation metrics.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Union

from agentevals.trajectory.llm import (
    TRAJECTORY_ACCURACY_PROMPT,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    create_trajectory_llm_as_judge,
)
from agentevals.trajectory.match import create_trajectory_match_evaluator
from langchain.chat_models.base import BaseChatModel
from langchain.evaluation import load_evaluator
from langchain_core.language_models.chat_models import BaseChatModel

from flotorch_eval.agent_eval.metrics.base import BaseMetric, MetricConfig
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory

# Define valid match modes
TrajectoryMatchMode = Literal["strict", "unordered", "subset", "superset"]
ToolArgsMatchMode = Literal["exact", "ignore", "subset", "superset"]


class LangChainAgentsEvalMixin:
    """Evaluates agent responses based on custom criteria using LangChain Agent Evals."""

    def _convert_to_standard_format(
        self, trajectory: Trajectory
    ) -> List[Dict[str, Any]]:
        """Convert trajectory to standard format for evaluation."""
        outputs = []
        for msg in trajectory.messages:
            output = {"role": msg.role, "content": msg.content}

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                output["tool_calls"] = [
                    {
                        "function": {
                                                "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                        }
                    }
                    for tool_call in msg.tool_calls
                ]

            outputs.append(output)
        return outputs


class TrajectoryEvalWithoutLLMMetric(BaseMetric, LangChainAgentsEvalMixin):
    """Evaluates the agent's trajectory including tool call accuracy."""

    @property
    def name(self) -> str:
        return "trajectory_eval_without_llm"

    def _setup(self) -> None:
        """Setup the trajectory evaluator."""
        metric_params = self.config.metric_params if self.config else {}

        # Get match modes with validation
        trajectory_match_mode = metric_params.get("trajectory_match_mode", "strict")
        tool_args_match_mode = metric_params.get("tool_args_match_mode", "exact")

        # Validate trajectory_match_mode
        if trajectory_match_mode not in ("strict", "unordered", "subset", "superset"):
            raise ValueError(
                "trajectory_match_mode must be one of: strict, unordered, subset, superset. "
                f"Got: {trajectory_match_mode}"
            )

        # Validate tool_args_match_mode
        if tool_args_match_mode not in ("exact", "ignore", "subset", "superset"):
            raise ValueError(
                "tool_args_match_mode must be one of: exact, ignore, subset, superset. "
                f"Got: {tool_args_match_mode}"
            )

        self.trajectory_match_mode = trajectory_match_mode
        self.tool_args_match_mode = tool_args_match_mode

        # Set up trajectory match evaluator
        self.evaluator = create_trajectory_match_evaluator(
            trajectory_match_mode=self.trajectory_match_mode,
            tool_args_match_mode=self.tool_args_match_mode,
        )

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Compute trajectory evaluation score including tool call accuracy.

        Args:
            trajectory: The trajectory to evaluate

        Returns:
            MetricResult with evaluation scores and details from LLM evaluation.
            Score is 1.0 for True and 0.0 for False.
        """
        # Get reference trajectory if available
        if self.config and self.config.metric_params:
            reference_outputs = self.config.metric_params.get("reference_outputs")

        if not reference_outputs:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "Reference trajectory required for evaluation"},
            )

        # Convert trajectories to standard format
        outputs = self._convert_to_standard_format(trajectory)

        # Evaluate using trajectory match evaluator
        try:
            result = self.evaluator(
                outputs=outputs, reference_outputs=reference_outputs
            )

            # Extract score (convert boolean to float) and details from result
            score = 1.0 if result.get("score", False) else 0.0
            details = {
                "trajectory_match_mode": self.trajectory_match_mode,
                "tool_args_match_mode": self.tool_args_match_mode,
                "evaluation_details": result,
            }

            return MetricResult(name=self.name, score=score, details=details)

        except Exception as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": f"Failed to evaluate trajectory: {str(e)}"},
            )


class TrajectoryEvalWithLLMMetric(BaseMetric, LangChainAgentsEvalMixin):
    """Evaluates the agent's trajectory using LLM as judge, optionally comparing against reference outputs."""

    requires_llm = True

    @property
    def name(self) -> str:
        return "trajectory_eval_with_llm"

    def _setup(self) -> None:
        """Setup the trajectory evaluator with LLM as judge."""
        metric_params = self.config.metric_params if self.config else {}

        # Get model identifier if provided in config
        model_identifier = metric_params.get("model")

        # Determine which prompt to use based on whether reference outputs are provided
        has_reference = bool(metric_params.get("reference_outputs"))
        prompt = (
            TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
            if has_reference
            else TRAJECTORY_ACCURACY_PROMPT
        )

        # Create LLM-based trajectory evaluator
        self.evaluator = create_trajectory_llm_as_judge(
            prompt=prompt,
            judge=self.llm,  # Can be OpenAI client, Bedrock client, or LangChain model
            model=model_identifier,  # Optional model identifier if needed
        )

    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Compute trajectory evaluation score using LLM as judge.

        Args:
            trajectory: The trajectory to evaluate

        Returns:
            MetricResult with evaluation scores and details from LLM evaluation.
            Score is 1.0 for True and 0.0 for False.
        """
        # Get reference outputs if available
        reference_outputs = None
        if self.config and self.config.metric_params:
            reference_outputs = self.config.metric_params.get("reference_outputs")

        # Convert trajectory to standard format
        outputs = self._convert_to_standard_format(trajectory)

        try:
            # Evaluate trajectory with or without reference
            if reference_outputs:
                result = self.evaluator(
                    outputs=outputs, reference_outputs=reference_outputs
                )
            else:
                result = self.evaluator(outputs=outputs)

            # Extract score (convert boolean to float) and details from result
            score = 1.0 if result.get("score", False) else 0.0

            # Extract only simple types for details
            details = {
                "comment": str(result.get("comment", "")),
                "has_reference": bool(reference_outputs is not None),
                "raw_score": bool(result.get("score", False)),
            }

            return MetricResult(name=self.name, score=score, details=details)

        except Exception as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "error": str(e),
                    "has_reference": bool(reference_outputs is not None),
                },
            )
