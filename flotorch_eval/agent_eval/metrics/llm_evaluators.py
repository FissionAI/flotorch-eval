import asyncio
from typing import Any, Dict
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.agent_eval.metrics.base import LLMBaseEval

output_structure = {
    "type": "json_schema",
    "json_schema": {
        "name": "llm_evaluation_output",
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "details": {"type": "string"},
            },
            "required": ["score", "details"],
            "additionalProperties": False,
        },
    },
}

class TrajectoryEvalWithLLM(LLMBaseEval):
    """
    Metric for evaluating an agent trajectory using an LLM.
    """

    @property
    def name(self) -> str:
        """Name of the metric."""
        return "trajectory_evaluation"

    @property
    def needs_llm(self) -> bool:
        """Indicates that this metric requires an LLM."""
        return True

    @property
    def run_async(self) -> bool:
        """Indicates that this metric should run asynchronously."""
        return True

    async def evaluate(self, trajectory: Trajectory, metric_params: Dict[str, Any]) -> MetricResult:
        """
        Evaluate the trajectory using an LLM.

        Args:
            trajectory (Trajectory): The agent trajectory to evaluate.
            metric_params (Dict[str, Any]): Additional parameters for the metric.

        Returns:
            MetricResult: The result of the evaluation.
        """
        prompt = self._prepare_prompt(
            trajectory=trajectory,
        )
        response = await self._call_llm(prompt, output_structure)
        response = self._parse_response(response)
        return response


class TrajectoryEvalWithLLMWithReference(LLMBaseEval):
    """
    Metric for evaluating an agent trajectory using an LLM, with a reference answer.
    """

    @property
    def name(self) -> str:
        """Name of the metric."""
        return "trajectory_evaluation_with_reference"

    @property
    def needs_llm(self) -> bool:
        """Indicates that this metric requires an LLM."""
        return True

    @property
    def run_async(self) -> bool:
        """Indicates that this metric should run asynchronously."""
        return True

    async def evaluate(self, trajectory: Trajectory, metric_params: Dict[str, Any]) -> MetricResult:
        """
        Evaluate the trajectory using an LLM, with a reference answer.

        Args:
            trajectory (Trajectory): The agent trajectory to evaluate.
            metric_params (Dict[str, Any]): Additional parameters for the metric, must include 'reference'.

        Returns:
            MetricResult: The result of the evaluation.
        """
        try:
            reference = metric_params["reference"]
        except Exception as e:
            return await asyncio.sleep(0, MetricResult(
                name=self.name,
                score=0.0,
                details={"error": f"Reference not found: {str(e)}"},
            ))
        prompt = self._prepare_prompt(trajectory=trajectory, reference=reference)
        response = await self._call_llm(prompt, output_structure)
        response = self._parse_response(response)
        return response

class ToolCallAccuracy(LLMBaseEval):
    """
    Metric for evaluating the accuracy of tool calls made by the agent using an LLM.
    """
    @property
    def name(self) -> str:
        """Name of the metric."""
        return "toolcall_accuracy"
    
    @property
    def needs_llm(self) -> bool:
        """Indicates that this metric requires an LLM."""
        return True

    @property
    def run_async(self) -> bool:
        """Indicates that this metric should run asynchronously."""
        return True

    async def evaluate(self, trajectory: Trajectory, metric_params: Dict[str, Any]) -> MetricResult:
        """
        Evaluate the tool call accuracy of the trajectory using an LLM.

        Args:
            trajectory (Trajectory): The agent trajectory to evaluate.
            metric_params (Dict[str, Any]): Additional parameters for the metric.

        Returns:
            MetricResult: The result of the evaluation.
        """
        prompt = self._prepare_prompt(
            trajectory=trajectory,
        )
        response = await self._call_llm(prompt, output_structure)
        response = self._parse_response(response)
        return response

class AgentGoalAccuracy(LLMBaseEval):
    """
    Metric for evaluating the agent's goal achievement accuracy using an LLM.
    """
    @property
    def name(self) -> str:
        """Name of the metric."""
        return "agent_goal_accuracy"
    
    @property
    def needs_llm(self) -> bool:
        """Indicates that this metric requires an LLM."""
        return True
    
    @property
    def run_async(self) -> bool:
        """Indicates that this metric should run asynchronously."""
        return True

    async def evaluate(self, trajectory: Trajectory, metric_params: Dict[str, Any]) -> MetricResult:
        """
        Evaluate the agent's goal achievement accuracy using an LLM.

        Args:
            trajectory (Trajectory): The agent trajectory to evaluate.
            metric_params (Dict[str, Any]): Additional parameters for the metric.

        Returns:
            MetricResult: The result of the evaluation.
        """
        prompt = self._prepare_prompt(
            trajectory=trajectory,
        )
        response = await self._call_llm(prompt, output_structure)
        response = self._parse_response(response)
        return response
