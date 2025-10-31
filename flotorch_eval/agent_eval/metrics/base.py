import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Awaitable, TYPE_CHECKING, Union
from pydantic import BaseModel, Field
from flotorch.sdk.llm import FlotorchLLM
from flotorch.sdk.utils.llm_utils import LLMResponse
from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.agent_eval.metrics.prompt_manager import PromptManager

if TYPE_CHECKING:
    from flotorch_eval.agent_eval.core.client import AgentEvaluator


class MetricConfig(BaseModel):
    """Base configuration for metrics."""

    metric_params: Dict[str, Any] = Field(
        default_factory=dict, description="Metric-specific parameters"
    )


class LLMBaseEval(ABC):
    """
    Abstract base class for all metric evaluators, including those that use LLMs.

    This class defines the interface and common utilities for metric evaluation,
    including prompt preparation, LLM invocation, and response parsing.

    Attributes:
        llm (Optional[str]): The LLM model identifier to use for evaluation.
        config (Optional[MetricConfig]): Configuration object containing metric parameters.
        client (Optional[AgentEvaluator]): Reference to the evaluation client.
        llm_evaluator (Optional[FlotorchLLM]): The LLM evaluator instance.
    """

    def __init__(
        self, llm: Optional[str] = None, config: Optional[MetricConfig] = None
    ):
        """
        Initialize the metric evaluator.

        Args:
            llm (Optional[str]): The LLM model identifier to use.
            config (Optional[MetricConfig]): Configuration for the metric.
        """
        self.llm = llm
        self.config = config
        self.client = None
        self.llm_evaluator = None

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the metric.

        Returns:
            str: The name of the metric.
        """
        pass

    @property
    def needs_llm(self) -> bool:
        """
        Indicates whether this metric requires an LLM.

        Returns:
            bool: True if the metric requires an LLM, False otherwise.
        """
        return False

    @property
    def run_async(self) -> bool:
        """
        Indicates whether this metric should run asynchronously.

        Returns:
            bool: True if the metric should run asynchronously, False otherwise.
        """
        return False

    @abstractmethod
    def evaluate(
        self, trajectory: Trajectory, metric_params: Dict[str, Any]
    ) -> Union[MetricResult, Awaitable[MetricResult]]:
        """
        Evaluate the metric on the given trajectory.

        Args:
            trajectory (Trajectory): The agent trajectory to evaluate.
            metric_params (Dict[str, Any]): Additional parameters for the metric.

        Returns:
            MetricResult: The result of the evaluation.
        """
        pass

    def prepare_llm(self, client: "AgentEvaluator") -> None:
        """
        Prepare the LLM evaluator using the provided client.

        Args:
            client (AgentEvaluator): The evaluation client.
        """
        if self.llm is not None:
            llm_to_use = self.llm
        else:
            if client.default_evaluator is not None:
                llm_to_use = client.default_evaluator
            else:
                raise ValueError(
                    "No evaluator set. Initialize the client with a default evaluator model "
                    "via constructor or set_default_evaluator, or provide a model to the metric."
                )

        self.llm_evaluator = FlotorchLLM(
            model_id=llm_to_use, api_key=client.api_key, base_url=client.base_url
        )

    def _parse_response(self, response: LLMResponse) -> MetricResult:
        """
        Parse the response from the LLM and convert it to a MetricResult.

        Args:
            response (LLMResponse): The response object from the LLM.

        Returns:
            MetricResult: The parsed metric result.
        """
        raw_content = response.content
        parsed_content = json.loads(raw_content)
        result = MetricResult(
            name=self.name,
            score=parsed_content.get("score", 0.0),
            details={"details": parsed_content.get("details", "No details provided")},
        )
        return result

    def _prepare_prompt(self, **kwargs: Any) -> str:
        """
        Prepare the prompt for the LLM using the PromptManager.

        Args:
            **kwargs (Any): Keyword arguments to format the prompt.

        Returns:
            str: The formatted prompt string.

        Raises:
            ValueError: If prompt formatting fails.
        """
        prompt_manager = PromptManager()
        prompt = prompt_manager.get_prompt(self.name)
        try:
            prompt = prompt.format(**kwargs)
        except Exception as e:
            raise ValueError(f"Error loading prompt: {e}") from e
        return prompt

    async def _call_llm(
        self, prompt: str, response_format: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Call the LLM asynchronously with the given prompt.

        Args:
            prompt (str): The prompt to send to the LLM.
            response_format (Optional[Dict[str, Any]]): The expected response format.

        Returns:
            LLMResponse: The response from the LLM.
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_evaluator.ainvoke(
            messages, response_format=response_format
        )
        return response
