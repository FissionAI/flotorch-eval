"""
Base classes and interfaces for evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from flotorch_eval.agent_eval.core.schemas import Trajectory, MetricResult
from flotorch_eval.common.metrics import BaseMetric as CommonBaseMetric
from flotorch_eval.common.metrics import MetricConfig


class MetricConfig(BaseModel):
    """Base configuration for metrics."""

    metric_params: Dict[str, Any] = Field(
        default_factory=dict, description="Metric-specific parameters"
    )


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""

    requires_llm: bool = False

    def __init__(
        self, llm: Optional[Any] = None, config: Optional[MetricConfig] = None
    ):
        """
        Initialize the metric.

        Args:
            llm: Language model to use for evaluation (if required)
            config: Configuration for the metric including metric-specific parameters
        """
        if self.requires_llm and llm is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an LLM for evaluation"
            )

        self.llm = llm
        self.config = config or MetricConfig()
        self._setup()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""
        pass

    @abstractmethod
    def _setup(self) -> None:
        """
        Setup the metric with necessary components.
        This method should be called in __init__ and when config changes.
        """
        pass

    @abstractmethod
    async def compute(self, trajectory: Trajectory) -> MetricResult:
        """
        Compute the metric for a given trajectory.

        Args:
            trajectory: The trajectory to evaluate

        Returns:
            MetricResult containing the score and optional details
        """
        pass

    def update_config(self, config: MetricConfig) -> None:
        """
        Update the metric configuration.

        Args:
            config: New configuration for the metric
        """
        self.config = config
        self._setup()

    def update_llm(self, llm: Any) -> None:
        """
        Update the LLM used by the metric.

        Args:
            llm: New language model to use
        """
        if self.requires_llm and llm is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an LLM for evaluation"
            )

        self.llm = llm
        self._setup()
