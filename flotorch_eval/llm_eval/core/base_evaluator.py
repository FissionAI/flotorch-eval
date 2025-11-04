"""
Base Evaluator Module.

This module defines the abstract base class for evaluation implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from flotorch_eval.llm_eval.core.schemas import EvaluationItem
from flotorch_eval.llm_eval.metrics.ragas_metrics.metric_keys import MetricKey

class BaseEvaluator(ABC):
    """
    Abstract base class for evaluation modules.
    """

    @abstractmethod
    def evaluate(
        self,
        data: List[EvaluationItem],
        metrics: Optional[List[MetricKey]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model output against the expected answers.

        Args:
            data (List[EvaluationItem]): List of evaluation inputs.

        Returns:
            Dict[str, Any]: Dictionary of evaluation results.
        """
        pass
