"""
Base Metrics Module.

This module defines the abstract base class for evaluation metric implementations.
"""
from abc import ABC, abstractmethod
from typing import List

class BaseEvaluationMetric(ABC):
    """
    Abstract base class to define metric registries.
    """
    @classmethod
    @abstractmethod
    def available_metrics(cls) -> List[str]:
        """
        Returns a list of available metrics.
        """
        pass

    @classmethod
    @abstractmethod
    def registered_metrics(cls) -> List[str]:
        """
        Returns a list of all registered metric keys.
        """
        pass

    @classmethod
    @abstractmethod
    def get_metric(cls, key: str):
        """
        Returns the metric associated with the key.
        """
        pass

    @classmethod
    @abstractmethod
    def initialize_metrics(cls, *args, **kwargs):
        """
        Initializes the metrics.
        """
        pass
