"""
Common metrics interfaces for all evaluation types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

class BaseMetric(ABC):
    """Base class for all evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""
        pass
    
    @abstractmethod
    async def compute(self, *args, **kwargs) -> Dict[str, Union[float, Dict[str, Union[str, float, bool, List[str]]]]]:
        """
        Compute the metric value.
        
        Returns:
            Dict containing score and optional details
        """
        pass

class MetricConfig:
    """Configuration for metrics."""
    
    def __init__(self, metric_params: Optional[Dict] = None):
        """
        Initialize metric configuration.
        
        Args:
            metric_params: Optional parameters for the metric
        """
        self.metric_params = metric_params or {} 