"""
FlotorchEval - A comprehensive evaluation framework for AI systems.
"""

__version__ = "1.0.1"

from flotorch_eval.common.metrics import BaseMetric, MetricConfig
from flotorch_eval.common.utils import convert_attributes

__all__ = [
    "BaseMetric",
    "MetricConfig",
    "convert_attributes",
]
