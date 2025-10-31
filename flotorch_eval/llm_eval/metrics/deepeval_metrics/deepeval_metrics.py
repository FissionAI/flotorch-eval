"""
DeepEval Metrics Module.

This module provides the registry and initialization logic for DeepEval evaluation metrics.
"""
from typing import Dict, Any
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
)
from flotorch_eval.llm_eval.metrics.base_metrics import BaseEvaluationMetric
from flotorch_eval.llm_eval.metrics.metric_keys import MetricKey

class DeepEvalEvaluationMetrics(BaseEvaluationMetric):
    """
    Registry of DeepEval metric classes and their initialized instances.
    """
    _registry: Dict[str, Dict[str, Any]] = {
        MetricKey.FAITHFULNESS: {
            "class": FaithfulnessMetric,
            "default_args": {"threshold": 0.7, "truths_extraction_limit": 30}
        },
        MetricKey.CONTEXT_RELEVANCY: {
            "class": ContextualRelevancyMetric,
            "default_args": {"threshold": 0.7}
        },
        MetricKey.CONTEXT_PRECISION: {
            "class": ContextualPrecisionMetric,
            "default_args": {"threshold": 0.7}
        },
        MetricKey.CONTEXT_RECALL: {
            "class": ContextualRecallMetric,
            "default_args": {"threshold": 0.7}
        },
        MetricKey.ANSWER_RELEVANCE: {
            "class": AnswerRelevancyMetric,
            "default_args": {"threshold": 0.7}
        },
        MetricKey.HALLUCINATION: {
            "class": HallucinationMetric,
            "default_args": {"threshold": 0.5}
        }
    }

    _initialized_metrics: Dict[str, object] = {}

    @classmethod
    def available_metrics(cls) -> list[str]:
        return list(cls._initialized_metrics.keys())

    @classmethod
    def registered_metrics(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def initialize_metrics(cls, *args, **kwargs):
        """
        Initializes metric instances and stores them in an internal dictionary.

        This method iterates over the registered metrics in `cls._registry`,
        initializes each metric,
        and stores it in `cls._initialized_metrics`.  
        You can optionally override default arguments for each metric using `metric_args`.

        Args:
            llm: A language model wrapper instance (used as `model` argument for each metric class).
            metric_args (Optional): A dictionary providing argument overrides for each metric.
                The structure is:

                {
                    "metric_name": {
                        "arg1": value1,
                        "arg2": value2,
                        ...
                    },
                    ...
                }

                - The outer key corresponds to a metric name (as used in `cls._registry`).
                - The inner dictionary provides argument overrides that will replace defaults.

        Example Usage:
            metric_args={
                "faithfulness": {
                "threshold": 0.8
                }                    
            }
            
        """
        # Extract parameters from args/kwargs
        llm = args[0] if args else kwargs.get('llm')
        metric_args = args[1] if len(args) > 1 else kwargs.get('metric_args', None)

        cls._initialized_metrics = {}
        metric_args = metric_args or {}

        for name, config in cls._registry.items():
            metric_config_args = config["default_args"].copy()
            metric_config_args.update(metric_args.get(name, {}))  # override defaults
            cls._initialized_metrics[name] = config["class"](model=llm, **metric_config_args)

    @classmethod
    def get_metric(cls, key: str) -> Dict[str, object]:
        """
        Retrieves an initialized metric instance by key.

        Args:
            key (str): The name of the metric to retrieve (must match a key in `cls._registry`).

        Returns:
            object: The initialized metric instance.

        Raises:
            ValueError: If the requested metric has not been initialized
            (i.e. not present in `cls._initialized_metrics`).

        Example Usage:
            faithfulness_metric = get_metric("faithfulness")
        """
        if key not in cls._initialized_metrics:
            raise ValueError(
    f"Metric '{key}' is not initialized for the 'deepeval' evaluation engine. "
    "Make sure to call `initialize_metrics()` before evaluation, "
    "or verify that you're using the correct evaluation engine."
)
        return cls._initialized_metrics[key]
