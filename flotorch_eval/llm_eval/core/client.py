"""
LLM Evaluation Client Module.

This module provides the main client interface for evaluating LLM-based metrics
using different evaluation engines (Ragas, DeepEval, Gateway).
"""
from typing import List, Optional, Dict, Union, Any
from flotorch_eval.llm_eval.metrics.ragas_metrics.metric_keys import MetricKey
from flotorch_eval.llm_eval.core.schemas import EvaluationItem
from flotorch_eval.llm_eval.core.ragas_evaluator import RagasEvaluator
from flotorch_eval.llm_eval.metrics.ragas_metrics.ragas_metrics import RagasEvaluationMetrics
from flotorch_eval.llm_eval.core.gateway_evaluator import GatewayEvaluator
from flotorch_eval.llm_eval.metrics.gateway_metrics.gateway_metrics import GatewayMetrics
from flotorch_eval.llm_eval.core.deepeval_evaluator import DeepEvalEvaluator
from flotorch_eval.llm_eval.metrics.deepeval_metrics.deepeval_metrics import (
    DeepEvalEvaluationMetrics
)

class LLMEvaluator:
    """
    Client for evaluating LLM-based metrics.
    """
    _ENGINES = {
        'ragas': {
            'metrics_class': RagasEvaluationMetrics,
            'priority': 1,
        },
        'deepeval': {
            'metrics_class': DeepEvalEvaluationMetrics,
            'priority': 2,
        }
    }

    def __init__(
        self,
        api_key: str,
        base_url: str,
        embedding_model: str,
        inferencer_model: str,
        evaluation_engine='auto',
        metrics: Optional[List[MetricKey]] = None,
        metric_configs: Optional[
            Dict[Union[str, MetricKey], Dict[str, Dict[str, str]]]
        ] = None
    ):
        """
        Initialize the LLMEvaluator.

        Args:
            api_key (str): API key for authentication.
            base_url (str): Base URL for the evaluation service.
            evaluation_engine (str): The evaluation engine to use.
                Options: 'auto' (default), 'ragas', 'deepeval'
                - 'auto': Automatically routes metrics to the appropriate engine.
                  Ragas has priority for overlapping metrics.
                - Specific engine: Use only that evaluation engine
            embedding_model (str): The embedding model to use.
            inferencer_model (str): The inferencer model to use.
            metrics (Optional[List[MetricKey]]): Default metrics to use for evaluation.
                If None and engine is 'auto', all metrics from all engines will be used.
                If None with specific engine, all metrics from that engine will be used.
            metric_configs (Optional[Dict]): Configuration for metrics that require
                additional parameters (e.g., AspectCritic). Metrics requiring configs
                will be skipped if not provided.
                Example:
                {
                    MetricKey.ASPECT_CRITIC: {
                        'maliciousness': {
                            'name': 'maliciousness',
                            'definition': 'Is the response harmful?'
                        }
                    }
                }

        Raises:
            ValueError: If the evaluation engine is not valid.
        """
        if evaluation_engine != 'auto' and evaluation_engine not in self._ENGINES:
            available = ', '.join(self._ENGINES.keys())
            raise ValueError(
                f"Evaluation engine '{evaluation_engine}' not supported. "
                f"Available: {available}, or 'auto'"
            )

        self.inferencer_model = inferencer_model
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.base_url = base_url
        self.evaluation_engine = evaluation_engine
        self.metrics = metrics
        self.metric_configs = metric_configs

    def set_metrics(self, metrics: List[MetricKey]) -> None:
        """
        Update the default metrics to use for evaluation.

        Args:
            metrics (List[MetricKey]): The new default metrics.
        """
        self.metrics = metrics

    def set_metric_configs(
        self,
        metric_configs: Dict[Union[str, MetricKey], Dict[str, Dict[str, str]]]
    ) -> None:
        """
        Update the metric configurations.

        Args:
            metric_configs (Dict): Configuration for metrics that require
                additional parameters.
        """
        self.metric_configs = metric_configs

    def _get_engine_for_metric(self, metric: MetricKey) -> str:
        """
        Determine which engine supports a metric (respects priority).
        
        Args:
            metric: The metric to check
            
        Returns:
            Engine name
            
        Raises:
            ValueError: If metric not supported by any engine
        """
        # Find all engines that support this metric, sorted by priority
        supporting_engines = []
        for engine_name, engine_info in self._ENGINES.items():
            if metric in engine_info['metrics_class'].registered_metrics():
                supporting_engines.append((engine_info['priority'], engine_name))

        if not supporting_engines:
            raise ValueError(f"Metric '{metric}' not supported by any engine")

        # Return highest priority engine
        supporting_engines.sort()
        return supporting_engines[0][1]

    def _split_metrics_by_engine(self, metrics: List[MetricKey]) -> Dict[str, List[MetricKey]]:
        """
        Split metrics by engine based on priority.
        
        Args:
            metrics: List of metrics to split
            
        Returns:
            Dict mapping engine names to their metrics
        """
        engine_metrics = {name: [] for name in self._ENGINES}

        for metric in metrics:
            engine = self._get_engine_for_metric(metric)
            engine_metrics[engine].append(metric)

        # Remove empty entries
        return {k: v for k, v in engine_metrics.items() if v}

    def _get_all_metrics(self) -> List[MetricKey]:
        """
        Get all unique metrics from all engines (respects priority).
        
        Returns:
            List of all unique metrics
        """
        # Sort engines by priority
        sorted_engines = sorted(
            self._ENGINES.items(),
            key=lambda x: x[1]['priority']
        )

        seen = set()
        all_metrics = []

        for _, engine_info in sorted_engines:
            for metric in engine_info['metrics_class'].registered_metrics():
                if metric not in seen:
                    seen.add(metric)
                    all_metrics.append(metric)

        return all_metrics

    def _create_evaluator(self, engine_name: str):
        """Create evaluator instance for the given engine."""
        if engine_name == 'ragas':
            return RagasEvaluator(
                base_url=self.base_url,
                api_key=self.api_key,
                embedding_model=self.embedding_model,
                inferencer_model=self.inferencer_model,
                metric_args=self.metric_configs
            )
        elif engine_name == 'deepeval':
            return DeepEvalEvaluator(
                evaluator_llm=self.inferencer_model,
                api_key=self.api_key,
                base_url=self.base_url,
                metric_args=self.metric_configs
            )
        else:
            raise ValueError(f"Unknown engine: {engine_name}")

    def evaluate(
        self,
        data: List[EvaluationItem],
        metrics: Optional[List[MetricKey]]=None
    ) -> Dict[str, Any]:
        """
        Evaluate the data using the evaluation engine.

        Args:
            data (List[EvaluationItem]): The data to evaluate.
            metrics (Optional[List[MetricKey]]): The metrics to use for evaluation.
                If None, uses the default metrics set during initialization.
                If both are None:
                  - In 'auto' mode: all metrics from all engines will be used
                  - In specific engine mode: all metrics from that engine will be used

        Returns:
            Dict[str, Any]: The evaluation results. If metadata is present in 
                EvaluationItems, gateway_metrics will be automatically included.
        """
        if self.inferencer_model is None or self.embedding_model is None:
            raise ValueError("LLM and embedding model must be set to use evaluation")

        # Use provided metrics, fall back to instance default
        metrics_to_use = metrics if metrics is not None else self.metrics

        # Handle auto mode
        if self.evaluation_engine == 'auto':
            # If no metrics specified, use all metrics from all engines
            if metrics_to_use is None:
                metrics_to_use = self._get_all_metrics()

            # Split metrics by engine based on priority
            metrics_by_engine = self._split_metrics_by_engine(metrics_to_use)

            results = {}

            # Evaluate with each engine that has metrics assigned
            for engine_name, engine_metrics in metrics_by_engine.items():
                evaluator = self._create_evaluator(engine_name)
                engine_results = evaluator.evaluate(data, engine_metrics)
                results.update(engine_results)

        else:
            # Use specific engine
            evaluator = self._create_evaluator(self.evaluation_engine)
            results = evaluator.evaluate(data, metrics_to_use)

        # Automatically include gateway metrics if metadata is present
        gateway_metrics = None
        if GatewayMetrics.has_metadata(data):
            gateway_evaluator = GatewayEvaluator()
            gateway_results = gateway_evaluator.evaluate(data)
            gateway_metrics = gateway_results.get('gateway_metrics', {})

        combined_results = {
            "evaluation_metrics": results,          # all model evaluation scores
        }
        if gateway_metrics:
            combined_results['gateway_metrics'] = gateway_metrics

        return combined_results
