"""
Client for evaluating agent trajectories using a set of metrics.
Handles both synchronous and asynchronous (LLM-based) metrics.
"""

import asyncio
from typing import Dict, Any, List, Optional
import requests
from flotorch_eval.agent_eval.metrics.base import LLMBaseEval
from flotorch_eval.agent_eval.core.converter import TraceConverter
from flotorch_eval.agent_eval.core.schemas import EvaluationResult, Trajectory
from flotorch_eval.agent_eval.metrics.llm_evaluators import (
    TrajectoryEvalWithLLM,
    TrajectoryEvalWithLLMWithReference,
    ToolCallAccuracy,
    AgentGoalAccuracy,
)
from flotorch_eval.agent_eval.metrics.usage_metrics import UsageMetric
from flotorch_eval.agent_eval.metrics.latency_metrics import LatencyMetric
from flotorch_eval.agent_eval.metrics.base import MetricConfig


class AgentEvaluator:
    """
    Client for evaluating agent trajectories using a set of metrics.
    Handles both synchronous and asynchronous (LLM-based) metrics.
    """

    def __init__(self, api_key, base_url, default_evaluator=None) -> None:
        """
        Initialize the AgentEvaluator.

        Args:
            api_key (str): API key for authentication.
            base_url (str): Base URL for the evaluation service.
            default_evaluator (str, optional): Default evaluator model or identifier.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_evaluator = default_evaluator

    def set_default_evaluator(self, default_evaluator: str) -> None:
        """
        Set the default evaluator model for the client.
        """
        self.default_evaluator = default_evaluator

    def fetch_traces(self, trace_id: str) -> Dict[str, Any]:
        """
        Fetches the traces from the Flotorch API for a given trace id.

        Args:
            trace_id: The ID of the trace to fetch.

        Returns:
            The traces from the Flotorch API.
        """
        api_key = self.api_key
        base_url = self.base_url

        if not api_key or not base_url:
            raise ValueError(
                "Flotorch client must be initialized with an API key and base URL"
            )

        if not trace_id:
            raise ValueError("Trace ID must be provided to fetch traces")

        url = f"{base_url}/v1/traces/{trace_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=10).json()
        trace = response.get("trace") if response.get("trace") else None
        return trace

    async def evaluate(
        self,
        trace: Dict[str, Any],
        metrics: Optional[List[LLMBaseEval]] = None,
        reference: Dict[str, Any] = None,
        reference_trace_id: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a trace using the provided metrics.

        Args:
            trace (Dict[str, Any]): The trace data (list of spans or similar).
            metrics (List[LLMBaseEval]): List of metric evaluators.
            reference (Dict[str, Any]): Reference trajectory.
        Returns:
            EvaluationResult: The result of the evaluation.

        Raises:
            ValueError: If no metrics or trace are provided.
            RuntimeError: If evaluation fails.
        """
        # Check for conflicting arguments
        if reference and reference_trace_id:
            raise ValueError("Provide either 'reference' or 'reference_trace_id', not both.")

        # Create reference from trace id if one is provided
        if reference_trace_id:
            print(f"Fetching reference trace with ID: {reference_trace_id}")
            try:
                reference_trace_data = self.fetch_traces(trace_id=reference_trace_id)
                if not reference_trace_data:
                    raise ValueError(f"Could not fetch or find trace for reference ID: {reference_trace_id}")
                
                converter = TraceConverter()
                reference_obj = converter.to_reference(reference_trace_data)
                reference = reference_obj.model_dump()
                print("Successfully converted trace to reference format.")
            except Exception as e:
                print(f"Failed to create reference from trace ID '{reference_trace_id}': {e}")
                raise
        try:
            if metrics is None:
                if self.default_evaluator is not None:
                    metrics = (
                        [  # TODO Make this cleaner with a method to get all metrics
                            TrajectoryEvalWithLLM(),
                            ToolCallAccuracy(),
                            AgentGoalAccuracy(),
                            UsageMetric(),
                            LatencyMetric(),
                        ]
                    )
                    if reference:
                        metrics.append(
                            TrajectoryEvalWithLLMWithReference(
                                config=MetricConfig(
                                    metric_params={"reference": reference}
                                )
                            )
                        )
                else:
                    raise ValueError(
                        "Default evaluator is not set. Initialize the client with a "
                        "default evaluator/use 'set_default_evaluator' method to set an evaluator"
                    )
            if not trace:
                raise ValueError("No spans provided for evaluation")

            try:
                if metrics is not None and not all(
                    isinstance(m, LLMBaseEval) for m in metrics
                ):
                    raise TypeError("metrics must be a list of LLMBaseEval instances")

                trajectory = self._trace_to_trajectory(trace)
                results = await self._run_evaluation(trajectory, metrics)
            except Exception as e:
                print(f"Evaluation failed: {str(e)}")
                raise RuntimeError(f"Evaluation process failed: {str(e)}") from e
            return results

        except Exception as e:
            print(f"Evaluation failed with error: {str(e)}")
            raise

    def _trace_to_trajectory(self, trace: Dict[str, Any]) -> Trajectory:
        """
        Convert a trace (list of spans) to a Trajectory object.

        Args:
            trace (Dict[str, Any]): The trace data.

        Returns:
            Trajectory: The converted trajectory object.
        """
        converter = TraceConverter()
        trajectory = converter.from_spans(trace)
        return trajectory

    async def _run_evaluation(
        self, trajectory: Trajectory, metrics: List[LLMBaseEval]
    ) -> EvaluationResult:
        """
        Run all provided metrics on the given trajectory.

        Args:
            trajectory (Trajectory): The trajectory object to evaluate.
            metrics (List[LLMBaseEval]): List of metric evaluators.

        Returns:
            EvaluationResult: The result containing all metric scores.
        """
        async_tasks = []
        sync_results = []

        for metric in metrics:
            if metric.needs_llm:
                metric.prepare_llm(self)

            metric_params = metric.config.metric_params if metric.config else {}
            # evalaute can be sync or async;
            result_or_task = metric.evaluate(trajectory, metric_params)

            if metric.run_async:
                async_tasks.append(result_or_task)
            else:
                sync_results.append(result_or_task)

        async_results = []
        if async_tasks:
            async_results = await asyncio.gather(*async_tasks, return_exceptions=False)

        all_scores = sync_results + list(async_results)
        return EvaluationResult(trajectory_id=trajectory.trace_id, scores=all_scores)
