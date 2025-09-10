import asyncio
from typing import Dict, Any, List
from flotorch_eval.agent_eval.metrics.base import LLMBaseEval
from flotorch_eval.agent_eval.core.converter import TraceConverter
from flotorch_eval.agent_eval.core.schemas import EvaluationResult


class FlotorchEvalClient():
    """
    Client for evaluating agent trajectories using a set of metrics.
    Handles both synchronous and asynchronous (LLM-based) metrics.
    """

    def __init__(self, api_key, base_url, default_evaluator=None):
        """
        Initialize the FlotorchEvalClient.

        Args:
            api_key (str): API key for authentication.
            base_url (str): Base URL for the evaluation service.
            default_evaluator (str, optional): Default evaluator model or identifier.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_evaluator = default_evaluator

    async def evaluate(self, trace: Dict[str, Any], metrics: List[LLMBaseEval]):
        """
        Evaluate a trace using the provided metrics.

        Args:
            trace (Dict[str, Any]): The trace data (list of spans or similar).
            metrics (List[LLMBaseEval]): List of metric evaluators.

        Returns:
            EvaluationResult: The result of the evaluation.

        Raises:
            ValueError: If no metrics or trace are provided.
            RuntimeError: If evaluation fails.
        """
        try:
            if not metrics:
                raise ValueError("No metrics provided for evaluation")
            if not trace:
                raise ValueError("No spans provided for evaluation")

            try:
                trajectory = self._trace_to_trajectory(trace)
                results = await self.run_evaluation(trajectory, metrics)
            except Exception as e:
                print(f"Evaluation failed: {str(e)}")
                raise RuntimeError(f"Evaluation process failed: {str(e)}")
            return results

        except Exception as e:
            print(f"Evaluation failed with error: {str(e)}")
            raise

    def _trace_to_trajectory(self, trace):
        """
        Convert a trace (list of spans) to a Trajectory object.

        Args:
            trace (Any): The trace data.

        Returns:
            Trajectory: The converted trajectory object.
        """
        converter = TraceConverter()
        trajectory = converter.from_spans(trace)
        return trajectory

    async def run_evaluation(self, trajectory, metrics):
        """
        Run all provided metrics on the given trajectory.

        Args:
            trajectory: The trajectory object to evaluate.
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
                async_tasks.append(metric.evaluate(trajectory, metric_params))
            else:
                metric_params = metric.config.metric_params if metric.config else {}
                result = metric.evaluate(trajectory, metric_params)
                sync_results.append(result)

        # Run async metrics concurrently
        async_results = await asyncio.gather(*async_tasks, return_exceptions=False)

        all_scores = sync_results + list(async_results)
        return EvaluationResult(trajectory_id=trajectory.trace_id, scores=all_scores)
        