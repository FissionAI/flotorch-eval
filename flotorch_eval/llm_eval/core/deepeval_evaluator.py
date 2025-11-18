"""
DeepEval Evaluator Module.

This module implements the DeepEval evaluator for LLM-based metrics.
"""

import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Type, Union
from pydantic import BaseModel
from tenacity import (
    retry,
    wait_exponential_jitter,
    retry_if_exception_type,
    stop_after_attempt,
)

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.evaluate import ErrorConfig

from flotorch.sdk.llm import FlotorchLLM
from flotorch_eval.llm_eval.core.base_evaluator import BaseEvaluator
from flotorch_eval.llm_eval.core.schemas import EvaluationItem
from flotorch_eval.llm_eval.metrics.deepeval_metrics.deepeval_metrics import (
    DeepEvalEvaluationMetrics,
)
from flotorch_eval.llm_eval.metrics.metric_keys import MetricKey


class DeepEvalEvaluator(BaseEvaluator):
    """
    Evaluator that uses DeepEval metrics to evaluate LLM outputs with optional custom metrics
    and support for asynchronous evaluation.

    Initializes with an LLM inferencer and
    allows configuration of custom metrics, asynchronous execution,
    concurrency limits, and optional metric-specific arguments.
    Args:
        evaluator_llm : The LLM inferencer used for evaluation.
        api_key : The API key for the FlotorchLLM.
        base_url : The base URL for the FlotorchLLM.
        custom_metrics :A list of additional metric instances to include in
        evaluation beyond the default DeepEval metrics registry.
        async_run :Whether to run evaluation asynchronously.
        If True, evaluation can run concurrently up to `max_concurrent` tasks.
        max_concurrent : Maximum number of concurrent asynchronous evaluation tasks to run.
        metric_args :Optional dictionary specifying per-metric configuration arguments.
    Example:
        metric_args = {
            "contextual_recall": {
                "threshold": 0.6
            },
            "hallucination": {
                "threshold": 0.4
            }
        }
    """

    def __init__(
        self,
        evaluator_llm: str,
        api_key: str,
        base_url: str,
        custom_metrics: Optional[List[Any]] = None,
        async_run: bool = True,
        max_concurrent: int = 5,
        metric_args: Optional[
            Dict[Union[str, MetricKey], Dict[str, Union[str, float, int]]]
        ] = None,
    ):
        class FloTorchLLMWrapper(DeepEvalBaseLLM):
            """
            Wrapper class for the FlotorchLLM.
            It is used to wrap the FlotorchLLM and make it compatible with the DeepEval framework.
            Args:
                inference_llm: The model ID of the underlying inference LLM.
                api_key: The API key for the FlotorchLLM.
                base_url: The base URL for the FlotorchLLM.
                *args: Additional arguments to pass to the DeepEvalBaseLLM class.
                **kwargs: Additional keyword arguments to pass to the DeepEvalBaseLLM class.
            Returns:
                The model ID of the underlying inference LLM.
            Raises:
                ValueError: If the inference LLM is not provided.
            """

            def __init__(
                self, inference_llm: str, api_key: str, base_url: str, *args, **kwargs
            ):
                """
                Initializes the FloTorchLLMWrapper.
                """
                self.api_key = api_key
                self.base_url = base_url
                self.inference_llm = inference_llm
                self.client = self.load_model()
                super().__init__(*args, **kwargs)

            def get_model_name(self, *args, **kwargs) -> str:
                """
                Returns the model ID of the underlying inference LLM.
                """
                return self.inference_llm

            def generate(self, *args, **kwargs):
                """
                Generates a response for a prompt and validates it against a schema if provided.
                
                Args:
                    prompt (str): The prompt to generate from.
                    schema (Optional[Type[BaseModel]]): Optional schema for validation.
                """
                prompt = args[0] if args else kwargs.get('prompt')
                schema = args[1] if len(args) > 1 else kwargs.get('schema', None)

                response = self.client.invoke(
                    messages=[{"role": "user", "content": prompt}]
                )

                # Handle tuple (response, headers) if return_headers=True was used
                if isinstance(response, tuple):
                    response = response[0]

                # By default (n=1), response is a single LLMResponse object
                completion = response.content
                return self._schema_validation(completion, schema)

            async def a_generate(self, *args, **kwargs) -> str:
                """
                Asynchronously generates a response for a prompt and
                validates it against a schema if provided.
                
                Args:
                    prompt (str): The prompt to generate from.
                    schema (Optional[Type[BaseModel]]): Optional schema for validation.
                """
                prompt = args[0] if args else kwargs.get('prompt')
                schema = args[1] if len(args) > 1 else kwargs.get('schema', None)

                response = await self.client.ainvoke(
                    messages=[{"role": "user", "content": prompt}]
                )

                # Handle tuple (response, headers) if return_headers=True was used
                if isinstance(response, tuple):
                    response = response[0]

                # By default (n=1), response is a single LLMResponse object
                completion = response.content
                return self._schema_validation(completion, schema)

            def load_model(self, *args, **kwargs):
                """
                Loads and returns the inference LLM client.
                """
                return FlotorchLLM(
                    model_id=self.inference_llm,
                    api_key=self.api_key,
                    base_url=self.base_url,
                )

            @retry(
                wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
                retry=retry_if_exception_type(ValueError),
                stop=stop_after_attempt(3),
            )
            def _schema_validation(
                self, completion: str, schema: Optional[Type[BaseModel]] = None
            ) -> str:
                if schema:
                    try:
                        return schema.model_validate(json.loads(completion))
                    except (json.JSONDecodeError, ValueError):
                        json_output = self.trim_json(completion)
                        return schema.model_validate(json.loads(json_output))
                else:
                    return completion.strip()

            def _llm_fix_json_prompt(self, bad_json: str) -> str:
                """
                Builds a prompt for fixing malformed JSON.
                Args:
                    bad_json (str): The malformed JSON to fix.
                Returns:
                    str: The fixed JSON.
                """
                return f"""The following is a malformed JSON
                (possibly incomplete or with syntax issues).
                Fix it so that it becomes **valid JSON**.
                Instructions:
                - Do **NOT** include Markdown formatting (no triple backticks, no ```json).
                - Do **NOT** add or invent any new keys or values.
                - Only fix unclosed strings, arrays, or braces.
                - Do **NOT** add commas or fields that were not originally present.
                - **Remove any trailing commas** at the end of JSON objects or arrays.
                - **Ensure all property names and string values are enclosed in double quotes**.
                - Preserve the original structure and values.
                - If a list like "truths" or "verdicts" or "statements" or "reason" seems incomplete, just close it properly.
                - Do **NOT** include Markdown formatting (no triple backticks, no ```json).
                - Do **NOT** add or invent any new keys or values.
                - Only fix unclosed strings, arrays, or braces.
                - Do **NOT** add commas or fields that were not originally present.
                - **Remove any trailing commas** at the end of JSON objects or arrays.
                - **Ensure all property names and string values are enclosed in double quotes**.
                - Preserve the original structure and values.
                - If a list like "truths" or "verdicts" or "statements" or "reason" seems incomplete, just close it properly.
                - Output **only** valid, raw JSON. No explanation, no surrounding text, no markdown.

                Malformed JSON to fix:
                # {json.dumps(bad_json)}
                {bad_json} #TODO See if this works
                """

            def fix_common_truncation(self, json_str: str) -> str:
                """
                Fixes common truncation issues in JSON strings.
                """
                if not json_str.endswith("]") and not json_str.endswith("}"):
                    json_str += '"}]}'
                return json_str

            def trim_json(self, completion: str) -> str:
                """
                Trims JSON strings to remove trailing commas and ensure they are valid JSON.
                """
                prompt = self._llm_fix_json_prompt(completion)

                response = self.client.invoke(
                    messages=[{"role": "user", "content": prompt}]
                )

                # Handle tuple (response, headers) if return_headers=True was used
                if isinstance(response, tuple):
                    response = response[0]

                # By default (n=1), response is a single LLMResponse object
                fixed_json = response.content
                fixed_json = fixed_json.strip()

                # Optional: Validate the output is valid JSON
                try:
                    json.loads(fixed_json)
                except json.JSONDecodeError as e:
                    print(f"Detected JSON truncation. Trying naive fix. {e}")
                    fixed_json = self.fix_common_truncation(fixed_json)
                    try:
                        json.loads(fixed_json)
                    except json.JSONDecodeError as e2:
                        error_msg = (
                            f"Model returned invalid JSON (even after fix): {e2}"
                            f"\n\nReturned:\n{fixed_json}"
                        )
                        raise ValueError(error_msg) from e2
                return fixed_json

        self.llm = FloTorchLLMWrapper(
            inference_llm=evaluator_llm, api_key=api_key, base_url=base_url
        )
        self.async_config = AsyncConfig(
            run_async=async_run, max_concurrent=max_concurrent
        )
        self.custom_metrics = custom_metrics or []
        self.metric_args = metric_args

        # Initialize DeepEval metrics from the registry
        DeepEvalEvaluationMetrics.initialize_metrics(
            llm=self.llm, metric_args=self.metric_args
        )

    def _build_test_cases(self, data: List[EvaluationItem]) -> List[LLMTestCase]:
        """
        Converts evaluation data into LLM test cases for DeepEval evaluation.
        """
        return [
            LLMTestCase(
                input=item.question,
                actual_output=item.generated_answer,
                expected_output=item.expected_answer,
                retrieval_context=item.context or [],
                context=item.context or [],
            )
            for item in data
        ]

    def _process_results(self, eval_results) -> Dict[str, float]:
        """
        Processes the raw DeepEval results to calculate average scores
        for each metric across all test cases.

        Args:
            eval_results: EvaluationResult object from deepeval.evaluate().

        Returns:
            A dictionary mapping metric keys (from MetricKey enum) to their
            average score.
        """
        deepeval_to_metric_key = {
            "Contextual Relevancy": MetricKey.CONTEXT_RELEVANCY,
            "Contextual Recall": MetricKey.CONTEXT_RECALL,
            "Hallucination": MetricKey.HALLUCINATION,
            "Faithfulness": MetricKey.FAITHFULNESS,
            "Answer Relevancy": MetricKey.ANSWER_RELEVANCE,
            "Contextual Precision": MetricKey.CONTEXT_PRECISION
        }

        metric_scores = defaultdict(list)

        for test_result in eval_results.test_results:
            for metric_data in test_result.metrics_data:
                metric_key_enum = deepeval_to_metric_key.get(metric_data.name)
                if metric_key_enum and metric_data.score is not None:
                    metric_scores[metric_key_enum.value].append(metric_data.score)

        averaged_results = {}
        for metric_name, scores in metric_scores.items():
            if scores:
                averaged_results[metric_name] = round(sum(scores) / len(scores), 2)

        return averaged_results

    def evaluate(
        self, data: List[EvaluationItem], metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        test_cases = self._build_test_cases(data)
        # example to fetch metrics, use like this
        if metrics is None:
            metrics = DeepEvalEvaluationMetrics.available_metrics()

        selected_metrics = [DeepEvalEvaluationMetrics.get_metric(m) for m in metrics]
        eval_results = evaluate(
            test_cases=test_cases,
            async_config=self.async_config,
            metrics=selected_metrics + self.custom_metrics,
            error_config=ErrorConfig(ignore_errors=True),
        )
        # Process results to get averaged scores
        processed_results = self._process_results(eval_results)
        return processed_results

