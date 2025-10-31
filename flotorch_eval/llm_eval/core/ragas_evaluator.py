"""
Ragas Evaluator Module.

This module implements the Ragas evaluator for RAG-based metrics evaluation.
"""
from typing import Any, Dict, List, Optional, Union
from itertools import chain
from ragas.evaluation import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from flotorch_eval.llm_eval.core.base_evaluator import BaseEvaluator
from flotorch_eval.llm_eval.core.schemas import EvaluationItem
from flotorch_eval.llm_eval.metrics.ragas_metrics.ragas_metrics import RagasEvaluationMetrics
from flotorch_eval.llm_eval.metrics.ragas_metrics.metric_keys import MetricKey

class RagasEvaluator(BaseEvaluator):
    """
    Evaluator that uses RAGAS metrics to score RAG-based QA performance.
    Initializes the RagasEvaluator with the given LLM and embedding wrappers.

    Args:
        evaluator_llm: The LLM to be used by RAGAS metrics (wrapped in LangchainLLMWrapper).
        embedding_llm: The embedding model to be used by RAGAS metrics
        (wrapped in LangchainEmbeddingsWrapper).
        metric_args: Optional configuration for metrics requiring per-instance arguments.

            Example:
            {
                MetricKey.ASPECT_CRITIC: {
                    "maliciousness": {
                        "name": "maliciousness",
                        "definition": "Is the response harmful?"
                    },
                    "bias": {
                        "name": "bias",
                        "definition": "Is the response biased or discriminatory?"
                    }
                }
            }
    """
    def __init__(
        self,
        base_url: str,
        api_key: str,
        embedding_model: str,
        inferencer_model: str,
        metric_args: Optional[
            Dict[Union[str, MetricKey], Dict[str, Dict[str, str]]]
        ] = None
    ):
        base_url = base_url + "/openai/v1"
        self._configure_models(
            base_url=base_url,
            api_key=api_key,
            embedding_model=embedding_model,
            inferencer_model=inferencer_model
        )
        self.metric_args = metric_args

        RagasEvaluationMetrics.initialize_metrics(
            llm=self.llm_model,
            embeddings=self.embedding_model,
            metric_args=self.metric_args
        )

    def _configure_models(
        self,
        base_url: str,
        api_key: str,
        embedding_model: str=None,
        inferencer_model: str=None
    ) -> None:
        """
        Configure the models for the RagasEvaluator.
        """
        # If both embedding_model and inferencer_model are None, raise an error
        if embedding_model is None and inferencer_model is None:
            raise ValueError("Either embedding_model or inferencer_model must be provided")

        # If inferencer_model is provided, configure the inferencer model
        if inferencer_model is not None:
            inferencer_llm_args = {
                "openai_api_base": base_url,
                "openai_api_key": api_key,
                "model": inferencer_model,
            }
            llm = ChatOpenAI(**inferencer_llm_args)

             # bypassing n as gateway currently does not support it.
            self.llm_model = LangchainLLMWrapper(llm, bypass_n=True)

        # If embedding_model is provided, configure the embedding model
        if embedding_model is not None:
            embedding_args = {
                "openai_api_base": base_url,
                "openai_api_key": api_key,
                "model": embedding_model,
                "check_embedding_ctx_length": False
            }
            embeddings = OpenAIEmbeddings(**embedding_args)
            self.embedding_model = LangchainEmbeddingsWrapper(embeddings=embeddings)


    def evaluate(
        self,
        data: List[EvaluationItem],
        metrics: Optional[List[MetricKey]] = None
    ) -> Dict[str, Any]:
        # example to fetch metrics, use like this
        if metrics is None:
            metrics = RagasEvaluationMetrics.available_metrics()

        selected_metrics = list(chain.from_iterable(
            RagasEvaluationMetrics.get_metric(m).values() for m in metrics
        ))

        answer_samples = []
        for item in data:
            sample_params = {
                "user_input": item.question,
                "response": item.generated_answer,
                "reference": item.expected_answer,
                "retrieved_contexts": item.context
            }
            answer_samples.append(SingleTurnSample(**sample_params))

        evaluation_dataset = EvaluationDataset(answer_samples)

        result = evaluate(evaluation_dataset, selected_metrics)
        results_dict = result._repr_dict
        rounded_results = {k: round(v, 2) for k, v in results_dict.items()}
        return rounded_results
