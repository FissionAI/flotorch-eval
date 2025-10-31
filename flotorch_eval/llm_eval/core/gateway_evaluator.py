"""
Gateway Evaluator for computing operational metrics from LLM call metadata.
"""
from typing import List, Dict, Any
from flotorch_eval.llm_eval.core.base_evaluator import BaseEvaluator
from flotorch_eval.llm_eval.core.schemas import EvaluationItem
from flotorch_eval.llm_eval.metrics.gateway_metrics.gateway_metrics import GatewayMetrics


class GatewayEvaluator(BaseEvaluator):
    """
    Evaluator for computing aggregate operational metrics (latency, cost, tokens)
    from gateway metadata in evaluation items.
    
    This evaluator does not require LLM or embedding models as it only aggregates
    metadata from the evaluation items.
    """
    def __init__(self):
        """Initialize the GatewayEvaluator."""
        pass

    def evaluate(
        self,
        data: List[EvaluationItem],
        metrics: Any = None  # Not used, kept for interface compatibility
    ) -> Dict[str, Any]:
        """
        Compute gateway metrics from evaluation data metadata.
        
        Args:
            data: List of EvaluationItems containing metadata with gateway metrics
            metrics: Not used for gateway evaluation (kept for interface compatibility)
            
        Returns:
            Dictionary containing aggregate gateway metrics:
            {
                'gateway_metrics': {
                    'total_latency_ms': float,
                    'average_latency_ms': float,
                    'total_cost': float,
                    'total_tokens': int,
                    'items_with_metadata': int,
                    'total_items': int
                }
            }
        """
        gateway_metrics = GatewayMetrics.compute_metrics(data)

        return {
            'gateway_metrics': gateway_metrics
        }
