"""
Gateway Metrics Module.

This module provides the computation of aggregate latency, cost, and token usage
from LLM call metadata.
"""
from typing import List, Dict, Any
from flotorch_eval.llm_eval.core.schemas import EvaluationItem


class GatewayMetrics:
    """
    Computes aggregate metrics from gateway metadata in EvaluationItems.
    
    Expected metadata keys:
    - 'x-gateway-total-latency': Total latency in milliseconds for the LLM call
    - 'x-total-cost': Total cost for the LLM call
    - 'x-total-tokens': Total tokens used in the LLM call
    """

    @staticmethod
    def compute_metrics(data: List[EvaluationItem]) -> Dict[str, Any]:
        """
        Compute aggregate gateway metrics from evaluation data.
        
        Args:
            data: List of EvaluationItems containing metadata
            
        Returns:
            Dictionary with aggregate metrics:
            {
                'total_latency_ms': Total latency across all items,
                'average_latency_ms': Average latency per item,
                'total_cost': Total cost across all items,
                'average_cost': Average cost per item,
                'total_tokens': Total tokens used across all items,
                'items_with_metadata': Number of items that had metadata,
                'total_items': Total number of items
            }
        """
        total_latency = 0.0
        total_cost = 0.0
        total_tokens = 0
        items_with_metadata = 0

        for item in data:
            if item.metadata:
                items_with_metadata += 1

                # Extract latency
                if 'x-gateway-total-latency' in item.metadata:
                    total_latency += float(item.metadata['x-gateway-total-latency'])

                # Extract cost
                if 'x-total-cost' in item.metadata:
                    total_cost += float(item.metadata['x-total-cost'])

                # Extract tokens
                if 'x-total-tokens' in item.metadata:
                    total_tokens += int(item.metadata['x-total-tokens'])

        # Calculate averages
        average_latency = total_latency / items_with_metadata if items_with_metadata > 0 else 0.0
        average_cost = total_cost / items_with_metadata if items_with_metadata > 0 else 0.0

        return {
            'total_latency_ms': round(total_latency, 4),
            'average_latency_ms': round(average_latency, 4),
            'total_cost': round(total_cost, 6),
            'average_cost': round(average_cost, 6),
            'total_tokens': total_tokens,
            'items_with_metadata': items_with_metadata,
            'total_items': len(data)
        }

    @staticmethod
    def has_metadata(data: List[EvaluationItem]) -> bool:
        """
        Check if any evaluation items contain gateway metadata.
        
        Args:
            data: List of EvaluationItems
            
        Returns:
            True if at least one item has metadata with gateway metrics
        """
        for item in data:
            if item.metadata and any(
                key in item.metadata 
                for key in ['x-gateway-total-latency', 'x-total-cost', 'x-total-tokens']
            ):
                return True
        return False
