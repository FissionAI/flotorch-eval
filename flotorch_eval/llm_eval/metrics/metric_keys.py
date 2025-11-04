"""
Metric Keys Module.

This module defines the enumeration of available evaluation metric keys.
"""
from enum import Enum


class MetricKey(str, Enum):
    """
    Enumeration of available evaluation metric keys.
    
    Used to reference metrics consistently across different evaluation engines.
    """
    CONTEXT_PRECISION = "context_precision"
    ASPECT_CRITIC = "aspect_critic"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RECALL = "contextual_recall"
    CONTEXT_RELEVANCY = "contextual_relevancy"
    HALLUCINATION = "hallucination"
