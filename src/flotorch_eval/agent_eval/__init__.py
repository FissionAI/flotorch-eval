"""
Agent evaluation package.
"""

from flotorch_eval.agent_eval.core.evaluator import Evaluator
from flotorch_eval.agent_eval.core.schemas import (
    EvaluationResult,
    Message,
    MetricResult,
    Span,
    SpanEvent,
    ToolCall,
    Trajectory,
)
from flotorch_eval.agent_eval.core.trace_converter import TraceConverter
from flotorch_eval.agent_eval.metrics.base import BaseMetric
from flotorch_eval.agent_eval.metrics.langchain_metrics import TrajectoryEvalWithLLMMetric
from flotorch_eval.agent_eval.metrics.ragas_metrics import (
    AgentGoalAccuracyMetric,
    ToolCallAccuracyMetric,
)

__all__ = [
    "BaseMetric",
    "Evaluator",
    "EvaluationResult",
    "Message",
    "MetricResult",
    "Span",
    "SpanEvent",
    "ToolCall",
    "Trajectory",
    "TraceConverter",
    "TrajectoryEvalWithLLMMetric",
    "AgentGoalAccuracyMetric",
    "ToolCallAccuracyMetric",
]
