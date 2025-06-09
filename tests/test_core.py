"""
Tests for core functionality.
"""

import json
from datetime import datetime, timezone
from unittest import TestCase, main

from opentelemetry.trace import Span as OTelSpan
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.span import SpanContext, TraceFlags

from flotorch_eval.agent_eval import Evaluator, TraceConverter
from flotorch_eval.agent_eval.metrics import ToolCallAccuracyMetric
from flotorch_eval.agent_eval.core.schemas import (
    Message,
    Span,
    SpanEvent,
    ToolCall,
    Trajectory,
)


def create_test_trajectory() -> Trajectory:
    """Create a test trajectory with known data."""
    now = datetime.now(timezone.utc)

    tool_call = ToolCall(
        name="test_tool",
        arguments={"key": "value"},
        output={"result": "success"},
        success=True,
        error=None,
        start_time=now,
        end_time=now,
    )

    message = Message(
        role="assistant", content="Test message", timestamp=now, tool_calls=[tool_call]
    )

    return Trajectory(trace_id="test-trace-id", messages=[message])


def test_tool_accuracy_metric():
    """Test the tool accuracy metric computation."""
    trajectory = create_test_trajectory()
    metric = ToolCallAccuracyMetric()

    result = metric.compute(trajectory)

    assert result.name == "tool_accuracy"
    assert result.score == 1.0  # All tool calls successful
    assert result.details["total_calls"] == 1
    assert result.details["successful_calls"] == 1
    assert result.details["failed_calls"] == 0


def test_evaluator():
    """Test the evaluator with multiple metrics."""
    trajectory = create_test_trajectory()
    evaluator = Evaluator([ToolCallAccuracyMetric()])

    results = evaluator.evaluate(trajectory)

    assert results.trajectory_id == "test-trace-id"
    assert len(results.scores) == 1
    assert results.scores[0].name == "tool_accuracy"
    assert results.scores[0].score == 1.0


def test_trajectory_validation():
    """Test that invalid trajectories raise validation errors."""
    with pytest.raises(ValueError):
        Trajectory(trace_id="", messages=[])  # Empty trace ID

    now = datetime.now(timezone.utc)
    invalid_tool_call = ToolCall(
        name="",  # Empty tool name
        arguments={},
        output={},
        success=True,
        error=None,
        start_time=now,
        end_time=now,
    )

    with pytest.raises(ValueError):
        Message(
            role="invalid_role",  # Invalid role
            content="Test",
            timestamp=now,
            tool_calls=[invalid_tool_call],
        )


def test_evaluator_no_metrics():
    """Test that evaluator raises error when no metrics are provided."""
    trajectory = create_test_trajectory()
    evaluator = Evaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(trajectory)
