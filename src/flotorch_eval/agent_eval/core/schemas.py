"""
Core schemas for agent evaluation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call made by an agent."""

    name: str = Field(description="Name of the tool called")
    arguments: Dict[str, Union[str, int, float, bool, List[str]]] = Field(
        description="Arguments passed to the tool"
    )
    output: Optional[str] = Field(None, description="Output from the tool")


class Message(BaseModel):
    """A message in an agent trajectory."""

    role: str = Field(description="Role of the message sender (user/assistant/tool)")
    content: str = Field(description="Content of the message")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls made in this message")
    timestamp: Optional[datetime] = Field(None, description="When the message was sent")


class SpanEvent(BaseModel):
    """An event in a span."""

    name: str = Field(description="Name of the event")
    timestamp: datetime = Field(description="When the event occurred")
    attributes: Dict[str, Union[str, int, float, bool, List[str]]] = Field(
        default_factory=dict, description="Attributes of the event"
    )


class Span(BaseModel):
    """A span in a trace."""

    span_id: str = Field(description="Unique identifier for the span")
    trace_id: str = Field(description="Identifier of the trace this span belongs to")
    parent_id: Optional[str] = Field(None, description="Identifier of the parent span")
    name: str = Field(description="Name of the span")
    start_time: datetime = Field(description="When the span started")
    end_time: datetime = Field(description="When the span ended")
    attributes: Dict[str, Union[str, int, float, bool, List[str]]] = Field(
        default_factory=dict, description="Attributes of the span"
    )
    events: List[SpanEvent] = Field(default_factory=list, description="Events in the span")


class Trajectory(BaseModel):
    """A trajectory of agent interactions."""

    trace_id: str = Field(description="Unique identifier for the trajectory")
    messages: List[Message] = Field(description="Messages in the trajectory")
    spans: List[Span] = Field(description="Spans in the trajectory")


class MetricResult(BaseModel):
    """Result from a single metric evaluation."""

    name: str
    score: float
    details: Optional[Dict[str, Union[str, int, float, bool, List[str]]]]


class EvaluationResult(BaseModel):
    """Complete evaluation results for a trajectory."""

    trajectory_id: str
    scores: List[MetricResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Union[str, int, float, bool, List[str]]] = Field(
        default_factory=dict
    )
