"""
Core schemas for agent evaluation.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call made by an agent."""

    name: str = Field(description="Name of the tool called")
    arguments: Dict[str, Union[str, int, float, bool, List[str]]] = Field(
        description="Arguments passed to the tool"
    )
    output: Optional[str] = Field(None, description="Output from the tool")
    timestamp: Optional[datetime] = Field(None, description="When the tool was invoked")


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
    score: Union[float, Literal["N/A"]]
    details: Optional[Dict[str, Any]]


class EvaluationResult(BaseModel):
    """Complete evaluation results for a trajectory."""

    trajectory_id: str
    scores: List[MetricResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Union[str, int, float, bool, List[str]]] = Field(
        default_factory=dict
    )


class TokenUsageRecord(BaseModel):
    """Represents token usage details for a single span."""
    span_name: str
    span_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class TokenTotals(BaseModel):
    """Aggregated totals for all token usage across spans."""
    input_tokens: int
    output_tokens: int
    total_tokens: int


class TokenUsageSummary(BaseModel):
    """Structured response containing per-span token usage and overall totals."""
    token_usage: List[TokenUsageRecord]
    totals: TokenTotals

class CostRecord(BaseModel):
    """Per-span cost breakdown."""
    span_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


class CostSummary(BaseModel):
    """Aggregate and per-span cost results."""
    total_cost: float
    average_cost_per_call: float
    cost_breakdown: List[CostRecord]

class LatencyBreakdownItem:
    def __init__(self, step_name: str, latency_ms: float, children: Optional[List['LatencyBreakdownItem']] = None):
        self.step_name = step_name
        self.latency_ms = latency_ms
        self.children = children or []

    def to_dict(self) -> Dict:
        return {
            "step_name": self.step_name,
            "latency_ms": self.latency_ms,
            "children": [child.to_dict() for child in self.children]
        }

class LatencySummary:
    def __init__(
        self,
        total_latency_ms: float,
        average_step_latency_ms: float,
        latency_breakdown: List[LatencyBreakdownItem]
    ):
        self.total_latency_ms = total_latency_ms
        self.average_step_latency_ms = average_step_latency_ms
        self.latency_breakdown = latency_breakdown

    def to_dict(self) -> Dict:
        return {
            "total_latency_ms": self.total_latency_ms,
            "average_step_latency_ms": self.average_step_latency_ms,
            "latency_breakdown": [item.to_dict() for item in self.latency_breakdown],
        }
    
class EvaluationScore(BaseModel):
    score: Union[float, str, int]
    metadata: Dict[str, Any]