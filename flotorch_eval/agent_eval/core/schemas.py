"""
Core schemas for agent evaluation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call made by an agent."""
    id: str
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
    tool_call_id: Optional[str] = None
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
    
class ReferenceToolCall(BaseModel):
    """A representation of an expected tool call for a reference trajectory."""
    name: str = Field(description="The name of the tool or function that should be called.")
    arguments: Dict[str, Any] = Field(description="The dictionary of arguments expected to be passed to the tool.")

class ReferenceTrajectory(BaseModel):
    """
    Defines the "golden path" for an agent interaction, serving as a reference for evaluation.
    This can be created manually or generated from an existing trace.
    """
    input: str = Field(description="The initial user input or prompt that starts the trajectory.")
    expected_tool_calls: List[ReferenceToolCall] = Field(
        default_factory=list,
        description="An ordered list of tool calls that the agent is expected to make."
    )
    final_response: Optional[str] = Field(
        default=None,
        description="The final text response expected from the agent after all tool calls are complete."
    )


class MetricResult(BaseModel):
    """Result from a single metric evaluation."""

    name: str
    score: float
    details: Optional[Dict[str, Union[str, int, float, bool, List[str], List[Dict[str, Union[str, int, float]]]]]]


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
    cost: str


class CostSummary(BaseModel):
    """Aggregate and per-span cost results."""
    total_cost: str
    average_cost_per_call: str
    cost_breakdown: List[CostRecord]

class LatencyBreakdownItem(BaseModel):
    """A Pydantic model for a single latency step."""
    step_name: str
    latency_ms: float

class LatencySummary(BaseModel):
    """A Pydantic model for the complete latency summary."""
    total_latency_ms: float
    average_step_latency_ms: float
    latency_breakdown: List[LatencyBreakdownItem]