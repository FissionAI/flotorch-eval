"""
Converter module for transforming OpenTelemetry traces into agent trajectories.
"""

from datetime import datetime
import ast
import json
import re
from typing import Dict, List, Optional, Union, Any

from opentelemetry.trace import Span as OTelSpan
from opentelemetry.trace import SpanKind

from flotorch_eval.agent_eval.core.schemas import Message, Span, SpanEvent, ToolCall, Trajectory
from flotorch_eval.common.utils import convert_attributes


class TraceConverter:
    """Converts OpenTelemetry traces from FloTorch Gateway into agent trajectories."""

    def from_spans(self, spans: List[OTelSpan]) -> Trajectory:
        """
        Constructs a Trajectory from a list of FloTorch OTel spans.

        This method processes spans based on the official OpenTelemetry GenAI semantic
        conventions, extracting conversation messages from span events.
        """
        sorted_spans = sorted(spans, key=lambda x: x.start_time)
        internal_spans = []

        for span in sorted_spans:
            internal_span = Span(
                span_id=format(span.context.span_id, "016x"),
                trace_id=format(span.context.trace_id, "032x"),
                parent_id=format(span.parent.span_id, "016x") if span.parent else None,
                name=span.name,
                start_time=datetime.fromtimestamp(span.start_time / 1e9),
                end_time=datetime.fromtimestamp(span.end_time / 1e9),
                attributes=self._convert_attributes(span.attributes),
                events=[
                    SpanEvent(
                        name=event.name,
                        timestamp=datetime.fromtimestamp(event.timestamp / 1e9),
                        attributes=self._convert_attributes(event.attributes),
                    )
                    for event in span.events
                ],
            )
            internal_spans.append(internal_span)

        messages: List[Message] = []
        tool_calls_map: Dict[str, ToolCall] = {}

        for span in internal_spans:
            sorted_events = sorted(span.events, key=lambda e: e.timestamp)
            
            for event in sorted_events:
                event_attrs = event.attributes
                
                # --- Handle User Message ---
                if event.name == "gen_ai.user.message":
                    content = event_attrs.get("message.content", "")
                    if content:
                        messages.append(
                            Message(
                                role="user",
                                content=content,
                                timestamp=event.timestamp,
                                tool_calls=[],
                            )
                        )
                
                # --- Handle Assistant Message (thought and tool calls) ---
                elif event.name in ("gen_ai.assistant.message", "gen_ai.choice"):
                    thought = event_attrs.get("message.content") or ""
                    tool_calls_str = event_attrs.get("message.tool_calls")
                    parsed_tool_calls: List[ToolCall] = []

                    if tool_calls_str and isinstance(tool_calls_str, str):
                        try:
                            tool_calls_data = json.loads(tool_calls_str)
                            for tc_data in tool_calls_data:
                                function_data = tc_data.get("function", {})
                                try:
                                    arguments = json.loads(function_data.get("arguments", "{}"))
                                except (json.JSONDecodeError, TypeError):
                                    arguments = {"raw": function_data.get("arguments")}

                                tool_call = ToolCall(
                                    id=tc_data.get("id"),
                                    name=function_data.get("name", ""),
                                    arguments=arguments,
                                    timestamp=event.timestamp,
                                    output=None,
                                )
                                parsed_tool_calls.append(tool_call)
                                if tool_call.id:
                                    tool_calls_map[tool_call.id] = tool_call
                        except (json.JSONDecodeError, TypeError, AttributeError):
                            pass
                    
                    messages.append(
                        Message(
                            role="assistant",
                            content=thought,
                            timestamp=event.timestamp,
                            tool_calls=parsed_tool_calls,
                        )
                    )
                    
                # --- Handle System Message ---
                elif event.name == "gen_ai.system.message":
                    content = event_attrs.get("message.content", "")
                    if content:
                        messages.append(
                            Message(
                                role="system",
                                content=content,
                                timestamp=event.timestamp,
                                tool_calls=[],
                            )
                        )
                
                # --- Handle Tool Message (tool result) ---
                elif event.name == "gen_ai.tool.message":
                    tool_output = event_attrs.get("message.content", "")
                    tool_call_id = event_attrs.get("tool.call.id")

                    # Link the output back to the original tool call
                    if tool_call_id and tool_call_id in tool_calls_map:
                        tool_calls_map[tool_call_id].output = tool_output

                    messages.append(
                        Message(
                            role="tool",
                            content=tool_output,
                            timestamp=event.timestamp,
                            tool_calls=[],
                        )
                    )

        return Trajectory(
            trace_id=format(spans[0].context.trace_id, "032x") if spans else "",
            messages=messages,
            spans=internal_spans,
        )

    def _convert_attributes(
        self, attributes: Dict[str, Any]
    ) -> Dict[str, Union[str, int, float, bool, List[str]]]:
        """
        Sanitizes attribute values to be JSON-serializable primitives.
        Complex objects are converted to JSON strings.
        """
        result = {}
        if not attributes:
            return result
            
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)) or (
                isinstance(value, list)
                and all(isinstance(x, (str, int, float, bool)) for x in value)
            ):
                result[key] = value
            else:
                try:
                    result[key] = json.dumps(value)
                except TypeError:
                    result[key] = str(value)
        return result
        """Extracts the user's explicit task from the initial prompt structure."""
        user_content = prompt.strip()

        # Try to extract from gen_ai.prompt dictionary
        if isinstance(user_content, dict) and "gen_ai.prompt" in user_content:
            user_content = user_content["gen_ai.prompt"]

        # Try to extract from JSON string
        try:
            data = json.loads(user_content)
            if isinstance(data, dict) and "gen_ai.prompt" in data:
                user_content = data["gen_ai.prompt"]
        except (json.JSONDecodeError, TypeError):
            pass

        # Look for task in system prompt format
        task_match = re.search(
            r"Current Task:\s*(.*?)(?=\n\nThis is the expected criteria|$)",
            user_content,
            re.DOTALL,
        )
        if task_match:
            user_content = task_match.group(1).strip()
            return user_content

        # Look for direct user message format
        if "user:" in user_content:
            user_content = user_content.split("user:", 1)[1].strip()

            # Remove any remaining system prompt parts
            if "system:" in user_content:
                user_content = user_content.split("system:", 1)[0].strip()

            # Remove any trailing JSON artifacts
            user_content = user_content.rstrip('"}')

            # Extract just the task part if criteria is included
            if "This is the expected criteria" in user_content:
                user_content = user_content.split("This is the expected criteria", 1)[
                    0
                ].strip()

            return user_content.strip()

        return user_content.strip()