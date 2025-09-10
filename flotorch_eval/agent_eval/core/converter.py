"""
Converter module for transforming OpenTelemetry traces into agent trajectories.

This module provides the TraceConverter class, which is responsible for parsing
OpenTelemetry trace data (in Protobuf JSON format) and converting it into a
structured Trajectory object suitable for downstream agent evaluation and analysis.

The conversion process extracts relevant span information, reconstructs the
conversation flow, and handles tool call associations.
"""

from datetime import datetime
import ast
import re
from typing import Dict, List, Any
from flotorch_eval.agent_eval.core.schemas import Message, Span, SpanEvent, ToolCall, Trajectory

class TraceConverter:
    """
    Converts OpenTelemetry traces into agent trajectories.

    This class provides methods to parse OpenTelemetry trace data and reconstruct
    the agent's conversation, including user/system/assistant messages, tool calls,
    and span metadata.
    """

    def from_spans(self, trace_data: Dict[str, Any]) -> Trajectory:
        """
        Constructs a Trajectory from an OpenTelemetry Protobuf JSON object.

        Args:
            trace_data (Dict[str, Any]): The OpenTelemetry trace data as a dictionary.

        Returns:
            Trajectory: The reconstructed agent trajectory, including messages and spans.

        This method parses the nested OTel structure and extracts span attributes
        to reconstruct the agent's conversation flow. It handles user/system/assistant
        messages, tool calls, and tool outputs, and sorts all messages chronologically.
        """
        # Extract resourceSpans from the trace data
        resource_spans = trace_data.get("resourceSpans", [])
        if not resource_spans:
            # Return an empty trajectory if no resourceSpans are present
            return Trajectory(trace_id="", messages=[], spans=[])

        # Flatten all spans from all scopeSpans in all resourceSpans
        raw_spans = []
        for rs in resource_spans:
            for ss in rs.get("scopeSpans", []):
                raw_spans.extend(ss.get("spans", []))

        if not raw_spans:
            # Return an empty trajectory if no spans are present
            return Trajectory(trace_id="", messages=[], spans=[])
            
        # Use the traceId from the first span as the trajectory's trace_id
        trace_id = raw_spans[0].get("traceId", "")

        internal_spans: List[Span] = []
        for span_dict in raw_spans:
            # Parse events for each span, if present
            events = [
                SpanEvent(
                    name=evt.get("name", ""),
                    timestamp=datetime.fromtimestamp(int(evt.get("timeUnixNano", 0)) / 1e9),
                    attributes=self._convert_otel_attributes(evt.get("attributes", [])),
                )
                for evt in span_dict.get("events", [])
            ]
            
            # Construct the Span object with parsed attributes and events
            span = Span(
                span_id=span_dict.get("spanId", ""),
                trace_id=trace_id,
                parent_id=span_dict.get("parentSpanId"),
                name=span_dict.get("name", ""),
                start_time=datetime.fromtimestamp(int(span_dict.get("startTimeUnixNano", 0)) / 1e9),
                end_time=datetime.fromtimestamp(int(span_dict.get("endTimeUnixNano", 0)) / 1e9),
                attributes=self._convert_otel_attributes(span_dict.get("attributes", [])),
                events=events,
            )
            internal_spans.append(span)

        # Sort spans by their start time to maintain chronological order
        sorted_spans = sorted(internal_spans, key=lambda s: s.start_time)

        messages: List[Message] = []
        tool_calls_map: Dict[str, ToolCall] = {}  # Maps tool call IDs to ToolCall objects

        # Iterate through each span to extract messages and tool calls
        for span in sorted_spans:
            attrs = span.attributes
            operation = attrs.get("gen_ai.operation.name")

            # Handle user/system messages from the request
            if "gen_ai.request.messages" in attrs:
                messages_str = attrs["gen_ai.request.messages"]
                try:
                    # Safely parse the string representation of the messages list
                    request_messages = ast.literal_eval(messages_str)
                    for msg in request_messages:
                        role = msg.get("role")
                        content = msg.get("content")
                        # Only add user or system messages with content
                        if role in ("user", "system") and content:
                            messages.append(
                                Message(
                                    role=role,
                                    content=content,
                                    timestamp=span.start_time,
                                    tool_calls=[]
                                )
                            )
                except (ValueError, SyntaxError):
                    # Ignore malformed message lists
                    pass
            
            # Handle assistant responses and tool calls
            if operation in ("chat", "invoke_agent") and "gen_ai.response.content" in attrs:
                thought = attrs.get("gen_ai.response.content", "")
                full_response_str = attrs.get("gen_ai.response.full", "")
                parsed_tool_calls: List[ToolCall] = []

                # Attempt to extract tool calls from the full response string
                if full_response_str:
                    # Use regex to find the tool_calls list in the string representation
                    match = re.search(r"tool_calls': (\[.*?\])", full_response_str.replace('\\', ''))
                    if match:
                        tool_calls_repr = match.group(1)
                        try:
                            tool_calls_data = ast.literal_eval(tool_calls_repr)
                            for tc_data in tool_calls_data:
                                function_data = tc_data.get("function", {})
                                arguments_raw = function_data.get("arguments", {})
                                # Ensure arguments are a dictionary; otherwise, wrap as raw
                                arguments = arguments_raw if isinstance(arguments_raw, dict) else {"raw": str(arguments_raw)}

                                tool_call = ToolCall(
                                    id=tc_data.get("id"),
                                    name=function_data.get("name", ""),
                                    arguments=arguments,
                                    timestamp=span.end_time,
                                    output=None,
                                )
                                parsed_tool_calls.append(tool_call)
                                # Store tool call by ID for later output association
                                if tool_call.id:
                                    tool_calls_map[tool_call.id] = tool_call
                        except (ValueError, SyntaxError):
                            # Ignore malformed tool call lists
                            pass
                
                # Add the assistant message, including any tool calls
                messages.append(
                    Message(
                        role="assistant",
                        content=thought,
                        timestamp=span.end_time,
                        tool_calls=parsed_tool_calls,
                    )
                )
                
            # Handle tool outputs (responses from tools)
            if "gen_ai.tool.call.id" in attrs:
                tool_call_id = attrs.get("gen_ai.tool.call.id")
                tool_output = attrs.get("gen_ai.response.content", "")

                # If the tool call was previously registered, attach the output
                if tool_call_id and tool_call_id in tool_calls_map:
                    tool_calls_map[tool_call_id].output = tool_output

                # Add the tool message to the conversation
                messages.append(
                    Message(
                        role="tool",
                        content=tool_output,
                        timestamp=span.end_time,
                        tool_call_id=tool_call_id,
                    )
                )

        # Sort all messages by timestamp to ensure correct conversation order
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)
        
        # Return the constructed Trajectory object
        return Trajectory(
            trace_id=trace_id,
            messages=sorted_messages,
            spans=sorted_spans,
        )

    def _convert_otel_attributes(
        self, attributes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Converts a list of OpenTelemetry attribute objects to a flat dictionary.

        Args:
            attributes (List[Dict[str, Any]]): List of OTel attribute objects.

        Returns:
            Dict[str, Any]: Flattened dictionary of attribute key-value pairs.

        This method handles various OTel value types, including strings, ints,
        doubles, booleans, arrays, and nested key-value lists.
        """
        if not attributes:
            return {}
        
        result = {}
        for attr in attributes:
            key = attr.get("key")
            value_obj = attr.get("value", {})
            if not key or not value_obj:
                continue
            
            # Handle different OTel value types
            if "stringValue" in value_obj:
                result[key] = value_obj["stringValue"]
            elif "intValue" in value_obj:
                try:
                    result[key] = int(value_obj["intValue"])
                except (ValueError, TypeError):
                    result[key] = value_obj["intValue"]
            elif "doubleValue" in value_obj:
                try:
                    result[key] = float(value_obj["doubleValue"])
                except (ValueError, TypeError):
                    result[key] = value_obj["doubleValue"]
            elif "boolValue" in value_obj:
                result[key] = value_obj["boolValue"]
            elif "arrayValue" in value_obj:
                result[key] = value_obj["arrayValue"]
            elif "kvlistValue" in value_obj:
                # Recursively convert nested key-value lists
                result[key] = self._convert_otel_attributes(value_obj.get('values', []))
            else:
                # Fallback: store the string representation of the value object
                result[key] = str(value_obj)
                
        return result