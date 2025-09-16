"""
Converter module for transforming OpenTelemetry traces into agent trajectories.

This module provides the TraceConverter class, which is responsible for parsing
OpenTelemetry trace data and converting it into a
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
        to reconstruct the agent's conversation flow. It primarily uses the final
        LLM call's attributes, which contain the full conversation history, to
        build the message list accurately and avoid duplication.
        """
        resource_spans = trace_data.get("resourceSpans", [])
        if not resource_spans:
            return Trajectory(trace_id="", messages=[], spans=[])

        raw_spans = []
        for rs in resource_spans:
            for ss in rs.get("scopeSpans", []):
                raw_spans.extend(ss.get("spans", []))

        if not raw_spans:
            return Trajectory(trace_id="", messages=[], spans=[])
            
        trace_id = raw_spans[0].get("traceId", "")

        internal_spans: List[Span] = []
        for span_dict in raw_spans:
            events = [
                SpanEvent(
                    name=evt.get("name", ""),
                    timestamp=datetime.fromtimestamp(int(evt.get("timeUnixNano", 0)) / 1e9),
                    attributes=self._convert_otel_attributes(evt.get("attributes", [])),
                )
                for evt in span_dict.get("events", [])
            ]
            
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

        sorted_spans = sorted(internal_spans, key=lambda s: s.start_time)
        
        messages: List[Message] = []
        tool_calls_map: Dict[str, ToolCall] = {}

        # Find all LLM call spans that have conversation history
        llm_spans = [
            s for s in sorted_spans
            if "gen_ai.request.messages" in s.attributes and s.attributes.get("gen_ai.operation.name") in ("chat", "invoke_agent")
        ]

        if not llm_spans:
            return Trajectory(trace_id=trace_id, messages=[], spans=sorted_spans)

        # The last LLM span contains the most complete history and the final response
        last_llm_span = llm_spans[-1]

        # 1. Process the conversation history from the last LLM call's request
        history_messages_str = last_llm_span.attributes.get("gen_ai.request.messages")
        if history_messages_str:
            try:
                history_messages_data = ast.literal_eval(history_messages_str)
                for msg_data in history_messages_data:
                    self._parse_and_append_message(
                        msg_data,
                        messages,
                        tool_calls_map,
                        # Use the start time of the span as an approximation for historical message timestamps
                        timestamp=last_llm_span.start_time
                    )
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse conversation history: {e}")
                pass

        # 2. Process the final response from the last LLM call span
        final_content = last_llm_span.attributes.get("gen_ai.response.content", "")
        final_tool_calls: List[ToolCall] = self._parse_tool_calls_from_response(last_llm_span)

        # Link any newly created tool calls for potential output processing
        for tc in final_tool_calls:
            if tc.id:
                tool_calls_map[tc.id] = tc

        # Add the final assistant message
        if final_content or final_tool_calls:
            messages.append(Message(
                role="assistant",
                content=final_content,
                timestamp=last_llm_span.end_time,
                tool_calls=final_tool_calls,
            ))

        # Sort all messages by timestamp to ensure correct conversation order
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)

        return Trajectory(
            trace_id=trace_id,
            messages=sorted_messages,
            spans=sorted_spans,
        )

    def _parse_and_append_message(
        self,
        msg_data: Dict[str, Any],
        messages: List[Message],
        tool_calls_map: Dict[str, ToolCall],
        timestamp: datetime
    ):
        """Parses a single message dictionary and updates the messages list and tool map."""
        role = msg_data.get("role")
        content = msg_data.get("content", "")

        if role in ("user", "system"):
            messages.append(Message(role=role, content=content, timestamp=timestamp))

        elif role == "assistant":
            parsed_tool_calls = []
            if "tool_calls" in msg_data:
                for tc_data in msg_data["tool_calls"]:
                    function_data = tc_data.get("function", {})

                    try:
                        # Arguments can be a stringified dict/json
                        arguments = ast.literal_eval(function_data.get("arguments", "{}"))
                    except (ValueError, SyntaxError):
                        arguments = {"raw": str(function_data.get("arguments"))}

                    tool_call = ToolCall(
                        id=tc_data.get("id"),
                        name=function_data.get("name", ""),
                        arguments=arguments,
                        timestamp=timestamp,
                    )
                    parsed_tool_calls.append(tool_call)
                    if tool_call.id:
                        tool_calls_map[tool_call.id] = tool_call

            messages.append(Message(
                role="assistant", content=content, timestamp=timestamp, tool_calls=parsed_tool_calls
            ))

        elif role == "tool":
            tool_call_id = msg_data.get("tool_call_id")

            try:
                # Tool output might be a stringified dict with a 'result' key
                tool_output_dict = ast.literal_eval(content)
                tool_output = tool_output_dict.get("result", content)
            except (ValueError, SyntaxError):
                tool_output = content

            if tool_call_id and tool_call_id in tool_calls_map:
                tool_calls_map[tool_call_id].output = tool_output

            messages.append(Message(
                role="tool", content=tool_output, timestamp=timestamp, tool_call_id=tool_call_id
            ))

    def _parse_tool_calls_from_response(self, span: Span) -> List[ToolCall]:
        """Extracts tool calls from the 'gen_ai.response.full' attribute of a span."""
        full_response_str = span.attributes.get("gen_ai.response.full", "")
        if not full_response_str:
            return []

        parsed_tool_calls = []
        # Use regex to find the tool_calls list in the string representation
        match = re.search(r"'tool_calls':\s*(\[.*?\])", full_response_str.replace('\\', ''))
        if match:
            tool_calls_repr = match.group(1)
            try:
                tool_calls_data = ast.literal_eval(tool_calls_repr)
                for tc_data in tool_calls_data:
                    function_data = tc_data.get("function", {})
                    arguments_raw = function_data.get("arguments", {})

                    # Arguments can be a dict or a stringified dict
                    arguments = arguments_raw if isinstance(arguments_raw, dict) else ast.literal_eval(str(arguments_raw))

                    tool_call = ToolCall(
                        id=tc_data.get("id"),
                        name=function_data.get("name", ""),
                        arguments=arguments,
                        timestamp=span.end_time,
                    )
                    parsed_tool_calls.append(tool_call)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse tool calls from full response: {e}")
                pass
        return parsed_tool_calls

    def _convert_otel_attributes(
        self, attributes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Converts a list of OpenTelemetry attribute objects to a flat dictionary.
        """
        if not attributes:
            return {}
        
        result = {}
        for attr in attributes:
            key = attr.get("key")
            value_obj = attr.get("value", {})
            if not key or not value_obj:
                continue
            
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
                result[key] = self._convert_otel_attributes(value_obj.get('values', []))
            else:
                result[key] = str(value_obj)
                
        return result