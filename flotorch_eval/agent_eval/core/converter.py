"""
Converter module for transforming OpenTelemetry traces into agent trajectories.

This module provides the TraceConverter class, which is responsible for parsing
OpenTelemetry trace data and converting it into a
structured Trajectory object suitable for downstream agent evaluation and analysis.

The conversion process extracts relevant span information, reconstructs the
conversation flow, and handles tool call associations.
"""

from datetime import datetime
import json
from typing import Dict, List, Any, Optional
from flotorch_eval.agent_eval.core.schemas import (
    Message,
    Span,
    SpanEvent,
    ToolCall,
    Trajectory,
    ReferenceTrajectory,
    ReferenceStep,
    ReferenceToolCall
)

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

        # Iterate through all spans and their events to build the conversation chronologically
        for span in sorted_spans:
            # Sort events within the span to ensure correct order
            sorted_events = sorted(span.events, key=lambda e: e.timestamp)
            for event in sorted_events:
                msg = self._parse_message_from_event(event, tool_calls_map)
                if msg:
                    messages.append(msg)

        # Sort all messages by timestamp to ensure correct final conversation order
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)

        merged_messages: List[Message] = []
        i = 0
        while i < len(sorted_messages):
            current_msg = sorted_messages[i]

            if (current_msg.role == 'assistant' and
                current_msg.content and
                not current_msg.tool_calls and
                (i + 1) < len(sorted_messages)):
                
                next_msg = sorted_messages[i+1]

                if (next_msg.role == 'assistant' and
                    (not next_msg.content or next_msg.content == "") and
                    next_msg.tool_calls and
                    (next_msg.timestamp - current_msg.timestamp).total_seconds() < 0.1):
                    
                    current_msg.tool_calls = next_msg.tool_calls

                    merged_messages.append(current_msg)
                    i += 2
                    continue
            merged_messages.append(current_msg)
            i += 1

        return Trajectory(
            trace_id=trace_id,
            messages=merged_messages,
            spans=sorted_spans,
        )

    def _parse_message_from_event(
        self,
        event: SpanEvent,
        tool_calls_map: Dict[str, ToolCall]
    ) -> Optional[Message]:
        """Parses a single SpanEvent to create a Message object if applicable."""
        
        attrs = event.attributes
        role = attrs.get("message.role")
        content = attrs.get("message.content", "")
        timestamp = event.timestamp
        
        # 1. User Message
        if role == "user":
            return Message(role="user", content=content, timestamp=timestamp)

        # 2. Assistant Message (can be text or tool call)
        if event.name == "event_choice":
            parsed_tool_calls = []
            tool_calls_str = attrs.get("message.tool_calls")
            
            if tool_calls_str:
                try:
                    tool_calls_data = json.loads(tool_calls_str)
                    for tc_data in tool_calls_data:
                        function_data = tc_data.get("function", {})
                        arguments = function_data.get("arguments", {})
                        
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {"raw": arguments}
                        
                        tool_call = ToolCall(
                            id=tc_data.get("id"),
                            name=function_data.get("name", ""),
                            arguments=arguments,
                            timestamp=timestamp,
                        )
                        parsed_tool_calls.append(tool_call)
                        if tool_call.id:
                            tool_calls_map[tool_call.id] = tool_call
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Warning: Could not parse tool calls from event: {e}")
            
            if not content and not parsed_tool_calls:
                return None

            return Message(
                role="assistant",
                content=content, 
                timestamp=timestamp, 
                tool_calls=parsed_tool_calls or None
            )

        # 3. Tool Message (output from a tool)
        if role == "tool":
            tool_call_id = attrs.get("tool.call.id")
            
            try:
                tool_output_dict = json.loads(content)
                tool_output = tool_output_dict.get("result", content)
            except (json.JSONDecodeError, TypeError):
                tool_output = content

            if tool_call_id and tool_call_id in tool_calls_map:
                tool_calls_map[tool_call_id].output = str(tool_output)

            return Message(
                role="tool", 
                content=str(tool_output), 
                timestamp=timestamp, 
                tool_call_id=tool_call_id
            )
            
        return None


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
                result[key] = [v.get('stringValue') for v in value_obj["arrayValue"].get('values', [])]
            elif "kvlistValue" in value_obj:
                result[key] = self._convert_otel_attributes(value_obj.get('values', []))
            else:
                result[key] = str(value_obj)
                
        return result
    
    def to_reference(self, trace_data: Dict[str, Any]) -> ReferenceTrajectory:
        """
        Converts a full trace data object into a detailed ReferenceTrajectory,
        including a generated "thought" for each step.
        (This method does not need changes as it depends on the output of from_spans)
        """
        trajectory = self.from_spans(trace_data)

        if not trajectory.messages:
            raise ValueError("Cannot create a reference from a trace with no messages.")

        initial_input = ""
        for msg in trajectory.messages:
            if msg.role == 'user':
                initial_input = msg.content
                break
        
        if not initial_input:
             raise ValueError("Cannot create a reference from a trace with no user input.")

        steps: List[ReferenceStep] = []
        
        # Collect tool call steps
        for msg in trajectory.messages:
            if msg.role == 'assistant' and msg.tool_calls:
                # Add assistant's thought/content if it exists before the tool call
                thought = msg.content or "The agent determined it needed to use a tool."
                for tc in msg.tool_calls:
                    steps.append(
                        ReferenceStep(
                            thought=thought,
                            tool_call=ReferenceToolCall(name=tc.name, arguments=tc.arguments)
                        )
                    )

        # Find and add the final response step
        for msg in reversed(trajectory.messages):
            if msg.role == 'assistant' and msg.content and not msg.tool_calls:
                thought = "The agent synthesized the available information to formulate a final answer."
                steps.append(
                    ReferenceStep(
                        thought=thought,
                        final_response=msg.content
                    )
                )
                break

        if not steps:
            raise ValueError("Could not extract any meaningful steps (tool calls or final response) from the trace.")

        return ReferenceTrajectory(
            input=initial_input,
            expected_steps=steps,
        )