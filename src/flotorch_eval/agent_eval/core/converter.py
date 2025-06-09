"""
Converter module for transforming OpenTelemetry traces into agent trajectories.
"""

from datetime import datetime
import ast
import json
import re
from typing import Dict, List, Optional, Union

from opentelemetry.trace import Span as OTelSpan
from opentelemetry.trace import SpanKind

from flotorch_eval.agent_eval.core.schemas import Message, Span, SpanEvent, ToolCall, Trajectory
from flotorch_eval.common.utils import convert_attributes


class TraceConverter:
    """Converts OpenTelemetry traces into agent trajectories using standardized conventions."""

    def from_spans(self, spans: List[OTelSpan]) -> Trajectory:
        sorted_spans = sorted(spans, key=lambda x: x.start_time)
        internal_spans = []

        # First convert all spans to our internal format
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
        current_tool_calls = []
        pending_tool_messages = (
            []
        )  # Store tool messages until we have an assistant message
        final_answer_added = False
        has_assistant_message = False

        # Process spans to build the conversation
        for span in internal_spans:
            if span.name.startswith("chat") or span.attributes.get(
                "gen_ai.operation.name"
            ) in ["chat", "completion"]:
                # Process LLM interaction span
                prompt = self._extract_prompt_from_events(span)
                completion = self._extract_completion_from_events(span)

                if prompt:
                    user_content = self._extract_user_content_from_prompt(prompt)
                    if user_content and not any(m.role == "user" for m in messages):
                        messages.append(
                            Message(
                                role="user",
                                content=user_content,
                                timestamp=span.start_time,
                                tool_calls=[],
                            )
                        )

                if completion:
                    tool_calls, thought = self._parse_assistant_output(
                        completion, span.start_time
                    )
                    if thought:
                        # Check if this is a final answer
                        if "Final Answer:" in completion:
                            if not final_answer_added:
                                final_answer = completion.split("Final Answer:", 1)[
                                    1
                                ].strip()
                                # Remove any trailing JSON artifacts
                                final_answer = final_answer.rstrip('"}')
                                messages.append(
                                    Message(
                                        role="assistant",
                                        content=final_answer,
                                        timestamp=span.end_time,
                                        tool_calls=[],
                                    )
                                )
                                final_answer_added = True
                                has_assistant_message = True
                        else:
                            # Remove any trailing JSON artifacts
                            thought = thought.rstrip('"}')
                            messages.append(
                                Message(
                                    role="assistant",
                                    content=thought,
                                    timestamp=span.start_time,
                                    tool_calls=tool_calls,
                                )
                            )
                            current_tool_calls.extend(tool_calls)
                            has_assistant_message = True

                            # Add any pending tool messages now that we have an assistant message
                            if has_assistant_message and pending_tool_messages:
                                messages.extend(pending_tool_messages)
                                pending_tool_messages = []

            elif span.name == "Tool Usage" or span.attributes.get("gen_ai.agent.tools"):
                # Process tool usage span
                tool_name = None
                tool_output = ""

                # Try to get tool name from tool definition
                if "gen_ai.agent.tools" in span.attributes:
                    try:
                        # Handle single-quoted string using ast.literal_eval first
                        tools_str = span.attributes["gen_ai.agent.tools"]
                        if isinstance(tools_str, str):
                            tools = ast.literal_eval(tools_str)
                            if tools and isinstance(tools, list) and len(tools) > 0:
                                tool_name = tools[0].get("name")
                    except (ValueError, SyntaxError, AttributeError):
                        pass

                # Get tool output from new format
                if "gen_ai.agent.tool_results" in span.attributes:
                    try:
                        # Handle single-quoted string using ast.literal_eval first
                        results_str = span.attributes["gen_ai.agent.tool_results"]
                        if isinstance(results_str, str):
                            results = ast.literal_eval(results_str)
                            if (
                                results
                                and isinstance(results, list)
                                and len(results) > 0
                            ):
                                tool_output = results[0].get("result", "")
                    except (ValueError, SyntaxError, AttributeError):
                        pass

                if tool_output and tool_name:
                    # Remove any trailing JSON artifacts
                    tool_output = tool_output.rstrip('"}')
                    tool_message = Message(
                        role="tool",
                        content=tool_output,
                        timestamp=span.end_time,
                        tool_calls=[],
                    )

                    # Only add tool message immediately if we have an assistant message
                    if has_assistant_message:
                        messages.append(tool_message)
                    else:
                        pending_tool_messages.append(tool_message)

                    # Update the corresponding tool call with the output
                    for msg in messages:
                        for tool_call in msg.tool_calls:
                            if tool_call.name == tool_name:
                                tool_call.output = tool_output

        return Trajectory(
            trace_id=format(spans[0].context.trace_id, "032x") if spans else "",
            messages=messages,
            spans=internal_spans,
        )

    def _convert_attributes(
        self, attributes: Dict[str, Union[str, int, float, bool, List[str]]]
    ) -> Dict[str, Union[str, int, float, bool, List[str]]]:
        """Convert span attributes to our internal format."""
        result = {}
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

    def _extract_prompt_from_events(self, span: Span) -> Optional[str]:
        """Extract prompt from span events."""
        for event in span.events:
            if "gen_ai.content.prompt" in event.name:
                prompt_data = event.attributes["gen_ai.prompt"]
                if isinstance(prompt_data, dict):
                    return prompt_data.get("gen_ai.prompt", "")
                return prompt_data
        return None

    def _extract_completion_from_events(self, span: Span) -> Optional[str]:
        """Extract completion from span events."""
        for event in span.events:
            if "gen_ai.content.completion" in event.name:
                completion_data = event.attributes["gen_ai.completion"]
                if isinstance(completion_data, dict):
                    return completion_data.get("gen_ai.completion", "")
                return completion_data
        return None

    def _parse_assistant_output(
        self, completion: str, timestamp: datetime
    ) -> tuple[List[ToolCall], Optional[str]]:
        """Parse the assistant's output to extract tool calls and thought."""
        tool_calls = []
        thought = None

        # Clean up completion string
        completion = completion.strip()
        if completion.startswith('"') and completion.endswith('"'):
            completion = completion[1:-1]

        # Try to parse as JSON if it looks like JSON
        try:
            if completion.startswith("{"):
                data = json.loads(completion)
                if isinstance(data, dict) and "gen_ai.completion" in data:
                    completion = data["gen_ai.completion"]
        except (json.JSONDecodeError, TypeError):
            pass

        # Extract thought if present
        thought_match = re.search(
            r"Thought:(.*?)(?=\nAction:|Final Answer:|$)", completion, re.DOTALL
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract tool calls
        action_pattern = r"Action:\s*(.*?)\nAction Input:\s*(.*?)(?=\n(?:Thought:|Action:|Final Answer:|Observation:|$)|\Z)"
        action_matches = list(re.finditer(action_pattern, completion, re.DOTALL))

        for match in action_matches:
            tool_name = match.group(1).strip()
            tool_input = match.group(2).strip()

            # Clean up tool input
            if tool_input.startswith('"') and tool_input.endswith('"'):
                tool_input = tool_input[1:-1]

            try:
                tool_args = json.loads(tool_input)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the JSON part
                json_match = re.search(r"\{.*\}", tool_input)
                if json_match:
                    try:
                        tool_args = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        tool_args = {"query": tool_input}
                else:
                    tool_args = {"query": tool_input}

            tool_calls.append(
                ToolCall(
                    name=tool_name,
                    arguments=tool_args,
                    output=None,
                )
            )

        return tool_calls, thought

    def _extract_user_content_from_prompt(self, prompt: str) -> str:
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
