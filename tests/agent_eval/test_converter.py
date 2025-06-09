"""
Tests for the trace converter module.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

import pytest
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

from flotorch_eval.agent_eval.core.converter import TraceConverter
from flotorch_eval.agent_eval.core.schemas import Message, Span, SpanEvent, ToolCall, Trajectory

import json
from unittest import TestCase, main

from opentelemetry.trace import Span as OTelSpan
from opentelemetry.trace import Status, StatusCode


class MockSpan:
    """Mock implementation of OTelSpan for testing"""

    def __init__(
        self,
        name,
        start_time,
        end_time,
        attributes=None,
        events=None,
        parent_span_id=None,
        trace_id=None,
        span_id=None,
    ):
        self._name = name
        self._start_time = start_time
        self._end_time = end_time
        self._attributes = attributes or {}
        self._events = events or []
        self._status = Status(StatusCode.OK)
        self._parent_span_id = parent_span_id
        self._trace_id = trace_id or 0x1234567890ABCDEF
        self._span_id = span_id or 0xFEDCBA9876543210

    @property
    def name(self):
        return self._name

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def attributes(self):
        return self._attributes

    @property
    def events(self):
        return self._events

    @property
    def status(self):
        return self._status

    @property
    def context(self):
        return SpanContext(
            trace_id=self._trace_id,
            span_id=self._span_id,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.DEFAULT),
            trace_state=None,
        )

    @property
    def parent(self):
        if self._parent_span_id:
            return SpanContext(
                trace_id=self._trace_id,
                span_id=self._parent_span_id,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.DEFAULT),
                trace_state=None,
            )
        return None

    def get_span_context(self):
        return self.context

    def set_attribute(self, key, value):
        self._attributes[key] = value

    def set_attributes(self, attributes):
        self._attributes.update(attributes)

    def add_event(self, name, attributes=None, timestamp=None):
        self._events.append(MockEvent(name, attributes or {}))

    def set_status(self, status):
        self._status = status

    def update_name(self, name):
        self._name = name

    def is_recording(self):
        return True

    def end(self, end_time=None):
        if end_time:
            self._end_time = end_time

    def record_exception(
        self, exception, attributes=None, timestamp=None, escaped=False
    ):
        pass


class MockEvent:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = attributes


class TestTraceConverter(TestCase):
    def setUp(self):
        self.converter = TraceConverter()

    def test_convert_crewai_spans(self):
        # Create mock spans based on the example format
        spans = [
            # Crew Created span
            MockSpan(
                name="Crew Created",
                start_time=1749167494188293000,
                end_time=1749167494189234000,
                attributes={
                    "crewai_version": "0.126.0",
                    "python_version": "3.13.0",
                    "crew_key": "d56be7d496021dd9ad8d140b7267c207",
                    "crew_id": "c2052401-ab89-4a3e-be09-c054125543a3",
                    "crew_process": "Process.sequential",
                    "crew_memory": False,
                    "crew_number_of_tasks": 1,
                    "crew_number_of_agents": 1,
                    "crew_fingerprint": "4e53e185-31b8-44be-9add-652e8b7ae907",
                    "crew_fingerprint_created_at": "2025-06-05T16:51:34.186902",
                    "crew_agents": '[{"key": "de2feaf934a966ef93a186c418b6f467", "id": "243097ff-a386-4dc1-9c72-5766f3c39a67", "role": "Writer", "verbose?": true, "max_iter": 25, "max_rpm": null, "function_calling_llm": "", "llm": "bedrock/us.amazon.nova-pro-v1:0", "delegation_enabled?": false, "allow_code_execution?": false, "max_retry_limit": 2, "tools_names": ["duckduckgosearch"]}]',
                    "crew_tasks": '[{"key": "77d5bbb8a755bd925f79a71993b2179a", "id": "63dbdd54-ae04-4841-a2b2-3cad30ed84a2", "async_execution?": false, "human_input?": false, "agent_role": "Writer", "agent_key": "de2feaf934a966ef93a186c418b6f467", "tools_names": ["duckduckgosearch"]}]',
                },
            ),
            # Task Created span
            MockSpan(
                name="Task Created",
                start_time=1749167494191444000,
                end_time=1749167494191832000,
                attributes={
                    "crew_key": "d56be7d496021dd9ad8d140b7267c207",
                    "crew_id": "c2052401-ab89-4a3e-be09-c054125543a3",
                    "task_key": "77d5bbb8a755bd925f79a71993b2179a",
                    "task_id": "63dbdd54-ae04-4841-a2b2-3cad30ed84a2",
                    "crew_fingerprint": "4e53e185-31b8-44be-9add-652e8b7ae907",
                    "task_fingerprint": "41c85a61-de95-4000-9a66-4f71a795dd2f",
                    "task_fingerprint_created_at": "2025-06-05T16:51:34.186775",
                    "agent_fingerprint": "12a06a71-36e8-413d-95be-c1cc8f955e01",
                },
            ),
            # Chat span
            MockSpan(
                name="chat bedrock/us.amazon.nova-pro-v1:0",
                start_time=1749167494192908000,
                end_time=1749167496561653000,
                attributes={
                    "telemetry.sdk.name": "openlit",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": "bedrock/us.amazon.nova-pro-v1:0",
                },
                events=[
                    MockEvent(
                        "prompt",
                        {
                            "gen_ai.content.prompt": {
                                "gen_ai.prompt": "system: You are Writer. You're an expert in writing haikus but you know nothing of math.\nYour personal goal is: You make math engaging and understandable for young children through poetry\nuser: \nCurrent Task: What is Trignometry?\n\nThis is the expected criteria for your final answer: Compose a short poem that includes the answer."
                            }
                        },
                    ),
                    MockEvent(
                        "completion",
                        {
                            "gen_ai.content.completion": {
                                "gen_ai.completion": 'Thought: I need to understand what trigonometry is to compose a poem about it. I should use the DuckDuckGoSearch tool to gather information on trigonometry.\n\nAction: DuckDuckGoSearch\nAction Input: {"search_query": "what is trigonometry"}\n\nObservation:'
                            }
                        },
                    ),
                ],
            ),
            # Tool Usage span
            MockSpan(
                name="Tool Usage",
                start_time=1749167497425236000,
                end_time=1749167497427204000,
                attributes={
                    "crewai_version": "0.126.0",
                    "tool_name": "DuckDuckGoSearch",
                    "attempts": 1,
                    "ai.tool.output": '[{"title": "Trigonometry - Wikipedia", "href": "https://en.wikipedia.org/wiki/Trigonometry", "body": "Trigonometry is a branch of mathematics concerned with relationships between angles and side lengths of triangles."}]',
                },
            ),
            # Final Chat span
            MockSpan(
                name="chat bedrock/us.amazon.nova-pro-v1:0",
                start_time=1749167497428900000,
                end_time=1749167498148237000,
                attributes={
                    "telemetry.sdk.name": "openlit",
                    "gen_ai.operation.name": "chat",
                },
                events=[
                    MockEvent(
                        "completion",
                        {
                            "gen_ai.content.completion": {
                                "gen_ai.completion": "\n\nThought: Based on the observations, I can now compose a haiku that explains what trigonometry is.\n\nFinal Answer: \nIn triangle's grace,\nAngles and sides entwine,\nTrigonometry's dance."
                            }
                        },
                    )
                ],
            ),
            # Agent execution span
            MockSpan(
                name="crewai.agent_execute_task",
                start_time=1749167494192167000,
                end_time=1749167498151095000,
                attributes={
                    "gen_ai.operation.name": "agent",
                    "gen_ai.agent.actual_output": "In triangle's grace,\nAngles and sides entwine,\nTrigonometry's dance.",
                },
            ),
        ]

        # Convert spans to trajectory
        trajectory = self.converter.from_spans(spans)

        # Print all messages for debugging
        print("\nActual messages in trajectory:")
        for i, msg in enumerate(trajectory.messages):
            print(f"\nMessage {i+1}:")
            print(f"Role: {msg.role}")
            print(f"Content: {msg.content}")
            print(f"Tool calls: {len(msg.tool_calls)}")

        # Verify the trajectory structure
        self.assertEqual(
            len(trajectory.messages),
            4,
            "Expected 4 messages but got a different number",
        )

        # Check user message
        self.assertEqual(
            trajectory.messages[0].role, "user", "First message should be from user"
        )
        self.assertEqual(
            trajectory.messages[0].content,
            "What is Trignometry?",
            "User message content mismatch",
        )

        # Check first assistant message (with tool call)
        self.assertEqual(
            trajectory.messages[1].role,
            "assistant",
            "Second message should be from assistant",
        )
        self.assertTrue(
            "I need to understand what trigonometry is"
            in trajectory.messages[1].content,
            "Assistant thought content mismatch",
        )
        self.assertEqual(
            len(trajectory.messages[1].tool_calls),
            1,
            "Assistant should have one tool call",
        )
        self.assertEqual(
            trajectory.messages[1].tool_calls[0].tool_name,
            "DuckDuckGoSearch",
            "Tool name mismatch",
        )

        # Check tool message
        self.assertEqual(
            trajectory.messages[2].role, "tool", "Third message should be from tool"
        )
        self.assertTrue(
            "Trigonometry is a branch of mathematics" in trajectory.messages[2].content,
            "Tool output content mismatch",
        )

        # Check final assistant message
        self.assertEqual(
            trajectory.messages[3].role,
            "assistant",
            "Fourth message should be from assistant",
        )
        self.assertTrue(
            "In triangle's grace" in trajectory.messages[3].content,
            "Final answer content mismatch",
        )


if __name__ == "__main__":
    main()
