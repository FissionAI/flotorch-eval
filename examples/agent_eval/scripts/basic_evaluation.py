"""
Basic example demonstrating how to use the agent-eval library.
"""

from datetime import datetime, timedelta

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from ragas.metrics import ContextualRelevancy

from agent_eval import Evaluator, TraceConverter
from agent_eval.integrations.ragas_utils import RagasMetricWrapper
from agent_eval.metrics import ToolAccuracyMetric


def create_sample_trace():
    """Create a sample trace for demonstration."""
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("message.human") as human_span:
        human_span.set_attribute("content", "What's the weather in London?")
        human_span.set_status(Status(StatusCode.OK))

        with tracer.start_as_current_span("message.assistant") as assistant_span:
            assistant_span.set_attribute("content", "Let me check the weather for you.")

            with tracer.start_as_current_span("tool.weather_api") as tool_span:
                tool_span.set_attribute("inputs", {"city": "London"})
                tool_span.set_attribute(
                    "outputs", {"temperature": 20, "condition": "sunny"}
                )
                tool_span.set_attribute("success", True)

            assistant_span.set_attribute(
                "content",
                "The weather in London is currently sunny with a temperature of 20°C.",
            )
            assistant_span.set_status(Status(StatusCode.OK))


def main():
    # Create sample trace
    create_sample_trace()

    # Get the trace
    tracer = trace.get_tracer(__name__)
    spans = []  # In real usage, you'd get this from your OpenTelemetry setup

    # Convert trace to trajectory
    converter = TraceConverter()
    trajectory = converter.from_spans(spans)

    # Set up evaluator with metrics
    evaluator = Evaluator(
        [ToolAccuracyMetric(), RagasMetricWrapper(ContextualRelevancy())]
    )

    # Evaluate trajectory
    results = evaluator.evaluate(trajectory)

    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for score in results.scores:
        print(f"\nMetric: {score.name}")
        print(f"Score: {score.score:.2f}")
        if score.details:
            print("Details:")
            for key, value in score.details.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
