# FlotorchEval

**FlotorchEval** is a comprehensive evaluation framework for AI systems. It enables analysis of LLM agents using OpenTelemetry traces, supports multiple evaluation metrics (including LangChain, Ragas, and custom metrics), and provides tooling for advanced cost and usage analysis.

---

## 📦 Features

- **Agent Evaluation**: Evaluate LLM agents using structured trajectories
- **Metrics Support**:
  - LangChain metrics
  - RAGAS metrics
  - Custom cost, usage, and goal accuracy metrics
- **Trace Conversion**: Convert OpenTelemetry traces to evaluation-ready formats
- **Cost & Token Tracking**: Calculate cost and token usage across models

---

## 🧰 Installation

Install the base package:

```bash
pip install flotorch-eval

# With agent evaluation support:
pip install "flotorch-eval[agent]"

# With development tools:
pip install "flotorch-eval[dev]"

# Install everything:
pip install "flotorch-eval[all]"
```

## Quick Start – Agent Evaluation

```bash
from flotorch_eval.agent_eval import TraceConverter, Evaluator
from flotorch_eval.agent_eval.metrics import (
    TrajectoryEvalWithLLMMetric,
    TrajectoryEvalWithoutLLMMetric,
    ToolCallAccuracyMetric,
    AgentGoalAccuracyMetric,
)
from flotorch_eval.agent_eval.metrics.base import MetricConfig

# Convert OpenTelemetry spans to trajectory
converter = TraceConverter()
trajectory = converter.from_spans(spans)

# Setup evaluator with multiple metrics
evaluator = Evaluator([
    TrajectoryEvalWithLLMMetric(
        llm=llm,
        config=MetricConfig(metric_params={"reference_trajectory": reference})
    ),
    TrajectoryEvalWithoutLLMMetric(
        config=MetricConfig(metric_params={"reference_trajectory": reference})
    ),
    ToolCallAccuracyMetric(),
    AgentGoalAccuracyMetric(
        llm=llm,
        config=MetricConfig(metric_params={
            "reference_answer": "Amazon Bedrock is a fully managed service that makes it easy to use foundation models from third-party providers and Amazon."
        })
    )
])

# Run evaluation
results = await evaluator.evaluate(trajectory)

results = await evaluator.evaluate(trajectory)
```

## Documentation
Full documentation is available at https://docs.flotorch.ai

## Contributing
We welcome contributions! Please see our CONTRIBUTING.md for guidelines.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
