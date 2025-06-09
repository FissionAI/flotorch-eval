# FlotorchEval

A comprehensive evaluation framework for AI systems.

## Features

- **Agent Evaluation**: Evaluate AI agents using OpenTelemetry traces
  - Convert traces to structured trajectories
  - Multiple evaluation metrics (LangChain, Ragas, custom)
  - Tool usage analysis
  - Extensible metric system

## Installation

Install the base package:
```bash
pip install flotorch-eval
```

Install with agent evaluation support:
```bash
pip install "flotorch-eval[agent]"
```

Install with development tools:
```bash
pip install "flotorch-eval[dev]"
```

Install everything:
```bash
pip install "flotorch-eval[all]"
```

## Quick Start

### Agent Evaluation

```python
from flotorch_eval.agent_eval import TraceConverter, Evaluator
from flotorch_eval.agent_eval.metrics import TrajectoryEvalWithLLMMetric

# Convert OpenTelemetry traces to trajectories
converter = TraceConverter()
trajectory = converter.from_spans(spans)

# Create evaluator with metrics
evaluator = Evaluator([
    TrajectoryEvalWithLLMMetric(
        reference_trajectory=reference,
        llm=llm
    )
])

# Run evaluation
results = evaluator.evaluate(trajectory)
```

## Documentation

For detailed documentation, visit [docs.flotorch.ai](https://docs.flotorch.ai).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

