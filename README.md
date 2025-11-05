# FlotorchEval

**FlotorchEval** is a comprehensive evaluation framework for AI systems. It enables evaluation of both LLM outputs using industry-standard metrics from DeepEval and Ragas and agent behaviors using our custom metrics, with support for OpenTelemetry traces, and advanced cost/usage analysis.

---

## 📦 Features

### LLM Evaluation
- **Multi-Engine Support**: DeepEval and Ragas metrics with automatic engine selection
- **RAG Metrics**: Faithfulness, context relevancy, context precision, context recall, answer relevance, and hallucination detection
- **Flexible Architecture**: Pluggable metric system with configurable thresholds
- **Priority-Based Routing**: Automatic metric-to-engine mapping based on priority

### Agent Evaluation
- **Custom Evaluation Framework**: Purpose-built evaluation system for agent trajectories
- **Trajectory Analysis**: Evaluate agent behavior using OpenTelemetry traces
- **LLM-Based Metrics**: Trajectory evaluation with and without reference comparisons
- **Goal Accuracy**: Measure if agent achieves intended goals
- **Tool Call Tracking**: Analyze tool usage and accuracy
- **Latency & Cost Metrics**: Track performance and resource usage

---

## 🧰 Installation

Install the base package:

```bash
pip install flotorch-eval

# With llm evaluation support:
pip install "flotorch-eval[llm]"

# With agent evaluation support:
pip install "flotorch-eval[agent]"

# With development tools:
pip install "flotorch-eval[dev]"

# Install everything:
pip install "flotorch-eval[all]"
```

---

## 🚀 Quick Start

### LLM Evaluation

Evaluate RAG system outputs using DeepEval and Ragas metrics:

```python
from flotorch_eval.llm_eval import LLMEvaluator, EvaluationItem, MetricKey

# Initialize evaluator
evaluator = LLMEvaluator(
    api_key="your-api-key",
    base_url="flotorch-base-url",
    evaluator_llm="flotorch/inference_model",
    embedding_model="flotorch/embedding_model"
)

# Prepare evaluation data
data = [
    EvaluationItem(
        question="What is machine learning?",
        generated_answer="Machine learning is a subset of AI...",
        expected_answer="Machine learning is a method of data analysis...",
        context=["Machine learning (ML) is a field of artificial intelligence..."]
    )
]

# Evaluate with specific metrics
results = evaluator.evaluate(
    data=data,
    metrics=[
        MetricKey.FAITHFULNESS,
        MetricKey.ANSWER_RELEVANCE,
        MetricKey.CONTEXT_RELEVANCY
    ]
)

print(results)
```
Passing the metrics is optional. If metrics are not provided, data will be evaluated on all the available metrics from both the evaluation engines.
The supported evaluation engines and metrics are given further below as you scroll down.

Note: Aspect critique requires a configuration to be passed as a metric configuration to the LLMEvaluation for it to work and so it will not be be added as default. The configuration structure will be further below as you scroll down.
### Advanced LLM Evaluation with Custom Thresholds

Deepeval metrics can be configured with specific threshold values which directly affects the score. 
The default score is 0.7.

```python
# Configure metric-specific arguments
metric_args = {
    "faithfulness": {"threshold": 0.8},
    "answer_relevance": {"threshold": 0.7},
    "hallucination": {"threshold": 0.3}
}

evaluator = LLMEvaluator(
    api_key="your-api-key",
    base_url="flotorch-base-url",
    evaluator_llm="flotorch/inference_model",
    embedding_model="flotorch/embedding_model",
    metric_args=metric_args
)

# Get all available metrics
available_metrics = evaluator.get_all_metrics()
print(f"Available metrics: {available_metrics}")

# Evaluate with all available metrics
results = evaluator.evaluate(data=data)
```

### Engine Selection Modes

Flotorch currently supports two evaluation backend engines: Ragas and Deepeval. Each offers distinct as well as overlapping metrics, and you can choose how you want to run them.
FlotorchEval supports flexible engine selection for LLM evaluation:

#### Auto Mode (Default - Recommended)

Automatically routes metrics to the best engine with priority-based selection:

```python
evaluator = LLMEvaluator(
    api_key="your-api-key",
    base_url="flotorch-base-url",
    evaluator_llm="flotorch/inference_model",
    embedding_model="flotorch/embedding_model",
    evaluation_engine='auto'  # Default behavior
)

# Automatically routes metrics to appropriate engines
# Ragas has priority for overlapping metrics (faithfulness, answer_relevance, context_precision)
results = evaluator.evaluate(data=data)
```

**How Auto Mode Works:**
- Metrics supported by multiple engines are routed to **Ragas** (priority 1)
- Metrics unique to an engine use that specific engine
- Example: `FAITHFULNESS` → Ragas, `HALLUCINATION` → DeepEval

#### Ragas-Only Mode

Use only Ragas metrics:

```python
evaluator = LLMEvaluator(
    api_key="your-api-key",
    base_url="flotorch-base-url",
    evaluator_llm="flotorch/inference_model",
    embedding_model="flotorch/embedding_model",
    evaluation_engine='ragas'
)

# Only Ragas metrics will be evaluated
# Metrics: faithfulness, answer_relevance, context_precision, aspect_critic
```

#### DeepEval-Only Mode

Use only DeepEval metrics:

```python
evaluator = LLMEvaluator(
    api_key="your-api-key",
    base_url="flotorch-base-url",
    evaluator_llm="flotorch/inference_model",
    embedding_model="flotorch/embedding_model",
    evaluation_engine='deepeval'
)

# Only DeepEval metrics will be evaluated
# Metrics: faithfulness, answer_relevance, context_relevancy, context_precision, 
#          context_recall, hallucination
```

#### Engine Priority

When using `auto` mode, metrics are routed based on priority:

| Priority | Engine | Overlapping Metrics |
|----------|--------|-------------------|
| 1 (Highest) | Ragas | faithfulness, answer_relevance, context_precision |
| 2 | DeepEval | faithfulness, answer_relevance, context_precision |

**Note:** Ragas requires an embedding model for most metrics. If no embedding model is provided, only DeepEval metrics will be available.

### Agent Evaluation

Evaluate agent trajectories using the FlotorchEvalClient:

```python
from flotorch_eval.agent_eval.core.client import FlotorchEvalClient

# Initialize the evaluation client
client = FlotorchEvalClient(
    api_key="your-api-key",
    base_url="flotorch-base-url",
    default_evaluator="flotorch/inference_model"  # Default LLM for metrics requiring evaluation
)

# Define a reference trajectory (optional)
reference_trajectory = {
    "input": "What is AWS Bedrock?",
    "expected_steps": [
        {
            "thought": "I need to search for information about AWS Bedrock",
            "tool_call": {
                "name": "search_tool",
                "arguments": {"query": "AWS Bedrock"}
            }
        },
        {
            "thought": "Now I can provide a comprehensive answer",
            "final_response": "AWS Bedrock is a fully managed service..."
        }
    ]
}

# Evaluate the trace

trace_ids = client.get_tracer_ids() 
for trace_id in trace_ids:
    evaluate_trajectory(
        trace_id=trace_id,
        reference=reference_trajectory  # Optional reference for comparison
        )

```

---

## 📊 Available Metrics

### LLM/RAG Metrics

| Metric | Engine | Description |
|--------|--------|-------------|
| `FAITHFULNESS` | DeepEval/Ragas | Measures if the answer is factually consistent with the context |
| `ANSWER_RELEVANCE` | DeepEval/Ragas | Evaluates how relevant the answer is to the question |
| `CONTEXT_RELEVANCY` | DeepEval | Assesses if the retrieved context is relevant to the question |
| `CONTEXT_PRECISION` | DeepEval/Ragas | Measures whether retrieved contexts are relevant |
| `CONTEXT_RECALL` | DeepEval | Measures the quality of retrival |
| `HALLUCINATION` | DeepEval | Detects if the model generates information not in the context |
| `ASPECT_CRITIC` | Ragas | Custom aspect-based evaluation (requires configuration) |
| `LATENCY` | Gateway | Measures total and average latency across LLM calls |
| `COST` | Gateway | Tracks total cost of LLM operations |
| `TOKEN_USAGE` | Gateway | Monitors total token consumption |

### Agent Metrics

- **TrajectoryEvalWithLLM**: LLM-based trajectory evaluation without reference
- **TrajectoryEvalWithLLMWithReference**: LLM-based trajectory evaluation with reference comparison
- **ToolCallAccuracy**: Measures correctness of tool invocations
- **AgentGoalAccuracy**: Validates if agent achieves specified goals
- **LatencyMetric**: Tracks agent response time and task completion speed
- **UsageMetric**: Monitors cost-efficiency and resource usage

---

## 🔧 Configuration

### Gateway Metrics (Latency, Cost, Token Usage)

Gateway metrics automatically track performance and usage statistics from your LLM calls. To enable these metrics, pass the response headers from FlotorchLLM as metadata:

```python
from flotorch.sdk.llm import FlotorchLLM
from flotorch_eval.llm_eval import LLMEvaluator, EvaluationItem

# Initialize FlotorchLLM
llm = FlotorchLLM(
    model_id="flotorch/gpt-4",
    api_key="your-api-key",
    base_url="flotorch-base-url"
)

# Make LLM call with return_headers=True to get metadata
response, headers = llm.invoke(
    messages=[{"role": "user", "content": "What is machine learning?"}],
    return_headers=True  # This returns headers containing latency, cost, and token info
)

# Create evaluation item with headers as metadata
eval_item = EvaluationItem(
    question="What is machine learning?",
    generated_answer=response.content,
    expected_answer="Machine learning is...",
    context=["Context documents..."],
    metadata=headers  # Pass headers directly as metadata
)

# Evaluate - Gateway metrics will be automatically computed
evaluator = LLMEvaluator(
    api_key="your-api-key",
    base_url="flotorch-base-url",
    inferencer_model="flotorch/gpt-4",
    embedding_model="flotorch/embedding-model"
)

results = evaluator.evaluate(data=[eval_item])

```
The results will include the gateway metrics total cost, average latency and total tokens

**Note:** Gateway metrics are computed automatically when metadata is present. No additional configuration is required.

### Metric Arguments

Customize metric thresholds and behavior:

```python
metric_args = {
    "faithfulness": {
        "threshold": 0.8,
        "truths_extraction_limit": 30
    },
    "answer_relevance": {
        "threshold": 0.7
    },
    "hallucination": {
        "threshold": 0.5
    },
    "context_precision": {
        "threshold": 0.7
    }
}
```

### Ragas Aspect Critic

Configure custom evaluation aspects:

```python
metric_args = {
    "aspect_critic": {
        "harmfulness": {
            "name": "harmfulness",
            "definition": "Does the response contain harmful content?"
        },
        "bias": {
            "name": "bias",
            "definition": "Does the response show bias or discrimination?"
        }
    }
}
```

---

## 📖 API Reference

### LLMEvaluator

```python
LLMEvaluator(
    api_key: str,
    base_url: str,
    inferencer_model: str,
    embedding_model: str,
    evaluation_engine: str = 'auto',
    metrics: Optional[List[MetricKey]] = None,
    metric_configs: Optional[Dict] = None
)
```

**Parameters:**
- `api_key` (str): API key for authentication
- `base_url` (str): Base URL for the Flotorch service
- `inferencer_model` (str): The LLM model to use for evaluation (e.g., "flotorch/gpt-4")
- `embedding_model` (str): The embedding model for metrics requiring embeddings
- `evaluation_engine` (str): Engine selection mode
  - `'auto'` (default): Automatically routes metrics with priority-based selection
  - `'ragas'`: Use only Ragas metrics
  - `'deepeval'`: Use only DeepEval metrics
- `metrics` (Optional[List[MetricKey]]): Default metrics to evaluate
  - If `None` with `auto`: Uses all available metrics from all engines
  - If `None` with specific engine: Uses all metrics from that engine
- `metric_configs` (Optional[Dict]): Configuration for metrics requiring additional parameters (e.g., AspectCritic)

**Methods:**
- `evaluate(data: List[EvaluationItem], metrics: Optional[List[MetricKey]] = None) -> Dict[str, Any]`
  - Evaluate the provided data using specified or default metrics
- `get_all_metrics() -> List[MetricKey]`
  - Returns all available metrics based on current engine configuration
- `set_metrics(metrics: List[MetricKey]) -> None`
  - Update the default metrics to use for evaluation

### EvaluationItem

```python
@dataclass
class EvaluationItem:
    question: str                    # The input question
    generated_answer: str            # Model's generated answer
    expected_answer: str             # Ground truth/expected answer
    context: List[str] = []          # Retrieved context documents
    metadata: Dict[str, Any] = {}    # Additional metadata
```

---

## 🤝 Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/flotorch/flotorch-eval.git
cd flotorch-eval

# Install in development mode
pip install -e ".[dev]"

# Run linters
pylint flotorch_eval/
black flotorch_eval/

```

---

## 📚 Documentation

Full documentation is available at [https://docs.flotorch.ai](https://docs.flotorch.ai)

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DeepEval**: Industry-standard evaluation metrics for LLMs
- **Ragas**: RAG-specific evaluation framework
- **OpenTelemetry**: Distributed tracing for agent evaluation
