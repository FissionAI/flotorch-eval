from opentel_utils import QueueSpanExporter
from flotorch_eval.agent_eval.core.converter import TraceConverter
import pandas as pd
from IPython.display import display
from flotorch_eval.agent_eval.metrics.base import BaseMetric
from flotorch_eval.agent_eval.core.evaluator import Evaluator
from typing import List

def get_all_spans(exporter: QueueSpanExporter)->list:
    """Extracts all spans from the queue exporter."""
    spans = []
    while not exporter.spans.empty():
        spans.append(exporter.spans.get())
    return spans

def create_trajectory(spans:list):
    """Converts a list of spans into a structured Trajectory object."""
    converter = TraceConverter()
    trajectory = converter.from_spans(spans)
    return trajectory

def display_evaluation_results(results):
    """
    Displays evaluation results in a clean, readable pandas DataFrame.
    
    Args:
        results: The results object from the evaluator.
    """
    if not results or not results.scores:
        print("No evaluation results were generated.")
        return
    
    # Convert the list of MetricResult objects into a list of dictionaries
    data = []
    for metric in results.scores:
        details = f"Error: {metric.details['error']}" if 'error' in metric.details else "Success"
        data.append({
            "Metric": metric.name,
            "Score": metric.score,
            "Details": details
        })
    df = pd.DataFrame(data)
    display(df)
    
def initialize_evaluator(metrics: List[BaseMetric]) -> Evaluator:
    """Initializes the Evaluator with a given list of metric objects."""
    return Evaluator(metrics=metrics)