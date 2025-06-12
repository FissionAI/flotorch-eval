from opentel_utils import QueueSpanExporter
from flotorch_eval.agent_eval.core.converter import TraceConverter
import pandas as pd
from IPython.display import display
from flotorch_eval.agent_eval.metrics.base import BaseMetric
from flotorch_eval.agent_eval.core.evaluator import Evaluator
from flotorch_eval.agent_eval.metrics.base import MetricResult
from typing import List
import textwrap

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

def display_evaluation_results(results: MetricResult):
    """
    Displays evaluation results generically as a clean text table.
    This version does NOT require the jinja2 library.

    Args:
        results: An object containing a list of MetricResult objects.
    """
    if not results or not getattr(results, 'scores', None):
        print("No evaluation results were generated.")
        return

    data = []
    for metric in results.scores:
        details_dict = metric.details
        display_parts = []

        if 'error' in details_dict:
            error_message = f"Error: {details_dict['error']}"
            display_parts.append(textwrap.fill(error_message, width=80))
            del details_dict['error']

        elif 'comment' in details_dict:
            comment = details_dict['comment']
            display_parts.append(textwrap.fill(comment, width=80))
            del details_dict['comment']


        other_details = []
        if details_dict:
            for key, value in details_dict.items():
                if key == 'errors' and isinstance(value, list) and value:
                    error_list = [f"  - Tool '{e.get('tool', '??')}': {e.get('error', 'No details')}" for e in value]
                    other_details.append("Detailed Errors:\n" + "\n".join(error_list))
                else:
                    other_details.append(f"- {key}: {value}")
            
        score_display = f"{metric.score:.2f}" if isinstance(metric.score, (int, float)) else "N/A"

        data.append({
            "Metric": metric.name,
            "Score": score_display,
            "Details": "".join(display_parts)
        })

    df = pd.DataFrame(data)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    
    display(df)
    
def initialize_evaluator(metrics: List[BaseMetric]) -> Evaluator:
    """Initializes the Evaluator with a given list of metric objects."""
    return Evaluator(metrics=metrics)