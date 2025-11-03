from flotorch_eval.agent_eval.core.schemas import MetricResult, Trajectory
from flotorch_eval.common.latency_utils import extract_latency_from_trajectory
from flotorch_eval.agent_eval.metrics.base import LLMBaseEval

class LatencyMetric(LLMBaseEval):
    """
    Metric to compute latency per step and overall for a given trajectory.

    This metric extracts latency information from the provided trajectory and summarizes
    it in the MetricResult details.
    """

    @property
    def name(self) -> str:
        """
        Returns the name of the metric.
        """
        return "latency_summary"

    @property
    def run_async(self) -> bool:
        """Indicates that this metric should run asynchronously."""
        return False

    @property
    def needs_llm(self) -> bool:
        """
        Indicates whether this metric requires an LLM.
        """
        return False

    def evaluate(self, trajectory: Trajectory, metric_params) -> MetricResult:
        """
        Evaluate the latency for the given trajectory.

        Args:
            trajectory (Trajectory): The trajectory to evaluate.
            metric_params (dict): Additional parameters for the metric (not used).

        Returns:
            MetricResult: The result containing latency summary.
        """
        latency_summary = extract_latency_from_trajectory(trajectory)

        return MetricResult(
            name=self.name,
            score=0.0, 
            details=latency_summary.model_dump()
        )