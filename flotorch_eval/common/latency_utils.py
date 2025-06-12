from typing import List
from flotorch_eval.agent_eval.core.schemas import Trajectory
from flotorch_eval.agent_eval.core.schemas import LatencyBreakdownItem, LatencySummary

def extract_latency_from_trajectory(trajectory: Trajectory) -> LatencySummary:
    breakdown: List[LatencyBreakdownItem] = []
    total_latency = 0.0

    for span in trajectory.spans:
        start = span.start_time
        end = span.end_time

        if start is not None and end is not None:
            latency_ms = (end - start).total_seconds() * 1000
            latency_ms = round(latency_ms, 2)

            item = LatencyBreakdownItem(step_name=span.name, latency_ms=latency_ms)
            breakdown.append(item)
            total_latency += latency_ms

    average_latency = round(total_latency / len(breakdown), 2) if breakdown else 0.0

    return LatencySummary(
        total_latency_ms=round(total_latency, 2),
        average_step_latency_ms=average_latency,
        latency_breakdown=breakdown
    )
