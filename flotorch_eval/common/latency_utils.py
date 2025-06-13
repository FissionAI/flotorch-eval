from typing import List, Dict
from collections import defaultdict
from flotorch_eval.agent_eval.core.schemas import Trajectory
from flotorch_eval.agent_eval.core.schemas import LatencyBreakdownItem, LatencySummary


def extract_latency_from_trajectory(trajectory: Trajectory) -> LatencySummary:
    id_to_item: Dict[str, LatencyBreakdownItem] = {}
    parent_to_children: Dict[str, List[LatencyBreakdownItem]] = defaultdict(list)
    root_items: List[LatencyBreakdownItem] = []

    for span in trajectory.spans:
        if span.start_time is None or span.end_time is None:
            continue

        latency_ms = round((span.end_time - span.start_time).total_seconds() * 1000, 2)
        item = LatencyBreakdownItem(
            step_name=span.name,
            latency_ms=latency_ms,
            children=[]
        )
        id_to_item[span.span_id] = item

    for span in trajectory.spans:
        item = id_to_item.get(span.span_id)
        if not item:
            continue

        parent_id = span.parent_span_id
        if parent_id and parent_id in id_to_item:
            parent_to_children[parent_id].append(item)
        else:
            root_items.append(item)

    for parent_id, children in parent_to_children.items():
        parent_item = id_to_item.get(parent_id)
        if parent_item:
            parent_item.children.extend(children)

    total_latency = round(sum(item.latency_ms for item in root_items), 2)
    average_latency = round(total_latency / len(root_items), 2) if root_items else 0.0

    return LatencySummary(
        total_latency_ms=total_latency,
        average_step_latency_ms=average_latency,
        latency_breakdown=root_items
    )
