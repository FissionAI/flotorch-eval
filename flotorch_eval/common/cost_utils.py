from typing import List
from flotorch_eval.agent_eval.core.schemas import TokenUsageSummary, CostSummary, CostRecord
from flotorch_eval.common.cost_compute_utils import calculate_bedrock_inference_cost

def calculate_cost_from_tokens(token_summary: TokenUsageSummary, aws_region: str) -> CostSummary:
    cost_breakdown = []
    total_cost = 0.0

    for record in token_summary.token_usage:
        cost = calculate_bedrock_inference_cost(
            record.input_tokens,
            record.output_tokens,
            record.model,
            aws_region
        )

        cost_breakdown.append(CostRecord(
            span_id=record.span_id,
            model=record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cost=round(cost, 6)
        ))
        total_cost += cost

    average_cost = total_cost / len(cost_breakdown) if cost_breakdown else 0.0

    return CostSummary(
        total_cost=round(total_cost, 6),
        average_cost_per_call=round(average_cost, 6),
        cost_breakdown=cost_breakdown
    )