from flotorch_eval.agent_eval.core.schemas import TokenUsageSummary, CostSummary, CostRecord
from flotorch_eval.common.cost_compute_utils import calculate_model_inference_cost

async def calculate_cost_from_tokens(token_summary: TokenUsageSummary) -> CostSummary:
    """
    Calculate the total and average cost of LLM usage from a token usage summary.

    This function iterates over each token usage record, computes the cost for each
    using the model's pricing, and aggregates the results into a cost summary.

    Args:
        token_summary (TokenUsageSummary): Summary of token usage per span/model.

    Returns:
        CostSummary: An object containing the total cost, average cost per call,
                     and a breakdown of costs per span/model.
    """
    cost_breakdown = []
    total_cost = 0.0

    for record in token_summary.token_usage:
        cost = await calculate_model_inference_cost(
            record.input_tokens,
            record.output_tokens,
            record.model
        )

        cost_breakdown.append(CostRecord(
            span_id=record.span_id,
            model=record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cost=f"{cost:.6f}"
        ))
        total_cost += cost

    average_cost = total_cost / len(cost_breakdown) if cost_breakdown else 0.0

    return CostSummary(
        total_cost=f"{total_cost:.6f}",
        average_cost_per_call=f"{average_cost:.6f}",
        cost_breakdown=cost_breakdown
    )
    