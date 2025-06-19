from typing import List, Dict
from collections import defaultdict
from flotorch_eval.agent_eval.core.schemas import (
    TokenUsageRecord,
    TokenUsageSummary,
    TokenTotals,
    Trajectory,
)

def extract_token_usage_from_trajectory(trajectory: Trajectory) -> TokenUsageSummary:
    span_map: Dict[str, TokenUsageRecord] = {}
    child_span_ids = set()

    for span in trajectory.spans:
        attributes = span.attributes

        input_tokens = attributes.get("gen_ai.usage.input_tokens")
        output_tokens = attributes.get("gen_ai.usage.output_tokens")

        if input_tokens is None and output_tokens is None:
            input_tokens = attributes.get("gen_ai.usage.prompt_tokens")
            output_tokens = attributes.get("gen_ai.usage.completion_tokens")

        model = attributes.get("gen_ai.response.model") or attributes.get("gen_ai.request.model")

        if input_tokens is not None and output_tokens is not None and model:
            record = TokenUsageRecord(
                span_name=span.name,
                span_id=span.span_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
            span_map[span.span_id] = record

        if span.parent_id:
            child_span_ids.add(span.parent_id)

    deduplicated_records = [
        record for sid, record in span_map.items() if sid not in child_span_ids
    ]

    total_input = sum(r.input_tokens for r in deduplicated_records)
    total_output = sum(r.output_tokens for r in deduplicated_records)

    return TokenUsageSummary(
        token_usage=deduplicated_records,
        totals=TokenTotals(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output
        )
    )
