from typing import List
from flotorch_eval.agent_eval.core.schemas import (
    TokenUsageRecord,
    TokenUsageSummary,
    TokenTotals,
    Trajectory,
)
import ast


def extract_token_usage_from_trajectory(trajectory: Trajectory) -> TokenUsageSummary:
    records = []
    total_input = 0
    total_output = 0

    for span in trajectory.spans:
        attributes = span.attributes

        input_tokens = attributes.get("gen_ai.request.token_count")
        output_tokens = attributes.get("gen_ai.response.token_count")

        # Get the model from response
        model = None
        model_response = attributes.get("gen_ai.response.full")
        if model_response is not None:
            if model_response.startswith("metadata="):
                model_response = model_response[len("metadata="):]

            # Sometimes there's a trailing ` content=...`, remove it
            if " content=" in model_response:
                model_response = model_response.split(" content=")[0].strip()

            # 2. Parse into Python dict safely
            parsed = ast.literal_eval(model_response)

            # 3. Extract model
            model = parsed["raw_response"]["model"]

        if input_tokens is not None and output_tokens is not None and model:
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
            record = TokenUsageRecord(
                span_name=span.name,
                span_id=span.span_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
            records.append(record)
            total_input += input_tokens
            total_output += output_tokens

    return TokenUsageSummary(
        token_usage=records,
        totals=TokenTotals(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output
        )
    )
