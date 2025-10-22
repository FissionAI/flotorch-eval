import ast
from flotorch_eval.agent_eval.core.schemas import (
    TokenUsageRecord,
    TokenUsageSummary,
    TokenTotals,
    Trajectory,
)


def extract_token_usage_from_trajectory(trajectory: Trajectory) -> TokenUsageSummary:
    """
    Extract token usage from a trajectory.

    This function iterates over each span in the trajectory, extracts the input and output tokens,
    and returns a summary of the token usage.

    Args:
        trajectory (Trajectory): The trajectory to extract token usage from.

    Returns:
        TokenUsageSummary: A summary of the token usage.
    """
    records = []
    total_input = 0
    total_output = 0

    for span in trajectory.spans:
        attributes = span.attributes

        input_tokens = attributes.get("gen_ai.usage.input_tokens")
        output_tokens = attributes.get("gen_ai.usage.output_tokens")

        # Get the model from response
        model = attributes.get("gen_ai.response.model")
        if not model:
            model = attributes.get("gen_ai.request.model")

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
