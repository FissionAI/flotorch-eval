import json
from typing import Any, Dict, List
from flotorch_eval.agent_eval.core.schemas import Trajectory

def convert_trajectory_to_langchain_format(trajectory: Trajectory) -> List[Dict[str, Any]]:
    """
    Convert a trajectory into a LangChain-compatible format:
    List of {role, content, tool_calls}, where tool_calls follow LangChain structure.

    Args:
        trajectory: Agent trajectory object

    Returns:
        List of dictionaries representing LangChain-style messages
    """
    outputs = []
    for msg in trajectory.messages:
        output = {"role": msg.role, "content": msg.content}

        if msg.tool_calls:
            output["tool_calls"] = [
                {
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    }
                }
                for tool_call in msg.tool_calls
            ]

        outputs.append(output)

    return outputs
