"""
Utility functions for Ragas integration.
"""

from typing import Dict, List, Optional, Tuple

import ragas.messages as r

from flotorch_eval.agent_eval.core.schemas import Message, ToolCall, Trajectory

def convert_to_ragas_format(trajectory: Trajectory) -> List[r.Message]:
    """
    Convert a trajectory to Ragas message format.
    
    Args:
        trajectory: The trajectory to convert
        
    Returns:
        List of Ragas messages
    """
    ragas_messages = []
    for msg in trajectory.messages:
        if msg.role == "user":
            ragas_messages.append(r.HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    tool_calls.append(
                        r.ToolCall(
                            name=tc.name,
                            arguments=tc.arguments,
                            output=tc.output if tc.output else ""
                        )
                    )
                ragas_messages.append(
                    r.AssistantMessage(
                        content=msg.content,
                        tool_calls=tool_calls
                    )
                )
            else:
                ragas_messages.append(r.AssistantMessage(content=msg.content))
        elif msg.role == "tool":
            ragas_messages.append(r.ToolMessage(content=msg.content))
    return ragas_messages
