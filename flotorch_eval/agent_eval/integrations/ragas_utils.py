"""
Utility functions for Ragas integration.
"""

from typing import Dict, List, Optional, Tuple

import ragas.messages as r

from flotorch_eval.agent_eval.core.schemas import Message, ToolCall, Trajectory

def convert_trajectory_to_ragas_messages(trajectory: Trajectory) -> Tuple[List[r.Message], List[r.ToolCall]]:
    """
    Convert a trajectory to Ragas message format.
    
    Args:
        trajectory: The trajectory to convert
        
    Returns:
        List of Ragas messages
    """
    ragas_messages = []
    reference_tool_calls = []
    
    for msg in trajectory.messages:
        if msg.role == "user":
            ragas_messages.append(r.HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    ragas_tool_call = r.ToolCall(name=tc.name, args=tc.arguments, output=tc.output if tc.output else "")
                    tool_calls.append(ragas_tool_call)
                    reference_tool_calls.append(ragas_tool_call)
                ragas_messages.append(
                    r.AIMessage(
                        content=msg.content,
                        tool_calls=tool_calls
                    )
                )
            else:
                ragas_messages.append(r.AIMessage(content=msg.content))
        elif msg.role == "tool":
            ragas_messages.append(r.ToolMessage(content=msg.content))
    return ragas_messages, reference_tool_calls


# def convert_trajectory_to_ragas_messages(trajectory: Trajectory) -> List[r.Message]:
#     ragas_messages = []
#     for span in trajectory.spans:
#         if span.role == "user":
#             ragas_messages.append(r.HumanMessage(content=span.content))
#         elif span.role == "assistant":
#             ragas_messages.append(r.AIMessage(content=span.content))
#         elif span.role == "tool":
#             ragas_messages.append(r.ToolMessage(content=span.content))
#     return ragas_messages