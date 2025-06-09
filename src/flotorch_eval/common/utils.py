"""
Common utilities for all evaluation types.
"""

from typing import Any, Dict, List, Optional, Union

def convert_attributes(attributes: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool, List[str]]]:
    """
    Convert attributes to a standardized format.
    
    Args:
        attributes: Dictionary of attributes to convert
        
    Returns:
        Dictionary with standardized attribute types
    """
    result = {}
    for key, value in attributes.items():
        if isinstance(value, (str, int, float, bool)) or (
            isinstance(value, list) and all(isinstance(x, str) for x in value)
        ):
            result[key] = value
        else:
            try:
                if isinstance(value, (dict, list)):
                    result[key] = str(value)
                else:
                    result[key] = str(value)
            except (TypeError, ValueError):
                result[key] = str(value)
    return result 