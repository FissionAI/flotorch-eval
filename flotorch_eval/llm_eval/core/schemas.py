from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class EvaluationItem:
    """
    Represents one unit of evaluation — a question and its corresponding model output and metadata.
    """
    question: str
    generated_answer: str
    expected_answer: str
    context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)