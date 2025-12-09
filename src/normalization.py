import re
from typing import List, Callable, Dict


def to_lower(text: str) -> str:
    """Lowercase the text."""
    return text.lower()


def collapse_repetitions(text: str) -> str:
    """
    Collapse repeated characters:
    'soooo happppy!!!' -> 'soo happy!!'
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "lower": to_lower,
    "collapse": collapse_repetitions,
}


def apply_normalization_pipeline(text: str, steps: List[str]) -> str:
    """
    Apply a sequence of normalization steps, in order.
    steps is a list like ["lower", "collapse"].
    """
    out = text
    for s in steps:
        func = NORMALIZERS.get(s)
        if func is not None:
            out = func(out)
    return out