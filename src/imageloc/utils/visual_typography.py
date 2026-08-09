"""Deterministic text transform detection for VisualStyle."""

from __future__ import annotations


def detect_text_transform(text: str) -> str:
    """Classify original text casing without guessing for non-letter strings."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "normal"
    if all(char.isupper() for char in letters):
        return "uppercase"
    if all(char.islower() for char in letters):
        return "lowercase"
    return "normal"
