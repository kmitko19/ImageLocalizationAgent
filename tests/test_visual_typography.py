"""Unit tests for deterministic text transform detection."""

from __future__ import annotations

import pytest

from imageloc.utils.visual_typography import detect_text_transform


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SALE", "uppercase"),
        ("привет", "lowercase"),
        ("Привет", "normal"),
        ("Hello WORLD", "normal"),
        ("12345", "normal"),
        ("", "normal"),
    ],
)
def test_detect_text_transform(text: str, expected: str):
    assert detect_text_transform(text) == expected
