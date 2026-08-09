"""Unit tests for conservative non-text OCR block detection."""

from __future__ import annotations

from imageloc.fusion.ocr_vision_fusion import FusedBlock
from imageloc.models.raw_blocks import BoundingBox, RawVisionBlock
from imageloc.models.text_block import OCRData, ReviewInfo, VisionData
from imageloc.utils.non_text_detector import (
    assess_obvious_non_text,
    is_significant_short_text,
    mark_obvious_non_text,
)


def _bbox(x=0, y=0, width=100, height=40) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=width, height=height)


def _fused(
    *,
    block_id: str = "block_001",
    text: str = "Sample",
    ocr_confidence: float = 0.9,
    bbox: BoundingBox | None = None,
    vision: VisionData | None = None,
    review: ReviewInfo | None = None,
) -> FusedBlock:
    bbox = bbox or _bbox()
    return FusedBlock(
        id=block_id,
        text=text,
        source_language="ru",
        confidence=ocr_confidence,
        bbox=bbox,
        polygon=[(bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), (bbox.x, bbox.y + bbox.height)],
        ocr=OCRData(raw_text=text, confidence=ocr_confidence),
        vision=vision,
        review=review or ReviewInfo(),
    )


def test_is_significant_short_text_keeps_prices_digits_and_words():
    assert is_significant_short_text("50%")
    assert is_significant_short_text("5")
    assert is_significant_short_text("за")
    assert is_significant_short_text("A")
    assert not is_significant_short_text("~")


def test_heart_like_glyph_is_obvious_non_text():
    fused = _fused(
        text="~",
        ocr_confidence=0.262,
        bbox=_bbox(x=135, y=549, width=132, height=160),
        vision=VisionData(
            corrected_text="~",
            semantic_role="other",
            language=None,
            confidence=0.99,
        ),
    )

    assessment = assess_obvious_non_text(fused)

    assert assessment.is_obvious_non_text is True
    assert "probable_non_text" not in assessment.signals
    assert "symbol_only_text" in assessment.signals
    assert "atypical_bbox" in assessment.signals


def test_short_russian_word_with_language_is_not_obvious_non_text():
    fused = _fused(
        text="за",
        ocr_confidence=0.384,
        bbox=_bbox(x=342, y=348, width=42, height=26),
        vision=VisionData(
            corrected_text="за",
            semantic_role="paragraph",
            language="ru",
            confidence=0.99,
        ),
    )

    assessment = assess_obvious_non_text(fused)

    assert assessment.is_obvious_non_text is False


def test_mark_obvious_non_text_adds_review_reason_and_skip_translation_flag():
    fused = _fused(
        text="~",
        ocr_confidence=0.262,
        bbox=_bbox(x=135, y=549, width=132, height=160),
        vision=VisionData(
            corrected_text="~",
            semantic_role="other",
            language=None,
            confidence=0.99,
        ),
    )

    updated, skip_translation = mark_obvious_non_text(fused)

    assert skip_translation is True
    assert updated.review.requires_review is True
    assert "probable_non_text" in updated.review.reasons


def test_borderline_other_role_without_shape_signals_is_not_obvious_non_text():
    fused = _fused(
        text="?",
        ocr_confidence=0.45,
        bbox=_bbox(width=18, height=22),
        vision=VisionData(
            corrected_text="?",
            semantic_role="other",
            language=None,
            confidence=0.8,
        ),
    )

    assessment = assess_obvious_non_text(fused)

    assert assessment.is_obvious_non_text is False
