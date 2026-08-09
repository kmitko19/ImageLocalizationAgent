"""Conservative detection of obvious non-text OCR regions (icons, glyphs, noise).

Uses a combination of signals — never a single criterion alone. Short real
labels, digits, prices and single letters must not be auto-excluded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from imageloc.fusion.ocr_vision_fusion import FusedBlock
from imageloc.models.raw_blocks import BoundingBox, RawVisionBlock
from imageloc.models.text_block import ReviewInfo

SIGNIFICANT_TEXT_ROLES = frozenset({"paragraph", "heading", "label", "caption", "title", "button"})

# Typical single-character text bbox is well below this (heart icon ~ 21k px²/char).
_AREA_PER_CHAR_SINGLE = 4000
_AREA_PER_CHAR_PAIR = 8000
_MIN_OBVIOUS_SIGNALS = 4


@dataclass(frozen=True)
class NonTextAssessment:
    is_obvious_non_text: bool
    signals: tuple[str, ...]


def is_significant_short_text(text: str) -> bool:
    """Return True when short text may still be meaningful for localization."""
    stripped = text.strip()
    if not stripped:
        return False
    if any(char.isdigit() for char in stripped):
        return True
    if re.search(r"[$€£¥₽%]", stripped):
        return True
    if re.match(r"^[\w\d]+$", stripped, flags=re.UNICODE) and any(char.isalnum() for char in stripped):
        return True
    return False


def is_symbol_or_decorative_glyph_only(text: str) -> bool:
    """True for punctuation-only glyphs such as '~' that are unlikely to be words."""
    stripped = text.strip()
    if not stripped:
        return True
    if is_significant_short_text(stripped):
        return False
    return not any(char.isalnum() for char in stripped)


def bbox_area_per_character(bbox: BoundingBox, text: str) -> float:
    characters = max(1, len(text.strip()))
    return (bbox.width * bbox.height) / characters


def is_bbox_atypical_for_text(bbox: BoundingBox, text: str) -> bool:
    """Large footprint relative to character count suggests icon/shape, not text."""
    stripped = text.strip()
    if not stripped:
        return False

    area_per_char = bbox_area_per_character(bbox, stripped)
    length = len(stripped)
    if length == 1:
        return area_per_char >= _AREA_PER_CHAR_SINGLE
    if length == 2:
        return area_per_char >= _AREA_PER_CHAR_PAIR
    return length <= 3 and area_per_char >= _AREA_PER_CHAR_PAIR * 2


def _vision_suggests_non_text(vision: RawVisionBlock | None) -> bool:
    if vision is None:
        return False
    return vision.semantic_role == "other" and vision.language is None


def _fused_vision_suggests_non_text(fused: FusedBlock) -> bool:
    if fused.vision is None:
        return False
    return fused.vision.semantic_role == "other" and fused.vision.language is None


def assess_obvious_non_text(
    fused: FusedBlock,
    vision_raw: RawVisionBlock | None = None,
) -> NonTextAssessment:
    """Score a fused block; obvious non-text requires multiple combined signals."""
    text = fused.text.strip()
    ocr_confidence = fused.ocr.confidence
    role = (
        vision_raw.semantic_role
        if vision_raw is not None
        else fused.vision.semantic_role
        if fused.vision is not None
        else "other"
    )
    language = (
        vision_raw.language
        if vision_raw is not None
        else fused.vision.language
        if fused.vision is not None
        else None
    )

    if is_significant_short_text(text):
        return NonTextAssessment(False, ())
    if role in SIGNIFICANT_TEXT_ROLES and language is not None:
        return NonTextAssessment(False, ())
    if role in SIGNIFICANT_TEXT_ROLES and ocr_confidence >= 0.5:
        return NonTextAssessment(False, ())

    signals: list[str] = []
    if role == "other":
        signals.append("semantic_role_other")
    if language is None:
        signals.append("language_null")
    if ocr_confidence < 0.5:
        signals.append("low_ocr_confidence")
    if is_symbol_or_decorative_glyph_only(text):
        signals.append("symbol_only_text")
    if is_bbox_atypical_for_text(fused.bbox, text):
        signals.append("atypical_bbox")
    if _vision_suggests_non_text(vision_raw) or (
        vision_raw is None and _fused_vision_suggests_non_text(fused)
    ):
        signals.append("vision_non_text")

    has_shape_signal = "atypical_bbox" in signals or (
        "symbol_only_text" in signals and ocr_confidence < 0.3
    )
    is_obvious = has_shape_signal and len(signals) >= _MIN_OBVIOUS_SIGNALS
    return NonTextAssessment(is_obvious, tuple(signals))


def mark_obvious_non_text(
    fused: FusedBlock,
    vision_raw: RawVisionBlock | None = None,
) -> tuple[FusedBlock, bool]:
    """Update review flags and return whether translation should be skipped."""
    assessment = assess_obvious_non_text(fused, vision_raw)
    if not assessment.is_obvious_non_text:
        return fused, False

    reasons = list(fused.review.reasons)
    if "probable_non_text" not in reasons:
        reasons.append("probable_non_text")

    updated_review = fused.review.model_copy(
        update={
            "requires_review": True,
            "reasons": reasons,
        }
    )
    return replace(fused, review=updated_review), True
