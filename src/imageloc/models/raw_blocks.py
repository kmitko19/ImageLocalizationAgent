"""Internal raw outputs from OCR and Vision providers, before fusion.

These are not part of the final JSON contract — OCRVisionFusion consumes
them and produces the public TextBlock model (see text_block.py).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


Point = tuple[int, int]


class RawOCRBlock(BaseModel):
    """One text region detected by EasyOCR."""

    id: str
    raw_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    polygon: list[Point]


class RawVisionBlock(BaseModel):
    """One text region reported by OpenAI Vision.

    `ocr_block_id` is populated when Vision explicitly references one of the
    OCR anchors it was given in the prompt. `bbox` is optional and, even when
    present, is never treated as authoritative geometry (EasyOCR owns
    coordinates) — see OCRVisionFusion.
    """

    ocr_block_id: str | None = None
    text: str
    corrected_text: str
    semantic_role: str = "other"
    language: str | None = None
    font_category: str | None = None
    font_style: str | None = None
    font_weight: str | None = None
    alignment: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    bbox: BoundingBox | None = None
