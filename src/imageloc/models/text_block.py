"""Public TextBlock contract — the main unit of the pipeline's JSON output.

Field naming follows the plan's JSON contract (section 10 of the MVP plan)
exactly, so downstream stages (rendering, review UI) can rely on stable keys.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from imageloc.models.raw_blocks import BoundingBox, Point


class OCRData(BaseModel):
    """Verbatim EasyOCR result for this block, kept for audit."""

    provider: str = "easyocr"
    raw_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class VisionData(BaseModel):
    """Verbatim OpenAI Vision result for this block, kept for audit."""

    provider: str = "openai"
    corrected_text: str
    semantic_role: str = "other"
    language: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class VisualStyle(BaseModel):
    """Deterministic pixel measurements (OpenCV/Pillow) plus Vision's
    approximate typography reading.

    `estimated_text_height_px` is a geometric measurement from the OCR
    polygon and must never be confused with `font_size_estimate`, which is
    an approximation of the actual point/pixel font size.
    """

    text_color: str | None = None
    background_color: str | None = None
    estimated_text_height_px: int | None = None
    font_size_estimate: int | None = None
    font_size_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    font_category: str | None = None
    font_style: str | None = None
    font_weight: str | None = None
    alignment: str | None = None
    rotation_deg: float = 0.0


class RenderHints(BaseModel):
    """Data a future rendering stage needs to fit translated text into the
    original block's footprint, without re-running analysis."""

    available_width_px: int
    available_height_px: int
    estimated_lines: int = 1
    target_line_count: int = 1
    text_direction: str = "horizontal"
    preserve_layout: bool = True


class ReviewInfo(BaseModel):
    """Fusion's confidence assessment plus the user's manual decision.

    `user_confirmed` / `user_action` stay null until the user acts on a
    `requires_review` block in the GUI. Blocks that never required review are
    treated as auto-approved by the orchestrator when the JSON is written.
    """

    requires_review: bool = False
    reasons: list[str] = Field(default_factory=list)
    ocr_vision_similarity: float | None = None
    user_confirmed: bool | None = None
    user_action: str | None = None


class TextBlock(BaseModel):
    id: str
    original_text: str
    translated_text: str | None = None
    source_language: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    bbox: BoundingBox
    polygon: list[Point]

    ocr: OCRData
    vision: VisionData | None = None

    visual: VisualStyle
    render_hints: RenderHints
    review: ReviewInfo
