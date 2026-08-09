"""OCRVisionFusion — deterministic matching and merging of EasyOCR and
OpenAI Vision results into a single, auditable text block.

Design rules (see the MVP plan, section 4):

- Geometry always comes from EasyOCR. Vision's optional `bbox` is only used
  to help matching; it never overrides OCR coordinates.
- Every discrepancy is recorded, never silently resolved. A block is
  flagged `requires_review` rather than guessed away.
- No ML/embeddings here — matching is IoU + `difflib` text similarity,
  entirely deterministic and easy to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from imageloc.config import FusionSettings
from imageloc.models.raw_blocks import BoundingBox, Point, RawOCRBlock, RawVisionBlock
from imageloc.models.text_block import OCRData, ReviewInfo, VisionData
from imageloc.providers.openai_vision_provider import VISION_SCOPE_FALLBACK_CONFIDENCE

DECORATIVE_FONT_KEYWORDS = ("script", "decorative", "handwritten", "calligraphy")


@dataclass
class FusedBlock:
    """OCR + Vision merged for one block, before visual analysis and
    translation add the remaining TextBlock fields."""

    id: str
    text: str
    source_language: str | None
    confidence: float
    bbox: BoundingBox
    polygon: list[Point]
    ocr: OCRData
    vision: VisionData | None
    review: ReviewInfo


@dataclass
class FusionResult:
    fused_blocks: list[FusedBlock] = field(default_factory=list)
    unmatched_vision_blocks: list[RawVisionBlock] = field(default_factory=list)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height

    inter_w = max(0, min(ax2, bx2) - max(a.x, b.x))
    inter_h = max(0, min(ay2, by2) - max(a.y, b.y))
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = a.width * a.height + b.width * b.height - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _is_decorative(font_category: str | None) -> bool:
    if not font_category:
        return False
    lowered = font_category.lower()
    return any(keyword in lowered for keyword in DECORATIVE_FONT_KEYWORDS)


def _find_match(
    ocr_block: RawOCRBlock,
    remaining_vision: list[RawVisionBlock],
    settings: FusionSettings,
) -> RawVisionBlock | None:
    """Priority: explicit ocr_block_id -> IoU + text -> fuzzy text only."""

    for vision_block in remaining_vision:
        if vision_block.ocr_block_id == ocr_block.id:
            return vision_block

    geometric_candidates = [v for v in remaining_vision if v.bbox is not None]
    best_geometric: tuple[float, RawVisionBlock] | None = None
    for vision_block in geometric_candidates:
        iou = _iou(ocr_block.bbox, vision_block.bbox)
        if iou < settings.iou_threshold:
            continue
        similarity = _similarity(ocr_block.raw_text, vision_block.text)
        if best_geometric is None or similarity > best_geometric[0]:
            best_geometric = (similarity, vision_block)
    if best_geometric is not None:
        return best_geometric[1]

    best_text: tuple[float, RawVisionBlock] | None = None
    for vision_block in remaining_vision:
        similarity = _similarity(ocr_block.raw_text, vision_block.text)
        if similarity < settings.text_similarity_threshold:
            continue
        if best_text is None or similarity > best_text[0]:
            best_text = (similarity, vision_block)
    return best_text[1] if best_text else None


def fuse(
    ocr_blocks: list[RawOCRBlock],
    vision_blocks: list[RawVisionBlock],
    settings: FusionSettings,
) -> FusionResult:
    remaining_vision = list(vision_blocks)
    fused_blocks: list[FusedBlock] = []

    for ocr_block in ocr_blocks:
        matched = _find_match(ocr_block, remaining_vision, settings)
        reasons: list[str] = []
        similarity: float | None = None

        if matched is not None:
            remaining_vision.remove(matched)
            similarity = _similarity(ocr_block.raw_text, matched.corrected_text)

            text = matched.corrected_text if similarity >= settings.text_similarity_threshold else ocr_block.raw_text
            if similarity < settings.text_mismatch_threshold:
                reasons.append("text_mismatch")

            if (
                matched.ocr_block_id == ocr_block.id
                and matched.confidence <= VISION_SCOPE_FALLBACK_CONFIDENCE
                and matched.corrected_text == ocr_block.raw_text
            ):
                reasons.append("vision_scope_fallback")

            confidence = min(ocr_block.confidence, matched.confidence)
            vision_data = VisionData(
                corrected_text=matched.corrected_text,
                semantic_role=matched.semantic_role,
                language=matched.language,
                confidence=matched.confidence,
            )
            if _is_decorative(matched.font_category):
                reasons.append("decorative_text")
        else:
            text = ocr_block.raw_text
            confidence = ocr_block.confidence * 0.7
            vision_data = None
            reasons.append("vision_no_match")

        if ocr_block.confidence < settings.low_ocr_confidence_threshold:
            reasons.append("low_ocr_confidence")

        fused_blocks.append(
            FusedBlock(
                id=ocr_block.id,
                text=text,
                source_language=matched.language if matched else None,
                confidence=confidence,
                bbox=ocr_block.bbox,
                polygon=ocr_block.polygon,
                ocr=OCRData(raw_text=ocr_block.raw_text, confidence=ocr_block.confidence),
                vision=vision_data,
                review=ReviewInfo(
                    requires_review=bool(reasons),
                    reasons=reasons,
                    ocr_vision_similarity=similarity,
                ),
            )
        )

    return FusionResult(fused_blocks=fused_blocks, unmatched_vision_blocks=remaining_vision)
