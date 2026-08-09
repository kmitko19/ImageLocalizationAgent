"""Unit tests for OCRVisionFusion matching and review-flag rules, using
synthetic OCR/Vision blocks (no real image or API calls needed)."""

from imageloc.config import FusionSettings
from imageloc.fusion.ocr_vision_fusion import fuse
from imageloc.models.raw_blocks import BoundingBox, RawOCRBlock, RawVisionBlock
from imageloc.providers.openai_vision_provider import VISION_SCOPE_FALLBACK_CONFIDENCE

SETTINGS = FusionSettings(
    iou_threshold=0.25,
    text_similarity_threshold=0.55,
    text_mismatch_threshold=0.5,
    low_ocr_confidence_threshold=0.5,
    low_font_size_confidence_threshold=0.4,
)


def _bbox(x=0, y=0, w=100, h=40) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=w, height=h)


def _ocr(id_="block_001", text="Новая коллекцiя", confidence=0.72, bbox=None) -> RawOCRBlock:
    return RawOCRBlock(
        id=id_,
        raw_text=text,
        confidence=confidence,
        bbox=bbox or _bbox(),
        polygon=[(0, 0), (100, 0), (100, 40), (0, 40)],
    )


def test_match_by_explicit_ocr_block_id():
    ocr = [_ocr()]
    vision = [
        RawVisionBlock(
            ocr_block_id="block_001",
            text="Новая коллекцiя",
            corrected_text="Новая коллекция",
            semantic_role="heading",
            confidence=0.9,
        )
    ]

    result = fuse(ocr, vision, SETTINGS)

    assert len(result.fused_blocks) == 1
    block = result.fused_blocks[0]
    assert block.text == "Новая коллекция"
    assert block.vision is not None
    assert not result.unmatched_vision_blocks


def test_match_by_iou_and_text_when_no_block_id():
    ocr = [_ocr(bbox=_bbox(100, 100, 200, 50))]
    vision = [
        RawVisionBlock(
            text="Новая коллекцiя",
            corrected_text="Новая коллекция",
            bbox=_bbox(105, 102, 190, 48),
            confidence=0.9,
        )
    ]

    result = fuse(ocr, vision, SETTINGS)

    assert result.fused_blocks[0].vision is not None
    assert not result.unmatched_vision_blocks


def test_fuzzy_text_only_match_without_bbox():
    ocr = [_ocr(text="Hello World")]
    vision = [RawVisionBlock(text="Hello World", corrected_text="Hello World", confidence=0.9)]

    result = fuse(ocr, vision, SETTINGS)

    assert result.fused_blocks[0].vision is not None


def test_ocr_block_without_vision_match_is_flagged():
    ocr = [_ocr(confidence=0.8)]
    vision: list[RawVisionBlock] = []

    result = fuse(ocr, vision, SETTINGS)

    block = result.fused_blocks[0]
    assert block.vision is None
    assert block.review.requires_review is True
    assert "vision_no_match" in block.review.reasons
    assert block.text == "Новая коллекцiя"  # falls back to raw OCR text


def test_severe_text_mismatch_is_flagged_and_keeps_raw_ocr_text():
    ocr = [_ocr(text="Скидка 50%", confidence=0.9)]
    vision = [
        RawVisionBlock(
            ocr_block_id="block_001",
            text="Скидка 50%",
            corrected_text="Совершенно другой текст здесь",
            confidence=0.9,
        )
    ]

    result = fuse(ocr, vision, SETTINGS)

    block = result.fused_blocks[0]
    assert "text_mismatch" in block.review.reasons
    assert block.text == "Скидка 50%"  # does not trust the wildly different correction


def test_decorative_font_category_is_flagged():
    ocr = [_ocr(confidence=0.9)]
    vision = [
        RawVisionBlock(
            ocr_block_id="block_001",
            text="Новая коллекцiя",
            corrected_text="Новая коллекция",
            font_category="decorative_script",
            confidence=0.9,
        )
    ]

    result = fuse(ocr, vision, SETTINGS)

    assert "decorative_text" in result.fused_blocks[0].review.reasons


def test_low_ocr_confidence_is_flagged():
    ocr = [_ocr(confidence=0.3)]
    vision = [
        RawVisionBlock(
            ocr_block_id="block_001",
            text="Новая коллекцiя",
            corrected_text="Новая коллекция",
            confidence=0.9,
        )
    ]

    result = fuse(ocr, vision, SETTINGS)

    assert "low_ocr_confidence" in result.fused_blocks[0].review.reasons


def test_vision_block_without_ocr_match_becomes_unmatched():
    ocr = [_ocr()]
    vision = [
        RawVisionBlock(
            ocr_block_id="block_001",
            text="Новая коллекцiя",
            corrected_text="Новая коллекция",
            confidence=0.9,
        ),
        RawVisionBlock(
            text="Совсем отдельная подпись",
            corrected_text="Совсем отдельная подпись",
            confidence=0.8,
        ),
    ]

    result = fuse(ocr, vision, SETTINGS)

    assert len(result.fused_blocks) == 1
    assert len(result.unmatched_vision_blocks) == 1
    assert result.unmatched_vision_blocks[0].text == "Совсем отдельная подпись"


def test_confidence_is_min_of_ocr_and_vision_when_matched():
    ocr = [_ocr(confidence=0.6)]
    vision = [
        RawVisionBlock(
            ocr_block_id="block_001",
            text="Новая коллекцiя",
            corrected_text="Новая коллекция",
            confidence=0.95,
        )
    ]

    result = fuse(ocr, vision, SETTINGS)

    assert result.fused_blocks[0].confidence == 0.6


def test_confidence_penalized_when_no_vision_match():
    ocr = [_ocr(confidence=0.8)]

    result = fuse(ocr, [], SETTINGS)

    assert result.fused_blocks[0].confidence == 0.8 * 0.7


def test_fusion_flags_vision_scope_fallback_without_weakening_text_mismatch():
    ocr = [_ocr(id_="block_001", text="шербне", confidence=0.93)]
    vision = [
        RawVisionBlock(
            ocr_block_id="block_001",
            text="шербне",
            corrected_text="шербне",
            semantic_role="paragraph",
            language="ru",
            confidence=VISION_SCOPE_FALLBACK_CONFIDENCE,
        )
    ]

    result = fuse(ocr, vision, SETTINGS)
    review = result.fused_blocks[0].review

    assert "vision_scope_fallback" in review.reasons
    assert "text_mismatch" not in review.reasons
    assert result.fused_blocks[0].text == "шербне"
