"""Tests for manual review workflow helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.result import PipelineResult, SourceImage, Stats
from imageloc.models.text_block import OCRData, RenderHints, ReviewInfo, TextBlock, VisionData, VisualStyle
from imageloc.utils.review_workflow import (
    approve_block,
    build_block_display,
    crop_block_image,
    edit_and_approve_block,
    next_pending_block_id,
    reject_block,
    review_progress,
    review_queue,
    save_reviewed_result,
    load_reviewed_result,
)


def _block(
    block_id: str,
    *,
    requires_review: bool = True,
    user_confirmed: bool | None = None,
    user_action: str | None = None,
    original_text: str = "Original",
    translated_text: str | None = "Translated",
    ocr_raw: str = "OCR raw",
    vision_corrected: str = "Vision corrected",
) -> TextBlock:
    return TextBlock(
        id=block_id,
        original_text=original_text,
        translated_text=translated_text,
        source_language="RU",
        confidence=0.8,
        bbox=BoundingBox(x=10, y=20, width=100, height=40),
        polygon=[(10, 20), (110, 20), (110, 60), (10, 60)],
        ocr=OCRData(raw_text=ocr_raw, confidence=0.7),
        vision=VisionData(
            corrected_text=vision_corrected,
            semantic_role="label",
            language="RU",
            confidence=0.9,
        ),
        visual=VisualStyle(estimated_text_height_px=40),
        render_hints=RenderHints(available_width_px=100, available_height_px=40),
        review=ReviewInfo(
            requires_review=requires_review,
            reasons=["text_mismatch"] if requires_review else [],
            user_confirmed=user_confirmed,
            user_action=user_action,
        ),
    )


def _sample_result(*blocks: TextBlock) -> PipelineResult:
    review_count = sum(1 for block in blocks if block.review.requires_review)
    return PipelineResult(
        source_image_id="review-test-id",
        source_image=SourceImage(
            filename="sample.png",
            path="data/output/review-test-id/source.png",
            width=200,
            height=120,
            format="PNG",
        ),
        source_language="RU",
        source_language_mode="auto",
        target_language="EN",
        processed_at=datetime.now(timezone.utc),
        text_blocks=list(blocks),
        stats=Stats(
            total_blocks=len(blocks),
            blocks_requiring_review=review_count,
            avg_confidence=0.8,
            ocr_blocks=len(blocks),
            vision_blocks=len(blocks),
        ),
    )


def test_review_queue_returns_only_pending_blocks():
    result = _sample_result(
        _block("block_001"),
        _block("block_002", user_confirmed=True, user_action="approved"),
        _block("block_003", requires_review=False),
    )

    pending = review_queue(result)

    assert [block.id for block in pending] == ["block_001"]


def test_review_progress_is_dynamic_not_from_stats_snapshot():
    result = _sample_result(
        _block("block_001"),
        _block("block_002"),
        _block("block_003", user_confirmed=True, user_action="approved"),
    )

    progress = review_progress(result)

    assert progress.total == 3
    assert progress.reviewed == 1
    assert progress.remaining == 2
    assert progress.is_complete is False
    assert progress.label == "Reviewed 1 / 3\nRemaining 2"


def test_review_progress_complete_label():
    result = _sample_result(
        _block("block_001", user_confirmed=True, user_action="approved"),
        _block("block_002", user_confirmed=False, user_action="rejected"),
    )

    progress = review_progress(result)

    assert progress.is_complete is True
    assert progress.label == "Review complete\nReviewed 2 / 2\nRemaining 0"


def test_next_pending_block_id_advances_queue():
    result = _sample_result(_block("block_001"), _block("block_002"))

    assert next_pending_block_id(result) == "block_001"
    assert next_pending_block_id(result, "block_001") == "block_002"
    assert next_pending_block_id(result, "block_002") is None


def test_approve_sets_user_action_approved():
    result = _sample_result(_block("block_001"))
    updated = approve_block(result, "block_001")
    review = updated.text_blocks[0].review

    assert review.user_confirmed is True
    assert review.user_action == "approved"


def test_edit_and_approve_sets_user_action_edited():
    result = _sample_result(_block("block_001"))
    updated = edit_and_approve_block(
        result,
        "block_001",
        original_text="Edited original",
        translated_text="Edited translation",
    )
    block = updated.text_blocks[0]

    assert block.review.user_confirmed is True
    assert block.review.user_action == "edited"
    assert block.original_text == "Edited original"
    assert block.translated_text == "Edited translation"
    assert block.ocr.raw_text == "OCR raw"
    assert block.vision.corrected_text == "Vision corrected"


def test_reject_sets_user_action_rejected():
    result = _sample_result(_block("block_001"))
    updated = reject_block(result, "block_001")
    review = updated.text_blocks[0].review

    assert review.user_confirmed is False
    assert review.user_action == "rejected"


def test_build_block_display_exposes_audit_fields():
    result = _sample_result(_block("block_001"))
    display = build_block_display(result.text_blocks[0])

    assert display.block_id == "block_001"
    assert display.ocr_raw_text == "OCR raw"
    assert display.vision_corrected_text == "Vision corrected"


def test_crop_block_image_respects_bbox(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 120), color="white").save(image_path)
    image = Image.open(image_path)
    bbox = BoundingBox(x=10, y=20, width=100, height=40)

    cropped = crop_block_image(image, bbox, padding=2)

    assert cropped.size == (104, 44)


def test_save_and_load_round_trip_after_review_actions(tmp_path):
    result = _sample_result(_block("block_001"), _block("block_002"), _block("block_003"))
    output_dir = tmp_path / "output"

    approved = approve_block(result, "block_001")
    edited = edit_and_approve_block(
        approved,
        "block_002",
        original_text="Edited",
        translated_text="Edited EN",
    )
    rejected = reject_block(edited, "block_003")

    saved_path = save_reviewed_result(rejected, output_dir)
    restored = load_reviewed_result(saved_path)

    by_id = {block.id: block for block in restored.text_blocks}
    assert by_id["block_001"].review.user_action == "approved"
    assert by_id["block_002"].review.user_action == "edited"
    assert by_id["block_003"].review.user_action == "rejected"
    assert restored.stats.blocks_requiring_review == 3
