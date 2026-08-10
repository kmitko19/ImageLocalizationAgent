"""Unit tests for deterministic render metadata finalization (R0.2A)."""

from __future__ import annotations

import pytest

from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import (
    BackgroundInfo,
    OCRData,
    RenderGeometry,
    RenderHints,
    ReviewInfo,
    TextBlock,
    VisualStyle,
)
from imageloc.utils.json_export import apply_block_review
from imageloc.utils.render_metadata import (
    build_background_info,
    build_render_geometry,
    finalize_render_hints,
    finalize_text_block_metadata,
    refresh_translation_dependent_render_hints,
)


def _sample_block(
    *,
    original_text: str = "SALE",
    translated_text: str | None = "РАСПРОДАЖА",
    rotation_deg: float = 0.0,
    background_color: str | None = "#F5E6D0",
    text_direction: str = "horizontal",
    estimated_lines: int = 1,
    target_line_count: int = 1,
) -> TextBlock:
    return TextBlock(
        id="block_001",
        original_text=original_text,
        translated_text=translated_text,
        source_language="EN",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=20, width=120, height=40),
        polygon=[(10, 20), (130, 20), (130, 60), (10, 60)],
        ocr=OCRData(raw_text=original_text, confidence=0.9),
        visual=VisualStyle(
            background_color=background_color,
            rotation_deg=rotation_deg,
        ),
        render_hints=RenderHints(
            available_width_px=120,
            available_height_px=40,
            estimated_lines=estimated_lines,
            target_line_count=target_line_count,
            text_direction=text_direction,
        ),
        review=ReviewInfo(),
    )


def test_finalize_render_hints_counts_sale_to_rasprodazha():
    hints = RenderHints(available_width_px=100, available_height_px=40)

    finalized = finalize_render_hints(
        hints,
        original_text="SALE",
        translated_text="РАСПРОДАЖА",
    )

    assert finalized.original_char_count == 4
    assert finalized.translated_char_count == len("РАСПРОДАЖА")
    assert finalized.length_ratio == pytest.approx(len("РАСПРОДАЖА") / 4)


def test_finalize_render_hints_empty_original_has_no_length_ratio():
    hints = RenderHints(available_width_px=100, available_height_px=40)

    finalized = finalize_render_hints(
        hints,
        original_text="",
        translated_text="Text",
    )

    assert finalized.original_char_count == 0
    assert finalized.translated_char_count == 4
    assert finalized.length_ratio is None


def test_finalize_render_hints_missing_translation_has_zero_translated_count():
    hints = RenderHints(available_width_px=100, available_height_px=40)

    finalized = finalize_render_hints(
        hints,
        original_text="SALE",
        translated_text=None,
    )

    assert finalized.translated_char_count == 0
    assert finalized.length_ratio == 0.0


def test_finalize_render_hints_sets_max_line_count():
    hints = RenderHints(
        available_width_px=100,
        available_height_px=80,
        estimated_lines=1,
        target_line_count=1,
    )

    finalized = finalize_render_hints(
        hints,
        original_text="DETAILS",
        translated_text="Details info with extra words",
        font_size_estimate=18,
    )

    assert finalized.max_line_count >= 2


def test_build_render_geometry_skips_defaults():
    visual = VisualStyle(rotation_deg=0.0)
    hints = RenderHints(available_width_px=100, available_height_px=40)

    assert build_render_geometry(visual, hints) is None


def test_build_render_geometry_copies_rotation_and_text_direction():
    visual = VisualStyle(rotation_deg=12.5)
    hints = RenderHints(
        available_width_px=100,
        available_height_px=40,
        text_direction="vertical",
    )

    geometry = build_render_geometry(visual, hints)

    assert geometry is not None
    assert geometry.baseline_angle_deg == pytest.approx(12.5)
    assert geometry.writing_direction == "vertical"
    assert geometry.perspective_detected is False


def test_build_background_info_uses_dominant_color_only():
    visual = VisualStyle(background_color="#AABBCC")

    background = build_background_info(visual)

    assert background is not None
    assert background.dominant_color == "#AABBCC"
    assert background.background_type == "unknown"
    assert background.classification_confidence == 0.0
    assert background.inpaint_required is False


def test_build_background_info_absent_without_background_color():
    assert build_background_info(VisualStyle()) is None


def test_finalize_text_block_metadata_populates_all_safe_fields():
    block = _sample_block(rotation_deg=3.0)

    finalized = finalize_text_block_metadata(block)

    assert finalized.render_hints.original_char_count == 4
    assert finalized.render_hints.translated_char_count == len("РАСПРОДАЖА")
    assert finalized.render_hints.max_line_count == 1
    assert finalized.render_geometry is not None
    assert finalized.render_geometry.baseline_angle_deg == pytest.approx(3.0)
    assert finalized.background is not None
    assert finalized.background.dominant_color == "#F5E6D0"
    assert finalized.background.background_type == "unknown"


def test_refresh_translation_dependent_render_hints_after_review_edit():
    block = finalize_text_block_metadata(_sample_block())
    edited = block.model_copy(update={"translated_text": "SALE EN"})

    refreshed = refresh_translation_dependent_render_hints(edited)

    assert refreshed.render_hints.original_char_count == 4
    assert refreshed.render_hints.translated_char_count == len("SALE EN")
    assert refreshed.render_hints.length_ratio == pytest.approx(len("SALE EN") / 4)


def test_apply_block_review_refreshes_translation_dependent_render_hints():
    from datetime import datetime, timezone

    from imageloc.models.result import PipelineResult, SourceImage, Stats

    block = finalize_text_block_metadata(_sample_block())
    result = PipelineResult(
        source_image_id="review-id",
        source_image=SourceImage(
            filename="sample.jpg",
            path="data/output/review-id/source.jpg",
            width=100,
            height=100,
            format="JPEG",
        ),
        target_language="EN",
        processed_at=datetime.now(timezone.utc),
        text_blocks=[block],
        stats=Stats(
            total_blocks=1,
            blocks_requiring_review=0,
            avg_confidence=0.9,
            ocr_blocks=1,
            vision_blocks=1,
        ),
    )

    updated = apply_block_review(
        result,
        "block_001",
        approved=True,
        user_action="edited",
        translated_text="UPDATED",
    )
    hints = updated.text_blocks[0].render_hints

    assert hints.translated_char_count == len("UPDATED")
    assert hints.length_ratio == pytest.approx(len("UPDATED") / 4)
    assert hints.original_char_count == 4
