"""Unit tests for R0.1 renderer-oriented model extensions."""

from __future__ import annotations

import json

from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.result import CONTRACT_VERSION, PipelineResult
from imageloc.models.text_block import (
    BackgroundInfo,
    OCRData,
    RenderGeometry,
    RenderHints,
    ReviewInfo,
    TextBlock,
    VisualStyle,
)


LEGACY_TEXT_BLOCK_JSON = """
{
  "id": "block_001",
  "original_text": "Беречь",
  "translated_text": "Save",
  "source_language": "ru",
  "confidence": 0.74,
  "bbox": {"x": 344, "y": 50, "width": 74, "height": 30},
  "polygon": [[344, 50], [418, 50], [418, 80], [344, 80]],
  "ocr": {"provider": "easyocr", "raw_text": "Биречь", "confidence": 0.74},
  "vision": {
    "provider": "openai",
    "corrected_text": "Беречь",
    "semantic_role": "other",
    "language": "ru",
    "confidence": 0.95
  },
  "visual": {
    "text_color": "#AFA79A",
    "background_color": "#101112",
    "estimated_text_height_px": 30,
    "font_size_estimate": 22,
    "font_size_confidence": 0.7,
    "rotation_deg": 0.0
  },
  "render_hints": {
    "available_width_px": 74,
    "available_height_px": 30,
    "estimated_lines": 1,
    "target_line_count": 1,
    "text_direction": "horizontal",
    "preserve_layout": true
  },
  "review": {
    "requires_review": false,
    "reasons": [],
    "ocr_vision_similarity": 0.83,
    "user_confirmed": true,
    "user_action": "approved"
  }
}
"""


def test_legacy_text_block_json_loads_without_new_fields():
    block = TextBlock.model_validate_json(LEGACY_TEXT_BLOCK_JSON)

    assert block.id == "block_001"
    assert block.render_geometry is None
    assert block.background is None
    assert block.visual.font_family_candidate is None
    assert block.visual.font_family_confidence == 0.0
    assert block.visual.fill_type == "solid"
    assert block.render_hints.font_scale_min == 0.65
    assert block.render_hints.overflow_strategy == "fit"


def test_visual_style_renderer_defaults():
    style = VisualStyle()

    assert style.font_family_candidate is None
    assert style.font_family_confidence == 0.0
    assert style.letter_spacing_px is None
    assert style.letter_spacing_confidence == 0.0
    assert style.line_height_ratio is None
    assert style.line_height_confidence == 0.0
    assert style.vertical_alignment is None
    assert style.text_opacity is None
    assert style.text_transform is None
    assert style.stroke_color is None
    assert style.stroke_width_px == 0.0
    assert style.stroke_confidence == 0.0
    assert style.shadow_color is None
    assert style.shadow_opacity is None
    assert style.shadow_offset_x_px is None
    assert style.shadow_offset_y_px is None
    assert style.shadow_blur_px is None
    assert style.shadow_confidence == 0.0
    assert style.fill_type == "solid"
    assert style.gradient_start_color is None
    assert style.gradient_end_color is None
    assert style.gradient_angle_deg is None
    assert style.gradient_confidence == 0.0


def test_render_hints_renderer_defaults():
    hints = RenderHints(available_width_px=100, available_height_px=40)

    assert hints.original_char_count is None
    assert hints.translated_char_count is None
    assert hints.length_ratio is None
    assert hints.font_scale_min == 0.65
    assert hints.font_scale_max == 1.10
    assert hints.letter_spacing_scale_min == 0.85
    assert hints.letter_spacing_scale_max == 1.15
    assert hints.line_height_scale_min == 0.85
    assert hints.line_height_scale_max == 1.15
    assert hints.max_line_count is None
    assert hints.bbox_expansion_allowed is False
    assert hints.bbox_expansion_px == 0
    assert hints.allow_font_substitution is True
    assert hints.allow_line_reflow is True
    assert hints.overflow_strategy == "fit"


def test_render_geometry_and_background_defaults():
    geometry = RenderGeometry()
    background = BackgroundInfo()

    assert geometry.baseline_angle_deg is None
    assert geometry.perspective_quad is None
    assert geometry.mask_bbox is None
    assert geometry.mask_path is None
    assert geometry.mask_confidence == 0.0
    assert geometry.writing_direction == "horizontal"
    assert geometry.perspective_detected is False

    assert background.background_type == "unknown"
    assert background.dominant_color is None
    assert background.secondary_color is None
    assert background.complexity_score == 0.0
    assert background.classification_confidence == 0.0
    assert background.inpaint_required is False
    assert background.preferred_restore_method == "auto"


def test_text_block_with_renderer_fields_round_trips():
    block = TextBlock(
        id="block_renderer",
        original_text="Sample",
        translated_text="Пример",
        source_language="en",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=20, width=120, height=40),
        polygon=[(10, 20), (130, 20), (130, 60), (10, 60)],
        ocr=OCRData(raw_text="Sample", confidence=0.9),
        visual=VisualStyle(
            text_color="#FFFFFF",
            font_family_candidate="Arial",
            font_family_confidence=0.8,
            fill_type="gradient",
            gradient_start_color="#111111",
            gradient_end_color="#222222",
            gradient_angle_deg=45.0,
        ),
        render_hints=RenderHints(
            available_width_px=120,
            available_height_px=40,
            original_char_count=6,
            translated_char_count=6,
            length_ratio=1.0,
            max_line_count=1,
        ),
        review=ReviewInfo(),
        render_geometry=RenderGeometry(
            baseline_angle_deg=1.5,
            perspective_quad=[(10, 20), (130, 22), (128, 60), (12, 58)],
            mask_bbox=BoundingBox(x=8, y=18, width=124, height=44),
            mask_path="masks/block_renderer.png",
            mask_confidence=0.91,
            writing_direction="horizontal",
            perspective_detected=True,
        ),
        background=BackgroundInfo(
            background_type="photo",
            dominant_color="#333333",
            secondary_color="#555555",
            complexity_score=0.72,
            classification_confidence=0.88,
            inpaint_required=True,
            preferred_restore_method="patchmatch",
        ),
    )

    restored = TextBlock.model_validate_json(block.model_dump_json())

    assert restored == block
    assert restored.render_geometry is not None
    assert restored.render_geometry.perspective_detected is True
    assert restored.background is not None
    assert restored.background.inpaint_required is True


def test_legacy_pipeline_result_version_1_1_still_loads():
    payload = {
        "version": "1.1",
        "source_image_id": "legacy-id",
        "source_image": {
            "filename": "legacy.jpg",
            "path": "data/output/legacy-id/source.jpg",
            "width": 100,
            "height": 100,
            "format": "JPEG",
        },
        "target_language": "EN",
        "processed_at": "2026-08-09T10:00:00Z",
        "text_blocks": [json.loads(LEGACY_TEXT_BLOCK_JSON)],
        "unmatched_vision_blocks": [],
        "stats": {
            "total_blocks": 1,
            "blocks_requiring_review": 0,
            "avg_confidence": 0.74,
            "ocr_blocks": 1,
            "vision_blocks": 1,
        },
    }

    result = PipelineResult.model_validate(payload)

    assert result.version == "1.1"
    assert result.text_blocks[0].render_geometry is None
    assert result.text_blocks[0].background is None


def test_new_pipeline_result_default_contract_version():
    assert CONTRACT_VERSION == "1.2"

    block = TextBlock.model_validate_json(LEGACY_TEXT_BLOCK_JSON)
    result = PipelineResult(
        source_image_id="new-id",
        source_image={
            "filename": "new.jpg",
            "path": "data/output/new-id/source.jpg",
            "width": 100,
            "height": 100,
            "format": "JPEG",
        },
        target_language="EN",
        processed_at="2026-08-09T10:00:00Z",
        text_blocks=[block],
        stats={
            "total_blocks": 1,
            "blocks_requiring_review": 0,
            "avg_confidence": 0.74,
            "ocr_blocks": 1,
            "vision_blocks": 1,
        },
    )

    assert result.version == "1.2"
