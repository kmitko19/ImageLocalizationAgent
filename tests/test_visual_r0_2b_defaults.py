"""R0.2B safe-default tests for deferred visual detections."""

from __future__ import annotations

from PIL import Image

from imageloc.fusion.ocr_vision_fusion import FusedBlock
from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import OCRData, ReviewInfo
from imageloc.utils.visual_analyzer import analyze_fused_block, analyze_visual_style


def _fused_block(text: str = "Sample") -> FusedBlock:
    return FusedBlock(
        id="block_001",
        text=text,
        source_language="EN",
        confidence=0.9,
        bbox=BoundingBox(x=20, y=20, width=100, height=40),
        polygon=[(20, 20), (120, 20), (120, 60), (20, 60)],
        ocr=OCRData(raw_text=text, confidence=0.9),
        vision=None,
        review=ReviewInfo(),
    )


def test_visual_style_keeps_deferred_fields_at_safe_defaults():
    image = Image.new("RGB", (200, 120), color=(240, 230, 220))
    bbox = BoundingBox(x=20, y=20, width=100, height=40)
    polygon = [(20, 20), (120, 20), (120, 60), (20, 60)]

    visual = analyze_visual_style(image, bbox, polygon, original_text="SALE")

    assert visual.text_transform == "uppercase"
    assert visual.text_opacity is None
    assert visual.font_family_candidate is None
    assert visual.font_family_confidence == 0.0
    assert visual.letter_spacing_px is None
    assert visual.letter_spacing_confidence == 0.0
    assert visual.line_height_ratio is None
    assert visual.line_height_confidence == 0.0
    assert visual.stroke_color is None
    assert visual.stroke_width_px == 0.0
    assert visual.stroke_confidence == 0.0
    assert visual.shadow_color is None
    assert visual.shadow_confidence == 0.0
    assert visual.fill_type == "solid"
    assert visual.gradient_start_color is None
    assert visual.gradient_confidence == 0.0


def test_analyze_fused_block_populates_background_and_text_transform():
    image = Image.new("RGB", (200, 120), color=(240, 230, 220))
    pixels = image.load()
    for row in range(20, 60):
        for col in range(20, 120):
            pixels[col, row] = (20, 10, 5)

    visual, hints, background = analyze_fused_block(image, _fused_block("SALE"))

    assert visual.text_transform == "uppercase"
    assert hints.available_width_px == 100
    assert background is not None
    assert background.background_type in {"solid", "unknown"}
    assert background.dominant_color is not None
