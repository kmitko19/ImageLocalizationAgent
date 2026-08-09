"""Unit tests for visual_analyzer — deterministic geometry and color sampling."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from imageloc.fusion.ocr_vision_fusion import FusedBlock
from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import OCRData, ReviewInfo
from imageloc.utils import visual_analyzer as va


def _make_text_on_background_image(
    *,
    bg=(240, 230, 220),
    text_color=(20, 10, 5),
    bbox=(10, 20, 100, 40),
) -> Image.Image:
    image = Image.new("RGB", (200, 120), color=bg)
    pixels = image.load()
    x, y, w, h = bbox
    for row in range(y, y + h):
        for col in range(x, x + w):
            pixels[col, row] = text_color
    return image


def test_estimated_text_height_px_uses_polygon_geometry():
    bbox = BoundingBox(x=10, y=20, width=100, height=40)
    polygon = [(10, 20), (110, 22), (108, 58), (12, 56)]

    height = va.estimated_text_height_px(polygon, bbox)

    assert height == 38


def test_rotation_from_polygon_returns_angle_in_degrees():
    polygon = [(0, 0), (10, 0), (10, 5), (0, 5)]

    assert va._rotation_from_polygon(polygon) == 0.0

    tilted = [(0, 0), (10, 10), (0, 20), (-10, 10)]
    assert va._rotation_from_polygon(tilted) == pytest.approx(45.0)


def test_sample_text_and_background_color_on_synthetic_image():
    image = _make_text_on_background_image()
    array = np.array(image)
    bbox = BoundingBox(x=10, y=20, width=100, height=40)

    text_color, background_color = va.sample_text_and_background_color(array, bbox)

    assert text_color.startswith("#")
    assert background_color.startswith("#")
    assert text_color != background_color


def test_estimate_font_size_lowers_confidence_for_decorative_font():
    regular_size, regular_conf = va.estimate_font_size(40, font_category="sans_serif")
    decorative_size, decorative_conf = va.estimate_font_size(
        40, font_category="decorative_script"
    )

    assert regular_size == 30
    assert decorative_size == 26
    assert decorative_conf < regular_conf


def test_analyze_visual_style_populates_visualstyle_fields():
    image = _make_text_on_background_image()
    bbox = BoundingBox(x=10, y=20, width=100, height=40)
    polygon = [(10, 20), (110, 20), (110, 60), (10, 60)]

    visual = va.analyze_visual_style(
        image,
        bbox,
        polygon,
        font_category="decorative_script",
        font_style="italic",
        font_weight="regular",
        alignment="center",
    )

    assert visual.estimated_text_height_px == 40
    assert visual.font_size_estimate == 26
    assert visual.font_size_confidence == pytest.approx(0.45)
    assert visual.font_category == "decorative_script"
    assert visual.alignment == "center"
    assert visual.text_color is not None
    assert visual.background_color is not None


def test_compute_render_hints_uses_bbox_and_visual_height():
    bbox = BoundingBox(x=0, y=0, width=200, height=80)
    visual = va.analyze_visual_style(
        _make_text_on_background_image(bbox=(0, 0, 200, 40)),
        bbox,
        [(0, 0), (200, 0), (200, 40), (0, 40)],
    )

    hints = va.compute_render_hints(bbox, "Two line\nsample text", visual)

    assert hints.available_width_px == 200
    assert hints.available_height_px == 80
    assert hints.estimated_lines == 2
    assert hints.preserve_layout is True


def test_analyze_fused_block_returns_visual_and_render_hints():
    fused = FusedBlock(
        id="block_001",
        text="Заголовок",
        source_language="RU",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=20, width=100, height=40),
        polygon=[(10, 20), (110, 20), (110, 60), (10, 60)],
        ocr=OCRData(raw_text="Заголовок", confidence=0.9),
        vision=None,
        review=ReviewInfo(),
    )
    image = _make_text_on_background_image()

    visual, hints = va.analyze_fused_block(
        image,
        fused,
        font_category="heading",
        alignment="left",
    )

    assert visual.estimated_text_height_px == 40
    assert hints.available_width_px == 100
    assert hints.available_height_px == 40


def test_analyze_visual_style_uses_mocked_color_sampler(monkeypatch):
    bbox = BoundingBox(x=0, y=0, width=50, height=20)
    polygon = [(0, 0), (50, 0), (50, 20), (0, 20)]
    image = Image.new("RGB", (100, 100), color="white")

    monkeypatch.setattr(
        va,
        "sample_text_and_background_color",
        lambda *_args, **_kwargs: ("#111111", "#EEEEEE"),
    )

    visual = va.analyze_visual_style(image, bbox, polygon)

    assert visual.text_color == "#111111"
    assert visual.background_color == "#EEEEEE"
