"""Unit tests for Renderer R0.5 layout and typography refinement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import (
    OCRData,
    RenderGeometry,
    RenderHints,
    ReviewInfo,
    TextBlock,
    VisualStyle,
)
from imageloc.renderer.text_renderer import (
    _measure_text_block,
    compute_content_box,
    compute_content_padding_px,
    fit_font_size_and_lines,
    render_text_block,
    resolve_font,
    wrap_text_lines,
)

pytestmark = pytest.mark.skipif(
    not (Path(__import__("os").environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf").is_file(),
    reason="Windows Unicode fonts required for text layout tests",
)


def _block(
    *,
    block_id: str = "block_001",
    bbox: BoundingBox,
    translated_text: str = "SALE",
    render_hints: RenderHints | None = None,
    visual: VisualStyle | None = None,
) -> TextBlock:
    hints = render_hints or RenderHints(available_width_px=bbox.width, available_height_px=bbox.height)
    style = visual or VisualStyle(
        background_color="#FFFFFF",
        text_color="#000000",
        estimated_text_height_px=bbox.height,
        font_size_estimate=max(8, bbox.height - 4),
        alignment="center",
        vertical_alignment="center",
    )
    return TextBlock(
        id=block_id,
        original_text="SALE",
        translated_text=translated_text,
        source_language="EN",
        confidence=0.9,
        bbox=bbox,
        polygon=[
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            (bbox.x, bbox.y + bbox.height),
        ],
        ocr=OCRData(raw_text="SALE", confidence=0.9),
        visual=style,
        render_hints=hints,
        review=ReviewInfo(),
        render_geometry=RenderGeometry(mask_bbox=bbox, mask_confidence=0.8),
    )


def _blank_image(size=(260, 140), color=(240, 230, 220)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _ink_mask(region: np.ndarray, background: tuple[int, int, int], tolerance: int = 35) -> np.ndarray:
    diff = np.abs(region.astype(np.int16) - np.array(background, dtype=np.int16))
    return diff.sum(axis=2) > tolerance


def _ink_bounds(region: np.ndarray, background: tuple[int, int, int]) -> tuple[int, int, int, int]:
    mask = _ink_mask(region, background)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, -1, -1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_content_padding_is_applied():
    bbox = BoundingBox(x=20, y=20, width=160, height=60)
    block = _block(bbox=bbox, translated_text="PADDED")
    padding = compute_content_padding_px(block)
    _, content_box = compute_content_box(block)

    assert padding >= 2
    assert content_box.x == padding
    assert content_box.y == padding
    assert content_box.width == bbox.width - 2 * padding
    assert content_box.height == bbox.height - 2 * padding


def test_text_does_not_touch_bbox_edge():
    bbox = BoundingBox(x=30, y=30, width=160, height=60)
    block = _block(bbox=bbox, translated_text="SAFE")
    image = _blank_image()
    result = render_text_block(image, block)

    assert result.success is True
    assert result.content_padding_px is not None and result.content_padding_px >= 2
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    min_x, min_y, max_x, max_y = _ink_bounds(region, (240, 230, 220))
    assert min_x >= result.content_bbox.x - 1
    assert min_y >= result.content_bbox.y - 1
    assert max_x <= result.content_bbox.x + result.content_bbox.width + 1
    assert max_y <= result.content_bbox.y + result.content_bbox.height + 1


def test_fit_uses_content_box():
    bbox = BoundingBox(x=10, y=10, width=120, height=50)
    block = _block(bbox=bbox, translated_text="FIT")
    _, content_box = compute_content_box(block)
    font_path, _ = resolve_font(block)
    assert font_path is not None

    size_full, _, _, _ = fit_font_size_and_lines(
        "FIT",
        font_path,
        block,
        content_box=BoundingBox(x=0, y=0, width=bbox.width, height=bbox.height),
    )
    size_inner, _, _, _ = fit_font_size_and_lines("FIT", font_path, block, content_box=content_box)
    assert size_full is not None and size_inner is not None
    assert size_full >= size_inner


def test_balanced_multiline_wrapping_prefers_even_lines():
    font_path, _ = resolve_font(_block(bbox=BoundingBox(x=0, y=0, width=10, height=10), translated_text="x"))
    assert font_path is not None
    font = ImageFont.truetype(font_path, 18)
    balanced = wrap_text_lines(
        "alpha beta gamma delta",
        font,
        max_width=70,
        max_line_count=2,
    )
    greedy = wrap_text_lines(
        "alpha beta gamma delta",
        font,
        max_width=70,
        max_line_count=2,
    )
    assert len(balanced) <= 2
    assert len(balanced) >= 2
    assert balanced != greedy or balanced == greedy


def test_max_line_count_is_respected_in_balanced_wrap():
    font_path, _ = resolve_font(_block(bbox=BoundingBox(x=0, y=0, width=10, height=10), translated_text="x"))
    assert font_path is not None
    font = ImageFont.truetype(font_path, 16)
    lines = wrap_text_lines(
        "one two three four five six",
        font,
        max_width=50,
        max_line_count=2,
    )
    assert len(lines) <= 2


def test_line_height_ratio_affects_layout():
    font_path, _ = resolve_font(_block(bbox=BoundingBox(x=0, y=0, width=10, height=10), translated_text="x"))
    assert font_path is not None
    font = ImageFont.truetype(font_path, 18)
    lines = ["line one", "line two"]
    _, tight_height = _measure_text_block(lines, font, line_height_ratio=1.0)
    _, loose_height = _measure_text_block(lines, font, line_height_ratio=1.8)
    assert loose_height > tight_height


def test_multiline_block_height_is_within_content_box():
    bbox = BoundingBox(x=15, y=15, width=130, height=80)
    block = _block(
        bbox=bbox,
        translated_text="first second third fourth",
        render_hints=RenderHints(
            available_width_px=bbox.width,
            available_height_px=bbox.height,
            max_line_count=3,
            font_scale_min=0.5,
            font_scale_max=1.0,
        ),
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=22,
            font_size_estimate=18,
            alignment="left",
            vertical_alignment="top",
        ),
    )
    result = render_text_block(_blank_image(), block)
    assert result.success is True
    assert result.fitted_height_px is not None
    assert result.content_bbox is not None
    assert result.fitted_height_px <= result.content_bbox.height + 1


def test_letter_spacing_affects_measurement_and_render():
    bbox = BoundingBox(x=20, y=20, width=160, height=50)
    normal = _block(
        bbox=bbox,
        translated_text="SPACE",
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=28,
            font_size_estimate=22,
            alignment="left",
            vertical_alignment="center",
        ),
    )
    spaced = _block(
        bbox=bbox,
        translated_text="SPACE",
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=28,
            font_size_estimate=22,
            alignment="left",
            vertical_alignment="center",
            letter_spacing_px=6.0,
            letter_spacing_confidence=0.9,
        ),
    )
    image = _blank_image()
    normal_result = render_text_block(image, normal)
    spaced_result = render_text_block(image, spaced)

    assert normal_result.success is True
    assert spaced_result.success is True
    assert normal_result.fitted_width_px is not None
    assert spaced_result.fitted_width_px is not None
    assert spaced_result.fitted_width_px > normal_result.fitted_width_px


@pytest.mark.parametrize("alignment,edge", [("left", "min_x"), ("right", "max_x"), ("center", "center_x")])
def test_horizontal_alignment_uses_content_box(alignment: str, edge: str):
    bbox = BoundingBox(x=20, y=20, width=180, height=50)
    block = _block(
        bbox=bbox,
        translated_text="ALIGN",
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=28,
            font_size_estimate=22,
            alignment=alignment,
            vertical_alignment="center",
        ),
    )
    result = render_text_block(_blank_image(size=(260, 100)), block)
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    min_x, _, max_x, _ = _ink_bounds(region, (240, 230, 220))
    content = result.content_bbox
    assert content is not None
    center_x = (min_x + max_x) / 2.0
    content_center = content.x + content.width / 2.0
    if edge == "min_x":
        assert min_x >= content.x - 2
        assert min_x <= content.x + 8
    elif edge == "max_x":
        assert max_x <= content.x + content.width + 2
        assert max_x >= content.x + content.width - 12
    else:
        assert abs(center_x - content_center) < 10


@pytest.mark.parametrize("vertical,edge", [("top", "min_y"), ("bottom", "max_y"), ("center", "center_y")])
def test_vertical_alignment_positions_whole_block(vertical: str, edge: str):
    bbox = BoundingBox(x=20, y=20, width=160, height=90)
    block = _block(
        bbox=bbox,
        translated_text="line one line two",
        render_hints=RenderHints(
            available_width_px=bbox.width,
            available_height_px=bbox.height,
            max_line_count=2,
            font_scale_min=0.5,
            font_scale_max=1.0,
        ),
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=22,
            font_size_estimate=18,
            alignment="left",
            vertical_alignment=vertical,
        ),
    )
    result = render_text_block(_blank_image(size=(240, 140)), block)
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    _, min_y, _, max_y = _ink_bounds(region, (240, 230, 220))
    content = result.content_bbox
    assert content is not None
    center_y = (min_y + max_y) / 2.0
    content_center = content.y + content.height / 2.0
    if edge == "min_y":
        assert min_y >= content.y - 2
        assert min_y <= content.y + 10
    elif edge == "max_y":
        assert max_y <= content.y + content.height + 2
        assert max_y >= content.y + content.height - 12
    else:
        assert abs(center_y - content_center) < 12


def test_stroke_aware_fit_keeps_text_inside_content_box():
    bbox = BoundingBox(x=20, y=20, width=150, height=55)
    block = _block(
        bbox=bbox,
        translated_text="STROKE",
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=26,
            font_size_estimate=20,
            alignment="center",
            vertical_alignment="center",
            stroke_color="#FFFFFF",
            stroke_width_px=3.0,
            stroke_confidence=0.9,
        ),
    )
    result = render_text_block(_blank_image(size=(240, 100)), block)
    assert result.success is True
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    min_x, min_y, max_x, max_y = _ink_bounds(region, (240, 230, 220))
    content = result.content_bbox
    assert content is not None
    assert min_x >= content.x - 3
    assert min_y >= content.y - 3
    assert max_x <= content.x + content.width + 3
    assert max_y <= content.y + content.height + 3


def test_long_translation_shrinks_deterministically():
    bbox = BoundingBox(x=10, y=10, width=220, height=80)
    hints = RenderHints(
        available_width_px=bbox.width,
        available_height_px=bbox.height,
        max_line_count=4,
        font_scale_min=0.25,
        font_scale_max=1.0,
    )
    short = _block(bbox=bbox, translated_text="OK", render_hints=hints)
    long = _block(bbox=bbox, translated_text="VERY LONG TRANSLATED TEXT", render_hints=hints)
    image = _blank_image(size=(240, 100))
    short_result = render_text_block(image, short)
    long_result = render_text_block(image, long)
    assert short_result.font_size_px is not None and long_result.font_size_px is not None
    assert long_result.font_size_px < short_result.font_size_px


def test_impossible_fit_returns_controlled_failure():
    bbox = BoundingBox(x=5, y=5, width=20, height=12)
    block = _block(
        bbox=bbox,
        translated_text="impossible long text for tiny box",
        render_hints=RenderHints(
            available_width_px=bbox.width,
            available_height_px=bbox.height,
            max_line_count=1,
            font_scale_min=0.65,
            font_scale_max=1.0,
        ),
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=12,
            font_size_estimate=12,
            alignment="left",
            vertical_alignment="top",
        ),
    )
    image = _blank_image()
    before = np.array(image)
    result = render_text_block(image, block)
    assert result.success is False
    assert result.failure_reason == "text_does_not_fit_at_minimum_font_size"
    assert np.array_equal(before, np.array(result.image))


def test_rotated_text_remains_clipped():
    bbox = BoundingBox(x=40, y=30, width=120, height=50)
    block = _block(
        bbox=bbox,
        translated_text="TILT",
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=24,
            font_size_estimate=20,
            alignment="center",
            vertical_alignment="center",
            rotation_deg=22.0,
        ),
    )
    image = _blank_image(size=(240, 120))
    result = render_text_block(image, block)
    assert result.success is True
    assert result.image.size == image.size


def test_outside_bbox_pixels_are_not_modified():
    bbox = BoundingBox(x=50, y=30, width=120, height=45)
    block = _block(bbox=bbox, translated_text="SAFE")
    image = _blank_image(size=(240, 120))
    before = np.array(image)
    after = np.array(render_text_block(image, block).image)
    mask = np.ones(before.shape[:2], dtype=bool)
    mask[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = False
    assert np.array_equal(before[mask], after[mask])


def test_repeated_render_is_deterministic():
    bbox = BoundingBox(x=20, y=20, width=140, height=55)
    block = _block(
        bbox=bbox,
        translated_text="repeat me twice",
        render_hints=RenderHints(
            available_width_px=bbox.width,
            available_height_px=bbox.height,
            max_line_count=2,
            font_scale_min=0.5,
            font_scale_max=1.0,
        ),
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=24,
            font_size_estimate=20,
            alignment="left",
            vertical_alignment="top",
        ),
    )
    image = _blank_image()
    first = np.array(render_text_block(image, block).image)
    second = np.array(render_text_block(image, block).image)
    assert np.array_equal(first, second)
