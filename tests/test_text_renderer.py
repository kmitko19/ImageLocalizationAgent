"""Unit tests for Renderer R0.4 text rendering."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from imageloc.config import PROJECT_ROOT
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
from imageloc.renderer.text_renderer import (
    render_localized_image,
    render_text_block,
    render_text_blocks,
    resolve_font,
    resolve_localized_image_path,
    save_localized_image,
    wrap_text_lines,
)
from imageloc.utils.text_mask_analyzer import MASK_FOREGROUND_VALUE

pytestmark = pytest.mark.skipif(
    not (Path(__import__("os").environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf").is_file(),
    reason="Windows Unicode fonts required for text renderer tests",
)


def _block(
    *,
    block_id: str = "block_001",
    bbox: BoundingBox,
    translated_text: str | None = "SALE",
    background: BackgroundInfo | None = None,
    render_hints: RenderHints | None = None,
    visual: VisualStyle | None = None,
    mask_confidence: float = 0.8,
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
        background=background,
        render_geometry=RenderGeometry(mask_bbox=bbox, mask_confidence=mask_confidence),
    )


def _blank_image(size=(240, 120), color=(240, 230, 220)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _mask_for_bbox(image: Image.Image, bbox: BoundingBox, *, dark_text: bool = True) -> np.ndarray:
    patch = np.array(image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    mask = np.zeros((bbox.height, bbox.width), dtype=np.uint8)
    threshold = 200 if dark_text else 120
    if dark_text:
        mask[np.mean(patch, axis=2) < threshold] = MASK_FOREGROUND_VALUE
    else:
        mask[np.mean(patch, axis=2) > threshold] = MASK_FOREGROUND_VALUE
    return mask


def _ink_mask(region: np.ndarray, background: tuple[int, int, int], tolerance: int = 35) -> np.ndarray:
    diff = np.abs(region.astype(np.int16) - np.array(background, dtype=np.int16))
    return diff.sum(axis=2) > tolerance


def _ink_centroid(region: np.ndarray, background: tuple[int, int, int]) -> tuple[float, float]:
    mask = _ink_mask(region, background)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def test_single_line_text_fits_in_bbox():
    bbox = BoundingBox(x=20, y=20, width=120, height=40)
    block = _block(bbox=bbox, translated_text="SALE")
    image = _blank_image()
    result = render_text_block(image, block)

    assert result.success is True
    assert result.rendered_lines == ["SALE"]
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    assert np.count_nonzero(_ink_mask(region, (240, 230, 220))) > 0


def test_text_color_is_applied():
    bbox = BoundingBox(x=10, y=10, width=100, height=40)
    block = _block(
        bbox=bbox,
        translated_text="RED",
        visual=VisualStyle(
            text_color="#FF0000",
            background_color="#FFFFFF",
            estimated_text_height_px=30,
            font_size_estimate=24,
            alignment="center",
            vertical_alignment="center",
        ),
    )
    image = _blank_image()
    result = render_text_block(image, block)
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    ink = region[_ink_mask(region, (240, 230, 220))]
    assert ink[:, 0].mean() > ink[:, 1].mean()
    assert ink[:, 0].mean() > ink[:, 2].mean()


def test_center_alignment():
    bbox = BoundingBox(x=30, y=20, width=120, height=50)
    block = _block(
        bbox=bbox,
        translated_text="MID",
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=30,
            font_size_estimate=24,
            alignment="center",
            vertical_alignment="center",
        ),
    )
    image = _blank_image()
    result = render_text_block(image, block)
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    cx, cy = _ink_centroid(region, (240, 230, 220))
    assert abs(cx - bbox.width / 2) < 12
    assert abs(cy - bbox.height / 2) < 12


@pytest.mark.parametrize("alignment", ["left", "right"])
def test_horizontal_alignment(alignment: str):
    bbox = BoundingBox(x=20, y=20, width=140, height=40)
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
    image = _blank_image()
    result = render_text_block(image, block)
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    cx, _ = _ink_centroid(region, (240, 230, 220))
    if alignment == "left":
        assert cx < bbox.width * 0.45
    else:
        assert cx > bbox.width * 0.55


def test_multiline_wrapping():
    bbox = BoundingBox(x=10, y=10, width=80, height=70)
    block = _block(
        bbox=bbox,
        translated_text="one two three four",
        render_hints=RenderHints(
            available_width_px=bbox.width,
            available_height_px=bbox.height,
            max_line_count=3,
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
    result = render_text_block(image, block)

    assert result.success is True
    assert len(result.rendered_lines) >= 2


def test_max_line_count_is_respected():
    font_path, _ = resolve_font(_block(bbox=BoundingBox(x=0, y=0, width=10, height=10), translated_text="x"))
    assert font_path is not None
    font = ImageFont.truetype(font_path, 18)
    lines = wrap_text_lines(
        "alpha beta gamma delta epsilon",
        font,
        max_width=60,
        max_line_count=2,
    )
    assert len(lines) <= 2


def test_long_text_reduces_font_size():
    bbox = BoundingBox(x=10, y=10, width=200, height=70)
    hints = RenderHints(
        available_width_px=bbox.width,
        available_height_px=bbox.height,
        max_line_count=4,
        font_scale_min=0.25,
        font_scale_max=1.0,
    )
    image = _blank_image(size=(240, 100))
    short_block = _block(bbox=bbox, translated_text="OK", render_hints=hints)
    long_block = _block(bbox=bbox, translated_text="VERY LONG TRANSLATED TEXT", render_hints=hints)

    short_result = render_text_block(image, short_block)
    long_result = render_text_block(image, long_block)

    assert short_result.success is True
    assert long_result.success is True
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
    assert any("does not fit" in warning for warning in result.warnings)
    assert np.array_equal(before, np.array(result.image))


def test_cyrillic_text_renders():
    bbox = BoundingBox(x=15, y=15, width=140, height=45)
    block = _block(
        bbox=bbox,
        translated_text="Привет",
        visual=VisualStyle(
            text_color="#101010",
            estimated_text_height_px=30,
            font_size_estimate=24,
            alignment="center",
            vertical_alignment="center",
        ),
    )
    image = _blank_image()
    result = render_text_block(image, block)

    assert result.success is True
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    assert np.count_nonzero(_ink_mask(region, (240, 230, 220))) > 20


def test_font_resolver_finds_unicode_fallback():
    block = _block(bbox=BoundingBox(x=0, y=0, width=80, height=30), translated_text="Тест")
    font_path, warnings = resolve_font(block)

    assert font_path is not None
    assert Path(font_path).is_file()
    assert not any("no Unicode-capable" in warning for warning in warnings)


def test_rotation_preserves_image_size():
    bbox = BoundingBox(x=40, y=30, width=100, height=40)
    block = _block(
        bbox=bbox,
        translated_text="TILT",
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=24,
            font_size_estimate=20,
            alignment="center",
            vertical_alignment="center",
            rotation_deg=25.0,
        ),
    )
    image = _blank_image(size=(220, 120))
    result = render_text_block(image, block)

    assert result.success is True
    assert result.image.size == image.size


def test_outside_bbox_pixels_are_not_modified():
    bbox = BoundingBox(x=50, y=30, width=100, height=40)
    block = _block(bbox=bbox, translated_text="SAFE")
    image = _blank_image(size=(220, 120))
    before = np.array(image)
    result = render_text_block(image, block)
    after = np.array(result.image)

    mask = np.ones(before.shape[:2], dtype=bool)
    mask[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = False
    assert np.array_equal(before[mask], after[mask])


def test_empty_translated_text_is_skipped_safely():
    bbox = BoundingBox(x=10, y=10, width=80, height=30)
    block = _block(bbox=bbox, translated_text="")
    image = _blank_image()
    before = np.array(image)
    result = render_text_block(image, block)

    assert result.success is True
    assert result.rendered_lines == []
    assert any("skipped" in warning for warning in result.warnings)
    assert np.array_equal(before, np.array(result.image))


def test_multiple_blocks_render_sequentially():
    image = _blank_image(size=(320, 120))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 120, 60), fill=(255, 255, 255))
    draw.text((25, 25), "A", fill=(0, 0, 0))
    draw.rectangle((150, 20, 280, 70), fill=(255, 255, 255))
    draw.text((155, 25), "LONG", fill=(0, 0, 0))

    block_a = _block(
        block_id="a",
        bbox=BoundingBox(x=20, y=20, width=100, height=40),
        translated_text="A",
    )
    block_b = _block(
        block_id="b",
        bbox=BoundingBox(x=150, y=20, width=130, height=50),
        translated_text="line one line two",
        render_hints=RenderHints(
            available_width_px=130,
            available_height_px=50,
            max_line_count=2,
            font_scale_min=0.5,
            font_scale_max=1.0,
        ),
        visual=VisualStyle(
            text_color="#000000",
            estimated_text_height_px=24,
            font_size_estimate=18,
            alignment="left",
            vertical_alignment="top",
        ),
    )
    rendered, results = render_text_blocks(image, [block_a, block_b])

    assert len(results) == 2
    assert all(result.success for result in results)
    assert rendered.size == image.size


def test_rgba_mode_is_preserved():
    bbox = BoundingBox(x=20, y=20, width=100, height=40)
    block = _block(bbox=bbox, translated_text="RGBA")
    image = Image.new("RGBA", (180, 90), (240, 230, 220, 128))
    before_alpha = np.array(image.getchannel("A"))
    result = render_text_block(image, block)

    assert result.image.mode == "RGBA"
    after_alpha = np.array(result.image.getchannel("A"))
    outside = np.ones(before_alpha.shape, dtype=bool)
    outside[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = False
    assert np.array_equal(before_alpha[outside], after_alpha[outside])


def test_save_localized_image_roundtrip():
    bbox = BoundingBox(x=10, y=10, width=80, height=30)
    block = _block(bbox=bbox, translated_text="SAVE")
    image = _blank_image()
    rendered = render_text_block(image, block).image

    output_dir = PROJECT_ROOT / "data/output"
    rel_path = save_localized_image(
        rendered,
        output_dir=output_dir,
        source_image_id="test_text_renderer_save",
    )
    abs_path = resolve_localized_image_path(rel_path)
    try:
        assert abs_path.is_file()
        assert rel_path.replace("\\", "/") == "data/output/test_text_renderer_save/render_assets/localized_image.png"
        loaded = Image.open(abs_path)
        assert loaded.size == rendered.size
    finally:
        shutil.rmtree(output_dir / "test_text_renderer_save", ignore_errors=True)


def test_render_localized_image_end_to_end():
    image = _blank_image(size=(260, 100))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 140, 60), fill=(30, 80, 160))
    draw.text((30, 28), "NEW", fill=(240, 240, 240))

    bbox = BoundingBox(x=20, y=20, width=120, height=40)
    block = _block(
        bbox=bbox,
        translated_text="НОВИНКА",
        background=BackgroundInfo(
            background_type="solid",
            dominant_color="#1E509F",
            preferred_restore_method="fill",
        ),
        visual=VisualStyle(
            text_color="#FFFFFF",
            background_color="#1E509F",
            estimated_text_height_px=28,
            font_size_estimate=22,
            alignment="center",
            vertical_alignment="center",
        ),
    )
    mask = np.zeros((bbox.height, bbox.width), dtype=np.uint8)
    mask[np.mean(np.array(image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width], axis=2) > 200] = (
        MASK_FOREGROUND_VALUE
    )

    localized, restore_results, render_results = render_localized_image(image, [block], masks={block.id: mask})

    assert localized.size == image.size
    assert len(restore_results) == 1
    assert restore_results[0].success is True
    assert len(render_results) == 1
    assert render_results[0].success is True
    region = np.array(localized)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    assert np.count_nonzero(_ink_mask(region, (30, 80, 160), tolerance=60)) > 10
