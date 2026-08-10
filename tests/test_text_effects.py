"""Unit tests for Renderer R0.6 geometry and visual effects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import (
    OCRData,
    RenderGeometry,
    RenderHints,
    ReviewInfo,
    TextBlock,
    VisualStyle,
)
from imageloc.renderer.text_effects import (
    GRADIENT_MIN_CONFIDENCE,
    SHADOW_MIN_CONFIDENCE,
    _resolve_fill_layer,
    apply_text_effects,
    render_glyph_alpha_mask,
    resolve_perspective_target_quad,
    rotate_layer,
    validate_perspective_quad,
)
from imageloc.renderer.text_renderer import (
    _compute_text_layout,
    compute_content_box,
    render_text_block,
)

pytestmark = pytest.mark.skipif(
    not (Path(__import__("os").environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf").is_file(),
    reason="Windows Unicode fonts required for text effects tests",
)


def _block(
    *,
    block_id: str = "block_001",
    bbox: BoundingBox,
    translated_text: str = "SALE",
    visual: VisualStyle | None = None,
    render_geometry: RenderGeometry | None = None,
    polygon: list[tuple[int, int]] | None = None,
) -> TextBlock:
    style = visual or VisualStyle(
        background_color="#FFFFFF",
        text_color="#000000",
        estimated_text_height_px=bbox.height,
        font_size_estimate=max(8, bbox.height - 4),
        alignment="center",
        vertical_alignment="center",
    )
    poly = polygon or [
        (bbox.x, bbox.y),
        (bbox.x + bbox.width, bbox.y),
        (bbox.x + bbox.width, bbox.y + bbox.height),
        (bbox.x, bbox.y + bbox.height),
    ]
    return TextBlock(
        id=block_id,
        original_text="SALE",
        translated_text=translated_text,
        source_language="EN",
        confidence=0.9,
        bbox=bbox,
        polygon=poly,
        ocr=OCRData(raw_text="SALE", confidence=0.9),
        visual=style,
        render_hints=RenderHints(available_width_px=bbox.width, available_height_px=bbox.height),
        review=ReviewInfo(),
        render_geometry=render_geometry or RenderGeometry(mask_bbox=bbox, mask_confidence=0.8),
    )


def _blank_image(size=(260, 140), color=(240, 230, 220)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _ink_mask(image: Image.Image, bg_color=(240, 230, 220), threshold: int = 15) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    bg = np.array(bg_color, dtype=np.int16)
    return np.abs(arr.astype(np.int16) - bg).sum(axis=2) > threshold


def _layout_and_effects_inputs(block: TextBlock):
    from imageloc.renderer.text_renderer import resolve_font

    font_path, _ = resolve_font(block)
    assert font_path is not None
    rendered = render_text_block(_blank_image(size=(420, 180)), block)
    assert rendered.success is True
    assert rendered.font_size_px is not None
    _, content = compute_content_box(block)
    layout = _compute_text_layout(
        block,
        font_path=font_path,
        font_size_px=rendered.font_size_px,
        lines=rendered.rendered_lines,
        content_box=content,
        letter_spacing_px=0.0,
        line_height_ratio=1.2,
        stroke_width=0.0,
    )
    return layout, rendered.font_size_px, rendered.rendered_lines


def _rotation_alpha_sum(
    block: TextBlock,
    layout,
    *,
    rotation_deg: float,
    preserve_ink: bool,
    with_shadow: bool = False,
    with_stroke: bool = False,
) -> int:
    mask = render_glyph_alpha_mask(block, layout)
    working = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    if with_shadow:
        from imageloc.renderer.text_effects import _render_shadow_layer, _shadow_settings

        shadow_settings = _shadow_settings(block)
        assert shadow_settings is not None
        color, opacity, offset_x, offset_y, blur = shadow_settings
        shadow_layer = _render_shadow_layer(
            mask,
            color=color,
            opacity=opacity,
            offset_x=offset_x,
            offset_y=offset_y,
            blur_radius=blur,
        )
        working = Image.alpha_composite(working, shadow_layer)
    fill_layer, _, _ = _resolve_fill_layer(mask, block, (20, 20, 20), 255)
    working = Image.alpha_composite(working, fill_layer)
    if with_stroke:
        from imageloc.renderer.text_effects import _apply_stroke_layer

        working = _apply_stroke_layer(
            working,
            block,
            layout,
            stroke_width=2.0,
            stroke_color=(255, 255, 255),
            text_opacity=255,
        )
    center = (block.bbox.width / 2.0, block.bbox.height / 2.0)
    rotated, _ = rotate_layer(
        working,
        rotation_deg,
        center=center,
        preserve_ink=preserve_ink,
    )
    return int(np.array(rotated)[:, :, 3].sum())


def _ink_outside_bbox(ink: np.ndarray, bbox: BoundingBox, *, pad: int = 25) -> int:
    y1 = max(0, bbox.y - pad)
    y2 = min(ink.shape[0], bbox.y + bbox.height + pad)
    x1 = max(0, bbox.x - pad)
    x2 = min(ink.shape[1], bbox.x + bbox.width + pad)
    region = ink[y1:y2, x1:x2].copy()
    inner_y1 = bbox.y - y1
    inner_y2 = inner_y1 + bbox.height
    inner_x1 = bbox.x - x1
    inner_x2 = inner_x1 + bbox.width
    region[inner_y1:inner_y2, inner_x1:inner_x2] = False
    return int(region.sum())


def _demo_rotation_block(*, rotation_deg: float, translated_text: str) -> TextBlock:
    bbox = BoundingBox(x=220, y=20, width=180, height=50)
    return _block(
        bbox=bbox,
        translated_text=translated_text,
        visual=VisualStyle(
            text_color="#141414",
            font_size_estimate=22,
            estimated_text_height_px=40,
            alignment="center",
            vertical_alignment="center",
            rotation_deg=rotation_deg,
        ),
    )


def test_positive_rotation():
    bbox = BoundingBox(x=30, y=30, width=120, height=50)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=22,
            estimated_text_height_px=40,
            rotation_deg=20.0,
        ),
    )
    result = render_text_block(_blank_image(size=(220, 120)), block)
    assert result.success is True
    assert result.applied_rotation == pytest.approx(20.0)


def test_negative_rotation():
    bbox = BoundingBox(x=30, y=30, width=120, height=50)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=22,
            estimated_text_height_px=40,
            rotation_deg=-20.0,
        ),
    )
    result = render_text_block(_blank_image(size=(220, 120)), block)
    assert result.success is True
    assert result.applied_rotation == pytest.approx(-20.0)


def test_fractional_rotation():
    bbox = BoundingBox(x=30, y=30, width=120, height=50)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=22,
            estimated_text_height_px=40,
            rotation_deg=12.5,
        ),
    )
    result = render_text_block(_blank_image(size=(220, 120)), block)
    assert result.applied_rotation == pytest.approx(12.5)


def test_canvas_size_unchanged():
    bbox = BoundingBox(x=40, y=30, width=100, height=45)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=20,
            estimated_text_height_px=35,
            rotation_deg=15.0,
        ),
    )
    image = _blank_image(size=(220, 120))
    result = render_text_block(image, block)
    assert result.image.size == image.size


def test_rgb_preserved():
    bbox = BoundingBox(x=20, y=20, width=100, height=40)
    block = _block(bbox=bbox)
    result = render_text_block(_blank_image(), block)
    assert result.image.mode == "RGB"


def test_rgba_alpha_preserved_outside_bbox():
    bbox = BoundingBox(x=20, y=20, width=100, height=40)
    block = _block(bbox=bbox)
    image = Image.new("RGBA", (180, 90), (240, 230, 220, 128))
    before = np.array(image.getchannel("A"))
    result = render_text_block(image, block)
    after = np.array(result.image.getchannel("A"))
    outside = np.ones(before.shape, dtype=bool)
    outside[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = False
    assert np.array_equal(before[outside], after[outside])


def test_perspective_valid_quadrilateral():
    bbox = BoundingBox(x=50, y=40, width=140, height=70)
    quad = [(50, 40), (190, 45), (185, 110), (55, 105)]
    block = _block(
        bbox=bbox,
        translated_text="WARP",
        visual=VisualStyle(text_color="#101010", font_size_estimate=20, estimated_text_height_px=35),
        render_geometry=RenderGeometry(
            mask_bbox=bbox,
            mask_confidence=0.8,
            perspective_detected=True,
            perspective_quad=quad,
        ),
        polygon=quad,
    )
    result = render_text_block(_blank_image(size=(280, 180)), block)
    assert result.success is True
    assert result.perspective_applied is True


def test_perspective_degenerate_fallback():
    bbox = BoundingBox(x=10, y=10, width=80, height=40)
    quad = [(10, 10), (12, 10), (11, 11), (10, 12)]
    block = _block(
        bbox=bbox,
        visual=VisualStyle(text_color="#000000", font_size_estimate=18, estimated_text_height_px=30),
        render_geometry=RenderGeometry(
            mask_bbox=bbox,
            mask_confidence=0.8,
            perspective_detected=True,
            perspective_quad=quad,
        ),
        polygon=quad,
    )
    result = render_text_block(_blank_image(), block)
    assert result.success is True
    assert result.perspective_applied is False
    assert any("fallback" in warning for warning in result.warnings)


def test_perspective_self_intersection_fallback():
    points = [(0, 0), (100, 0), (0, 50), (100, 50)]
    valid, reason = validate_perspective_quad(points, max_width=120, max_height=60)
    assert valid is False
    assert reason is not None


def test_perspective_out_of_bounds_safe():
    points = [(0, 0), (5000, 0), (5000, 5000), (0, 5000)]
    valid, reason = validate_perspective_quad(points, max_width=120, max_height=60)
    assert valid is False


def test_shadow_applied():
    bbox = BoundingBox(x=20, y=20, width=120, height=45)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=22,
            estimated_text_height_px=35,
            shadow_color="#000000",
            shadow_opacity=0.6,
            shadow_offset_x_px=3.0,
            shadow_offset_y_px=3.0,
            shadow_blur_px=2.0,
            shadow_confidence=0.9,
        ),
    )
    result = render_text_block(_blank_image(size=(200, 100)), block)
    assert result.shadow_applied is True


def test_shadow_low_confidence_skipped():
    bbox = BoundingBox(x=20, y=20, width=120, height=45)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=22,
            estimated_text_height_px=35,
            shadow_color="#000000",
            shadow_confidence=0.1,
        ),
    )
    result = render_text_block(_blank_image(size=(200, 100)), block)
    assert result.shadow_applied is False


def test_shadow_does_not_corrupt_neighbors():
    bbox = BoundingBox(x=60, y=30, width=100, height=40)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=20,
            estimated_text_height_px=30,
            shadow_color="#000000",
            shadow_opacity=0.8,
            shadow_offset_x_px=4.0,
            shadow_offset_y_px=4.0,
            shadow_blur_px=3.0,
            shadow_confidence=0.9,
        ),
    )
    image = _blank_image(size=(240, 120))
    before = np.array(image)
    result = render_text_block(image, block)
    after = np.array(result.image)
    mask = np.ones(before.shape[:2], dtype=bool)
    mask[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = False
    assert np.array_equal(before[mask], after[mask])


def test_gradient_text_applied():
    bbox = BoundingBox(x=20, y=20, width=140, height=50)
    block = _block(
        bbox=bbox,
        translated_text="GRAD",
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=22,
            estimated_text_height_px=35,
            fill_type="gradient",
            gradient_start_color="#FF0000",
            gradient_end_color="#0000FF",
            gradient_angle_deg=0.0,
            gradient_confidence=0.9,
        ),
    )
    result = render_text_block(_blank_image(size=(220, 100)), block)
    assert result.gradient_applied is True
    region = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    assert region[:, :, 0].max() > region[:, :, 2].min()


def test_gradient_fallback_to_solid():
    bbox = BoundingBox(x=20, y=20, width=120, height=45)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#224488",
            font_size_estimate=20,
            estimated_text_height_px=30,
            fill_type="gradient",
            gradient_confidence=0.1,
        ),
    )
    result = render_text_block(_blank_image(), block)
    assert result.gradient_applied is False
    assert any("gradient" in warning for warning in result.warnings)


def test_stroke_and_gradient():
    bbox = BoundingBox(x=20, y=20, width=140, height=50)
    block = _block(
        bbox=bbox,
        translated_text="BOLD",
        visual=VisualStyle(
            text_color="#FF0000",
            font_size_estimate=22,
            estimated_text_height_px=35,
            fill_type="gradient",
            gradient_start_color="#FF0000",
            gradient_end_color="#FF8800",
            gradient_confidence=GRADIENT_MIN_CONFIDENCE,
            stroke_color="#FFFFFF",
            stroke_width_px=2.0,
            stroke_confidence=0.9,
        ),
    )
    result = render_text_block(_blank_image(size=(220, 100)), block)
    assert result.success is True
    assert result.gradient_applied is True


def test_rotation_and_shadow():
    bbox = BoundingBox(x=40, y=30, width=120, height=50)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=20,
            estimated_text_height_px=32,
            rotation_deg=18.0,
            shadow_color="#000000",
            shadow_confidence=SHADOW_MIN_CONFIDENCE,
            shadow_blur_px=2.0,
        ),
    )
    result = render_text_block(_blank_image(size=(220, 120)), block)
    assert result.shadow_applied is True
    assert result.applied_rotation == pytest.approx(18.0)


def test_perspective_and_gradient():
    bbox = BoundingBox(x=40, y=30, width=130, height=60)
    quad = [(40, 30), (170, 35), (165, 90), (45, 85)]
    block = _block(
        bbox=bbox,
        translated_text="PG",
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=20,
            estimated_text_height_px=32,
            fill_type="gradient",
            gradient_start_color="#00AAFF",
            gradient_end_color="#0044AA",
            gradient_confidence=0.9,
        ),
        render_geometry=RenderGeometry(
            mask_bbox=bbox,
            mask_confidence=0.8,
            perspective_detected=True,
            perspective_quad=quad,
        ),
        polygon=quad,
    )
    result = render_text_block(_blank_image(size=(240, 140)), block)
    assert result.gradient_applied is True
    assert result.perspective_applied is True


def test_deterministic_repeated_render():
    bbox = BoundingBox(x=20, y=20, width=120, height=45)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=20,
            estimated_text_height_px=30,
            rotation_deg=10.0,
            shadow_color="#000000",
            shadow_confidence=0.8,
            fill_type="gradient",
            gradient_start_color="#AA0000",
            gradient_end_color="#0000AA",
            gradient_confidence=0.8,
        ),
    )
    image = _blank_image(size=(200, 100))
    first = np.array(render_text_block(image, block).image)
    second = np.array(render_text_block(image, block).image)
    assert np.array_equal(first, second)


def test_outside_allowed_region_unchanged():
    bbox = BoundingBox(x=80, y=50, width=120, height=45)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=18,
            estimated_text_height_px=28,
            rotation_deg=25.0,
        ),
    )
    image = _blank_image(size=(320, 200))
    before = np.array(image)
    after = np.array(render_text_block(image, block).image)
    pad = 40
    mask = np.ones(before.shape[:2], dtype=bool)
    mask[
        max(0, bbox.y - pad) : bbox.y + bbox.height + pad,
        max(0, bbox.x - pad) : bbox.x + bbox.width + pad,
    ] = False
    assert np.array_equal(before[mask], after[mask])


def test_warnings_controlled_for_bad_geometry():
    bbox = BoundingBox(x=0, y=0, width=50, height=20)
    block = _block(
        bbox=bbox,
        render_geometry=RenderGeometry(
            mask_bbox=bbox,
            mask_confidence=0.8,
            perspective_detected=True,
            perspective_quad=[(0, 0), (1, 0), (2, 2), (0, 1)],
        ),
    )
    quad, warnings = resolve_perspective_target_quad(block)
    assert quad is None
    assert warnings


def test_no_unhandled_exceptions_for_bad_geometry():
    bbox = BoundingBox(x=10, y=10, width=60, height=30)
    block = _block(
        bbox=bbox,
        visual=VisualStyle(text_color="#000000", font_size_estimate=16, estimated_text_height_px=24),
        render_geometry=RenderGeometry(
            mask_bbox=bbox,
            mask_confidence=0.8,
            perspective_detected=True,
            perspective_quad=[(10, 10), (5000, 10), (5000, 5000), (10, 5000)],
        ),
        polygon=[(10, 10), (5000, 10), (5000, 5000), (10, 5000)],
    )
    result = render_text_block(_blank_image(size=(200, 120)), block)
    assert result.success is True
    assert result.perspective_applied is False


def test_positive_rotation_ink_not_internally_clipped():
    block = _demo_rotation_block(rotation_deg=20.0, translated_text="Поворот +20")
    layout, _, _ = _layout_and_effects_inputs(block)
    expanded = _rotation_alpha_sum(block, layout, rotation_deg=20.0, preserve_ink=True)
    clipped = _rotation_alpha_sum(block, layout, rotation_deg=20.0, preserve_ink=False)
    assert expanded > clipped

    image = _blank_image(size=(500, 150))
    result = render_text_block(image, block)
    ink = _ink_mask(result.image)
    bbox = block.bbox
    assert ink[max(0, bbox.y - 25) : bbox.y, bbox.x : bbox.x + bbox.width].sum() >= 20


def test_negative_rotation_ink_not_internally_clipped():
    block = _demo_rotation_block(rotation_deg=-20.0, translated_text="Rotate -20")
    layout, _, _ = _layout_and_effects_inputs(block)
    expanded = _rotation_alpha_sum(block, layout, rotation_deg=-20.0, preserve_ink=True)
    clipped = _rotation_alpha_sum(block, layout, rotation_deg=-20.0, preserve_ink=False)
    assert expanded > clipped

    image = _blank_image(size=(500, 150))
    result = render_text_block(image, block)
    ink = _ink_mask(result.image)
    assert _ink_outside_bbox(ink, block.bbox) >= 8


def test_fractional_rotation_ink_preserved():
    block = _demo_rotation_block(rotation_deg=12.5, translated_text="Longer fractional rotation text")
    layout, _, _ = _layout_and_effects_inputs(block)
    expanded = _rotation_alpha_sum(block, layout, rotation_deg=12.5, preserve_ink=True)
    clipped = _rotation_alpha_sum(block, layout, rotation_deg=12.5, preserve_ink=False)
    assert expanded > clipped


def test_rotation_with_shadow_preserves_ink():
    bbox = BoundingBox(x=40, y=30, width=160, height=50)
    block = _block(
        bbox=bbox,
        translated_text="ShadowRot",
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=22,
            estimated_text_height_px=36,
            rotation_deg=18.0,
            shadow_color="#000000",
            shadow_opacity=0.6,
            shadow_offset_x_px=4.0,
            shadow_offset_y_px=4.0,
            shadow_blur_px=3.0,
            shadow_confidence=0.9,
        ),
    )
    layout, _, _ = _layout_and_effects_inputs(block)
    expanded = _rotation_alpha_sum(
        block,
        layout,
        rotation_deg=18.0,
        preserve_ink=True,
        with_shadow=True,
    )
    clipped = _rotation_alpha_sum(
        block,
        layout,
        rotation_deg=18.0,
        preserve_ink=False,
        with_shadow=True,
    )
    assert expanded > clipped
    result = render_text_block(_blank_image(size=(320, 160)), block)
    assert result.shadow_applied is True


def test_rotation_with_stroke_preserves_ink():
    bbox = BoundingBox(x=40, y=30, width=160, height=50)
    block = _block(
        bbox=bbox,
        translated_text="StrokeRot",
        visual=VisualStyle(
            text_color="#101010",
            font_size_estimate=22,
            estimated_text_height_px=36,
            rotation_deg=16.0,
            stroke_color="#FFFFFF",
            stroke_width_px=2.0,
            stroke_confidence=0.9,
        ),
    )
    layout, _, _ = _layout_and_effects_inputs(block)
    expanded = _rotation_alpha_sum(
        block,
        layout,
        rotation_deg=16.0,
        preserve_ink=True,
        with_stroke=True,
    )
    clipped = _rotation_alpha_sum(
        block,
        layout,
        rotation_deg=16.0,
        preserve_ink=False,
        with_stroke=True,
    )
    assert expanded > clipped


def test_rotation_repeated_render_is_deterministic():
    block = _demo_rotation_block(rotation_deg=20.0, translated_text="Deterministic")
    image = _blank_image(size=(500, 150))
    first = np.array(render_text_block(image, block).image)
    second = np.array(render_text_block(image, block).image)
    assert np.array_equal(first, second)
