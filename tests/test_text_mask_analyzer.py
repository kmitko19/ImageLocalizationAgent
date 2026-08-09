"""Unit tests for R0.2C text mask and geometry analysis."""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from imageloc.config import PROJECT_ROOT
from imageloc.fusion.ocr_vision_fusion import FusedBlock
from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import BackgroundInfo, OCRData, RenderHints, ReviewInfo, VisualStyle
from imageloc.utils.text_mask_analyzer import (
    MASK_BACKGROUND_VALUE,
    MASK_FOREGROUND_VALUE,
    MASK_SAVE_MIN_CONFIDENCE,
    TextMaskAnalysis,
    analyze_perspective_quad,
    analyze_text_mask,
    analyze_text_mask_for_block,
    expand_text_mask,
    load_text_mask_png,
    normalize_quad_order,
    resolve_mask_path,
    save_text_mask_png,
)


def _fused_block(
    *,
    block_id: str = "block_001",
    bbox: BoundingBox | None = None,
    polygon: list[tuple[int, int]] | None = None,
    text: str = "SALE",
) -> FusedBlock:
    bbox = bbox or BoundingBox(x=20, y=20, width=120, height=40)
    polygon = polygon or [
        (bbox.x, bbox.y),
        (bbox.x + bbox.width, bbox.y),
        (bbox.x + bbox.width, bbox.y + bbox.height),
        (bbox.x, bbox.y + bbox.height),
    ]
    return FusedBlock(
        id=block_id,
        text=text,
        source_language="EN",
        confidence=0.9,
        bbox=bbox,
        polygon=polygon,
        ocr=OCRData(raw_text=text, confidence=0.9),
        vision=None,
        review=ReviewInfo(),
    )


def _draw_text_image(
    *,
    size=(200, 80),
    text: str = "SALE",
    fg=(0, 0, 0),
    bg=(255, 255, 255),
    position=(20, 20),
) -> np.ndarray:
    image = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, fill=fg)
    return np.array(image)


def _analyze_patch(
    image_array: np.ndarray,
    bbox: BoundingBox,
    *,
    polygon: list[tuple[int, int]] | None = None,
    visual: VisualStyle | None = None,
    background: BackgroundInfo | None = None,
) -> TextMaskAnalysis:
    polygon = polygon or [
        (bbox.x, bbox.y),
        (bbox.x + bbox.width, bbox.y),
        (bbox.x + bbox.width, bbox.y + bbox.height),
        (bbox.x, bbox.y + bbox.height),
    ]
    visual = visual or VisualStyle(
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=bbox.height,
    )
    patch = image_array[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    mask_bbox = BoundingBox(x=bbox.x, y=bbox.y, width=bbox.width, height=bbox.height)
    return analyze_text_mask(
        patch,
        mask_bbox,
        polygon=polygon,
        visual=visual,
        background=background,
        estimated_text_height=bbox.height,
    )


def _image_from_mask(
    shape_mask: np.ndarray,
    *,
    fg=(0, 0, 0),
    bg=(255, 255, 255),
) -> np.ndarray:
    image = np.full((shape_mask.shape[0], shape_mask.shape[1], 3), bg, dtype=np.uint8)
    image[shape_mask > 0] = fg
    return image


def _synthetic_i_shape(height: int = 70, width: int = 24) -> np.ndarray:
    shape = np.zeros((height, width), dtype=np.uint8)
    shape[18:65, 10:14] = MASK_FOREGROUND_VALUE
    shape[4:8, 9:15] = MASK_FOREGROUND_VALUE
    return shape


def _synthetic_yo_shape(height: int = 70, width: int = 34) -> np.ndarray:
    shape = np.zeros((height, width), dtype=np.uint8)
    shape[18:65, 12:16] = MASK_FOREGROUND_VALUE
    shape[18:65, 20:24] = MASK_FOREGROUND_VALUE
    shape[4:8, 11:15] = MASK_FOREGROUND_VALUE
    shape[4:8, 21:25] = MASK_FOREGROUND_VALUE
    return shape


def _synthetic_punctuation_line(width: int = 180, height: int = 40) -> np.ndarray:
    shape = np.zeros((height, width), dtype=np.uint8)
    shape[12:32, 8:18] = MASK_FOREGROUND_VALUE
    shape[12:32, 24:34] = MASK_FOREGROUND_VALUE
    shape[12:32, 40:50] = MASK_FOREGROUND_VALUE
    shape[12:32, 56:66] = MASK_FOREGROUND_VALUE
    shape[22:28, 72:76] = MASK_FOREGROUND_VALUE
    shape[12:32, 82:92] = MASK_FOREGROUND_VALUE
    shape[12:32, 98:108] = MASK_FOREGROUND_VALUE
    shape[12:32, 114:124] = MASK_FOREGROUND_VALUE
    shape[12:32, 130:140] = MASK_FOREGROUND_VALUE
    shape[12:32, 146:150] = MASK_FOREGROUND_VALUE
    shape[12:32, 156:160] = MASK_FOREGROUND_VALUE
    return shape


def _synthetic_numeric_punctuation(width: int = 160, height: int = 40) -> np.ndarray:
    shape = np.zeros((height, width), dtype=np.uint8)
    shape[10:30, 8:18] = MASK_FOREGROUND_VALUE
    shape[22:26, 22:26] = MASK_FOREGROUND_VALUE
    shape[10:30, 30:40] = MASK_FOREGROUND_VALUE
    shape[22:30, 44:48] = MASK_FOREGROUND_VALUE
    shape[18:22, 48:52] = MASK_FOREGROUND_VALUE
    shape[10:30, 58:68] = MASK_FOREGROUND_VALUE
    shape[10:30, 74:78] = MASK_FOREGROUND_VALUE
    shape[10:30, 84:94] = MASK_FOREGROUND_VALUE
    shape[10:30, 100:110] = MASK_FOREGROUND_VALUE
    return shape


def _count_components(mask: np.ndarray) -> int:
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return max(0, num_labels - 1)


def test_synthetic_i_preserves_stem_and_dot():
    shape = _synthetic_i_shape()
    image = _image_from_mask(shape)
    bbox = BoundingBox(x=0, y=0, width=shape.shape[1], height=shape.shape[0])
    visual = VisualStyle(
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=shape.shape[0],
    )

    analysis = _analyze_patch(image, bbox, visual=visual)

    assert analysis.mask[18:65, 10:14].any()
    assert analysis.mask[4:8, 9:15].any()
    assert _count_components(analysis.mask) >= 2


def test_synthetic_yo_preserves_stem_and_two_dots():
    shape = _synthetic_yo_shape()
    image = _image_from_mask(shape)
    bbox = BoundingBox(x=0, y=0, width=shape.shape[1], height=shape.shape[0])
    visual = VisualStyle(
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=shape.shape[0],
    )

    analysis = _analyze_patch(image, bbox, visual=visual)

    assert analysis.mask[18:65, 12:16].any()
    assert analysis.mask[18:65, 20:24].any()
    assert analysis.mask[4:8, 11:15].any()
    assert analysis.mask[4:8, 21:25].any()
    assert _count_components(analysis.mask) >= 4


def test_synthetic_punctuation_line_keeps_small_marks():
    shape = _synthetic_punctuation_line()
    image = _image_from_mask(shape)
    bbox = BoundingBox(x=0, y=0, width=shape.shape[1], height=shape.shape[0])
    visual = VisualStyle(
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=shape.shape[0],
    )

    analysis = _analyze_patch(image, bbox, visual=visual)

    assert analysis.mask[22:28, 72:76].any()
    assert analysis.mask[12:32, 146:150].any()
    assert analysis.mask[12:32, 156:160].any()


def test_synthetic_numeric_punctuation_keeps_dot_and_colon():
    shape = _synthetic_numeric_punctuation()
    image = _image_from_mask(shape)
    bbox = BoundingBox(x=0, y=0, width=shape.shape[1], height=shape.shape[0])
    visual = VisualStyle(
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=shape.shape[0],
    )

    analysis = _analyze_patch(image, bbox, visual=visual)

    assert analysis.mask[22:26, 22:26].any()
    assert analysis.mask[22:30, 44:48].any()
    assert analysis.mask[18:22, 48:52].any()


def test_isolated_single_pixel_noise_is_removed():
    shape = _synthetic_i_shape()
    canvas = np.full((100, 80, 3), 255, dtype=np.uint8)
    bbox = BoundingBox(x=0, y=0, width=shape.shape[1], height=shape.shape[0])
    canvas[bbox.y : bbox.y + shape.shape[0], bbox.x : bbox.x + shape.shape[1]] = _image_from_mask(shape)
    canvas[bbox.y + 2, bbox.x + 2] = (0, 0, 0)
    canvas[bbox.y + 60, bbox.x + 2] = (0, 0, 0)
    visual = VisualStyle(
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=shape.shape[0],
    )

    analysis = _analyze_patch(canvas, bbox, visual=visual)

    assert not analysis.mask[2, 2]
    assert not analysis.mask[60, 2]
    assert analysis.mask[4:8, 9:15].any()


def test_mask_path_uses_project_relative_convention():
    import shutil

    output_dir = PROJECT_ROOT / "data" / "output"
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[4:8, 5:11] = MASK_FOREGROUND_VALUE

    stored_path = save_text_mask_png(
        mask,
        output_dir=output_dir,
        source_image_id="mask-path-case",
        block_id="block_001",
    )

    try:
        assert not Path(stored_path).is_absolute()
        assert stored_path.startswith("data/output/")
        assert stored_path.endswith("render_assets/block_001_mask.png")

        resolved = resolve_mask_path(stored_path)
        assert resolved == output_dir / "mask-path-case" / "render_assets" / "block_001_mask.png"
        assert resolved.exists()
        assert load_text_mask_png(stored_path).shape == mask.shape
    finally:
        shutil.rmtree(output_dir / "mask-path-case", ignore_errors=True)


def test_black_text_on_white_mask_selects_foreground():
    image = _draw_text_image(text="SALE")
    bbox = BoundingBox(x=15, y=15, width=130, height=45)

    analysis = _analyze_patch(image, bbox)

    assert analysis.mask.shape == (bbox.height, bbox.width)
    assert np.count_nonzero(analysis.mask) > 0
    assert analysis.coverage_ratio < 1.0
    assert analysis.confidence >= 0.45
    assert analysis.mask.max() == MASK_FOREGROUND_VALUE
    assert analysis.mask.min() == MASK_BACKGROUND_VALUE


def test_white_text_on_dark_background_mask_works():
    image = _draw_text_image(text="SALE", fg=(255, 255, 255), bg=(20, 20, 20))
    bbox = BoundingBox(x=15, y=15, width=130, height=45)
    visual = VisualStyle(
        text_color="#FFFFFF",
        background_color="#141414",
        estimated_text_height_px=bbox.height,
    )

    analysis = _analyze_patch(image, bbox, visual=visual)

    assert np.count_nonzero(analysis.mask) > 0
    assert analysis.coverage_ratio < 1.0
    assert analysis.confidence >= 0.35


def test_colored_text_on_solid_background_builds_mask():
    image = Image.new("RGB", (200, 80), color=(240, 230, 220))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "SALE", fill=(180, 20, 40))
    bbox = BoundingBox(x=15, y=15, width=130, height=45)
    visual = VisualStyle(
        text_color="#B41428",
        background_color="#F0E6DC",
        estimated_text_height_px=bbox.height,
    )
    background = BackgroundInfo(background_type="solid", classification_confidence=0.9)

    analysis = _analyze_patch(np.array(image), bbox, visual=visual, background=background)

    assert np.count_nonzero(analysis.mask) > 0
    assert analysis.coverage_ratio < 1.0


def test_uniform_patch_does_not_create_confident_false_mask(tmp_path: Path):
    image = np.full((80, 160, 3), 240, dtype=np.uint8)
    bbox = BoundingBox(x=10, y=10, width=120, height=40)
    block = _fused_block(bbox=bbox)
    visual = VisualStyle(
        background_color="#F0F0F0",
        text_color="#101010",
        estimated_text_height_px=40,
    )

    result = analyze_text_mask_for_block(
        Image.fromarray(image),
        block,
        visual=visual,
        render_hints=RenderHints(available_width_px=120, available_height_px=40),
        background=BackgroundInfo(background_type="solid", classification_confidence=0.9),
        source_image_id="uniform-case",
        output_dir=tmp_path,
    )

    assert result.mask_analysis is not None
    assert result.mask_analysis.confidence < MASK_SAVE_MIN_CONFIDENCE
    assert result.render_geometry is None or result.render_geometry.mask_path is None
    assets_dir = tmp_path / "uniform-case" / "render_assets"
    assert not assets_dir.exists() or not any(assets_dir.glob("*_mask.png"))


def test_tiny_bbox_is_safe_without_exception(tmp_path: Path):
    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    bbox = BoundingBox(x=30, y=30, width=3, height=3)
    block = _fused_block(bbox=bbox, block_id="tiny")

    result = analyze_text_mask_for_block(
        image,
        block,
        visual=VisualStyle(background_color="#FFFFFF", estimated_text_height_px=3),
        render_hints=RenderHints(available_width_px=3, available_height_px=3),
        background=None,
        source_image_id="tiny-case",
        output_dir=tmp_path,
    )

    assert result.mask_analysis is None or result.mask_analysis.confidence <= 0.5
    assert result.render_geometry is None or (result.render_geometry.mask_confidence or 0.0) <= 0.5


def test_noisy_background_has_lower_confidence_than_simple_contrast():
    simple = _draw_text_image(text="SALE")
    rng = np.random.default_rng(3)
    noisy_bg = rng.integers(180, 256, size=(80, 200, 3), dtype=np.uint8)
    noisy_image = Image.fromarray(noisy_bg)
    draw = ImageDraw.Draw(noisy_image)
    draw.text((20, 20), "SALE", fill=(0, 0, 0))
    noisy = np.array(noisy_image)

    bbox = BoundingBox(x=15, y=15, width=130, height=45)
    visual = VisualStyle(
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=45,
    )
    solid_background = BackgroundInfo(background_type="solid", classification_confidence=0.9)
    noisy_background = BackgroundInfo(background_type="texture", classification_confidence=0.75)

    simple_analysis = _analyze_patch(simple, bbox, visual=visual, background=solid_background)
    noisy_analysis = _analyze_patch(noisy, bbox, visual=visual, background=noisy_background)

    assert simple_analysis.confidence > noisy_analysis.confidence


def test_complex_background_metadata_lowers_confidence():
    image = _draw_text_image(text="SALE")
    bbox = BoundingBox(x=15, y=15, width=130, height=45)
    visual = VisualStyle(
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=45,
    )

    solid_analysis = _analyze_patch(
        image,
        bbox,
        visual=visual,
        background=BackgroundInfo(background_type="solid", classification_confidence=0.9),
    )
    complex_analysis = _analyze_patch(
        image,
        bbox,
        visual=visual,
        background=BackgroundInfo(background_type="complex", classification_confidence=0.8),
    )

    assert solid_analysis.confidence > complex_analysis.confidence


def test_mask_is_not_full_bbox_rectangle():
    image = _draw_text_image(text="HELLO")
    bbox = BoundingBox(x=10, y=10, width=150, height=50)

    analysis = _analyze_patch(image, bbox)

    assert analysis.coverage_ratio < 0.95
    assert np.count_nonzero(analysis.mask) < analysis.mask.size


def test_mask_png_is_saved_and_reloadable(tmp_path: Path):
    image = _draw_text_image(text="SAVE")
    bbox = BoundingBox(x=15, y=15, width=130, height=45)
    block = _fused_block(bbox=bbox, block_id="block/save")
    visual = VisualStyle(
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=45,
    )

    result = analyze_text_mask_for_block(
        Image.fromarray(image),
        block,
        visual=visual,
        render_hints=RenderHints(available_width_px=bbox.width, available_height_px=bbox.height),
        background=BackgroundInfo(background_type="solid", classification_confidence=0.9),
        source_image_id="mask-save-case",
        output_dir=tmp_path,
    )

    assert result.render_geometry is not None
    assert result.render_geometry.mask_path is not None
    assert result.render_geometry.mask_bbox == bbox
    assert result.render_geometry.mask_confidence >= MASK_SAVE_MIN_CONFIDENCE

    mask_file = PROJECT_ROOT / result.render_geometry.mask_path
    assert mask_file.exists()
    loaded = load_text_mask_png(mask_file)
    assert loaded.shape == (bbox.height, bbox.width)
    assert set(np.unique(loaded)).issubset({0, 255})


def test_low_confidence_mask_is_not_persisted(tmp_path: Path):
    image = np.full((60, 120, 3), 200, dtype=np.uint8)
    bbox = BoundingBox(x=10, y=10, width=100, height=40)
    block = _fused_block(bbox=bbox)

    result = analyze_text_mask_for_block(
        Image.fromarray(image),
        block,
        visual=VisualStyle(background_color="#C8C8C8", text_color="#000000"),
        render_hints=RenderHints(available_width_px=100, available_height_px=40),
        background=BackgroundInfo(background_type="solid", classification_confidence=0.9),
        source_image_id="low-confidence",
        output_dir=tmp_path,
    )

    assert result.render_geometry is None or result.render_geometry.mask_path is None


def test_save_text_mask_png_uses_safe_filename(tmp_path: Path):
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[3:7, 5:15] = MASK_FOREGROUND_VALUE

    path = save_text_mask_png(
        mask,
        output_dir=tmp_path,
        source_image_id="case-1",
        block_id="../evil/block",
    )

    assert ".." not in path
    assert path.endswith("_mask.png")
    assert (PROJECT_ROOT / path).exists()


def test_expand_text_mask_is_monotonic_and_preserves_raw():
    raw = np.zeros((20, 40), dtype=np.uint8)
    raw[8:14, 10:30] = MASK_FOREGROUND_VALUE
    raw_copy = raw.copy()

    expanded = expand_text_mask(raw, estimated_text_height=20)

    assert np.count_nonzero(expanded) >= np.count_nonzero(raw_copy)
    assert np.array_equal(raw, raw_copy)


def test_expand_text_mask_empty_is_safe():
    empty = np.zeros((10, 10), dtype=np.uint8)
    expanded = expand_text_mask(empty, estimated_text_height=10)
    assert expanded.shape == empty.shape
    assert not expanded.any()


def test_axis_aligned_rectangle_has_no_perspective():
    polygon = [(10, 20), (130, 20), (130, 60), (10, 60)]
    normalized, detected = analyze_perspective_quad(polygon)

    assert detected is False
    assert normalized == normalize_quad_order(polygon)


def test_rotated_rectangle_has_no_perspective():
    polygon = [(40, 10), (120, 40), (100, 80), (20, 50)]
    _, detected = analyze_perspective_quad(polygon)
    assert detected is False


def test_trapezoid_can_be_detected_as_perspective():
    polygon = [(20, 20), (180, 30), (160, 80), (40, 70)]
    normalized, detected = analyze_perspective_quad(polygon)

    assert detected is True
    assert normalized == normalize_quad_order(polygon)


def test_degenerate_polygon_falls_back_safely():
    polygon = [(10, 10), (10, 10), (10, 10), (10, 10)]
    normalized, detected = analyze_perspective_quad(polygon)

    assert detected is False
    assert normalized is not None


def test_normalize_quad_order_is_stable():
    polygon = [(10, 60), (130, 20), (130, 60), (10, 20)]
    ordered = normalize_quad_order(polygon)

    assert ordered[0][1] <= ordered[1][1]
    assert ordered[0][0] <= ordered[3][0]
    assert ordered[1][0] >= ordered[0][0]
    assert ordered[2][1] >= ordered[0][1]


def test_baseline_angle_matches_visual_rotation_semantics(tmp_path: Path):
    polygon = [(0, 0), (100, 100), (80, 120), (-20, 20)]
    block = _fused_block(polygon=polygon)
    visual = VisualStyle(
        rotation_deg=45.0,
        text_color="#000000",
        background_color="#FFFFFF",
        estimated_text_height_px=40,
    )
    image = _draw_text_image(text="T", size=(160, 160))

    result = analyze_text_mask_for_block(
        Image.fromarray(image),
        block,
        visual=visual,
        render_hints=RenderHints(available_width_px=120, available_height_px=40),
        background=None,
        source_image_id="rotation-case",
        output_dir=tmp_path,
    )

    assert result.render_geometry is not None
    assert result.render_geometry.baseline_angle_deg == pytest.approx(45.0)


def test_mask_analysis_has_no_runtime_warnings():
    image = _draw_text_image(text="ABC")
    bbox = BoundingBox(x=15, y=15, width=130, height=45)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        _analyze_patch(image, bbox)

    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
