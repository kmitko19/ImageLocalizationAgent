"""Unit tests for Renderer R0.3 background restoration."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

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
from imageloc.renderer.background_restorer import (
    RESTORE_MIN_MASK_CONFIDENCE,
    restore_backgrounds,
    restore_text_background,
    resolve_clean_background_path,
    sample_solid_fill_color,
    save_clean_background,
    select_restore_method,
)
from imageloc.utils.text_mask_analyzer import MASK_FOREGROUND_VALUE, expand_text_mask


def _block(
    *,
    block_id: str = "block_001",
    bbox: BoundingBox,
    mask: np.ndarray,
    background: BackgroundInfo | None,
    mask_confidence: float = 0.8,
    text_height: int | None = None,
) -> TextBlock:
    return TextBlock(
        id=block_id,
        original_text="SALE",
        translated_text="SALE",
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
        visual=VisualStyle(
            background_color="#FFFFFF",
            text_color="#000000",
            estimated_text_height_px=text_height or bbox.height,
        ),
        render_hints=RenderHints(available_width_px=bbox.width, available_height_px=bbox.height),
        review=ReviewInfo(),
        background=background,
        render_geometry=RenderGeometry(
            mask_bbox=bbox,
            mask_confidence=mask_confidence,
        ),
    )


def _solid_image_with_bar(
    *,
    size=(160, 80),
    bg=(240, 230, 220),
    bar=(20, 10, 5),
    bar_box=(40, 20, 80, 30),
) -> tuple[np.ndarray, np.ndarray, BoundingBox]:
    image = np.full((size[1], size[0], 3), bg, dtype=np.uint8)
    x, y, w, h = bar_box
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, :] = MASK_FOREGROUND_VALUE
    image[y : y + h, x : x + w] = bar
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    return image, mask, bbox


def _gradient_image_with_bar(
    *,
    size=(180, 80),
    bar_box=(50, 20, 70, 30),
) -> tuple[np.ndarray, np.ndarray, BoundingBox]:
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for row in range(size[1]):
        value = int(255 * row / max(1, size[1] - 1))
        image[row, :, :] = (value, value // 2, 255 - value)
    x, y, w, h = bar_box
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[5:25, 10:60] = MASK_FOREGROUND_VALUE
    image[y : y + h, x : x + w][mask > 0] = (10, 10, 10)
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    return image, mask, bbox


def _outside_change_ratio(before: np.ndarray, after: np.ndarray, mask: np.ndarray, bbox: BoundingBox) -> float:
    expanded = expand_text_mask(mask, estimated_text_height=bbox.height)
    canvas = np.zeros(before.shape[:2], dtype=bool)
    canvas[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = expanded > 0
    outside = ~canvas
    if not outside.any():
        return 0.0
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16))
    changed = (diff.sum(axis=2) > 0) & outside
    return float(np.count_nonzero(changed)) / float(np.count_nonzero(outside))


def test_select_restore_method_respects_background_type():
    assert select_restore_method(BackgroundInfo(background_type="solid"), mask_confidence=0.8) == "fill"
    assert select_restore_method(BackgroundInfo(background_type="gradient"), mask_confidence=0.8) == "gradient"
    assert select_restore_method(BackgroundInfo(background_type="photo"), mask_confidence=0.8) == "inpaint"
    assert select_restore_method(BackgroundInfo(background_type="complex"), mask_confidence=0.8) == "advanced_inpaint"
    assert select_restore_method(None, mask_confidence=0.2) == "skip"


def test_solid_restoration_removes_text_and_preserves_outside():
    image_array, mask, bbox = _solid_image_with_bar()
    block = _block(
        bbox=bbox,
        mask=mask,
        background=BackgroundInfo(background_type="solid", preferred_restore_method="fill"),
    )
    before = Image.fromarray(image_array)

    result = restore_text_background(before, block, mask=mask)
    after = np.array(result.image)

    assert result.success is True
    assert result.method == "fill"
    assert after.shape == image_array.shape
    restored_region = after[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    assert np.mean(np.abs(restored_region.astype(np.int16) - 240)) < 20
    assert _outside_change_ratio(image_array, after, mask, bbox) < 0.01


def test_gradient_restoration_beats_constant_fill():
    image_array, mask, bbox = _gradient_image_with_bar()
    block = _block(
        bbox=bbox,
        mask=mask,
        background=BackgroundInfo(background_type="gradient", preferred_restore_method="gradient"),
    )
    before = Image.fromarray(image_array)
    expanded = expand_text_mask(mask, estimated_text_height=bbox.height)

    restored = restore_text_background(before, block, mask=mask)
    assert restored.success is True

    constant = image_array.copy()
    fill_color = np.median(image_array, axis=(0, 1))
    patch = constant[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    patch[expanded > 0] = fill_color

    result_array = np.array(restored.image)
    patch_result = result_array[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    true_patch = image_array[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]

    constant_error = float(np.mean(np.abs(patch[expanded > 0].astype(np.int16) - true_patch[expanded > 0].astype(np.int16))))
    restored_error = float(
        np.mean(np.abs(patch_result[expanded > 0].astype(np.int16) - true_patch[expanded > 0].astype(np.int16)))
    )
    assert restored_error < constant_error


def test_texture_background_uses_inpaint():
    rng = np.random.default_rng(4)
    image = rng.integers(120, 200, size=(90, 160, 3), dtype=np.uint8)
    x, y, w, h = 50, 20, 60, 30
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[8:22, 10:50] = MASK_FOREGROUND_VALUE
    image[y : y + h, x : x + w][mask > 0] = (10, 10, 10)
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    block = _block(
        bbox=bbox,
        mask=mask,
        background=BackgroundInfo(background_type="texture", preferred_restore_method="inpaint"),
    )

    result = restore_text_background(Image.fromarray(image), block, mask=mask)

    assert result.success is True
    assert result.method == "inpaint"
    restored = np.array(result.image)
    assert restored.shape == image.shape
    assert not np.array_equal(restored[y : y + h, x : x + w][mask > 0], image[y : y + h, x : x + w][mask > 0])


def test_low_confidence_restoration_is_skipped():
    image_array, mask, bbox = _solid_image_with_bar()
    block = _block(
        bbox=bbox,
        mask=mask,
        background=BackgroundInfo(background_type="solid"),
        mask_confidence=RESTORE_MIN_MASK_CONFIDENCE - 0.1,
    )

    result = restore_text_background(Image.fromarray(image_array), block, mask=mask)

    assert result.success is False
    assert result.method == "skip"
    assert np.array_equal(np.array(result.image), image_array)


def test_empty_mask_is_skipped():
    image_array, mask, bbox = _solid_image_with_bar()
    empty = np.zeros_like(mask)
    block = _block(bbox=bbox, mask=empty, background=BackgroundInfo(background_type="solid"))

    result = restore_text_background(Image.fromarray(image_array), block, mask=empty)

    assert result.success is False
    assert "empty mask" in result.warnings[0]


def test_rgba_alpha_is_preserved():
    image_array, mask, bbox = _solid_image_with_bar()
    alpha = np.full((image_array.shape[0], image_array.shape[1]), 180, dtype=np.uint8)
    rgba = np.dstack([image_array, alpha])
    block = _block(bbox=bbox, mask=mask, background=BackgroundInfo(background_type="solid"))

    result = restore_text_background(Image.fromarray(rgba, mode="RGBA"), block, mask=mask)

    assert result.success is True
    restored = np.array(result.image)
    assert restored.shape[2] == 4
    assert np.array_equal(restored[:, :, 3], alpha)


def test_restore_backgrounds_processes_multiple_blocks():
    image = np.full((100, 200, 3), 250, dtype=np.uint8)
    mask_a = np.zeros((20, 40), dtype=np.uint8)
    mask_a[5:15, 10:30] = MASK_FOREGROUND_VALUE
    bbox_a = BoundingBox(x=20, y=20, width=40, height=20)
    image[bbox_a.y : bbox_a.y + bbox_a.height, bbox_a.x : bbox_a.x + bbox_a.width][mask_a > 0] = (0, 0, 0)

    mask_b = np.zeros((20, 40), dtype=np.uint8)
    mask_b[5:15, 10:30] = MASK_FOREGROUND_VALUE
    bbox_b = BoundingBox(x=120, y=60, width=40, height=20)
    image[bbox_b.y : bbox_b.y + bbox_b.height, bbox_b.x : bbox_b.x + bbox_b.width][mask_b > 0] = (0, 0, 0)

    blocks = [
        _block(block_id="a", bbox=bbox_a, mask=mask_a, background=BackgroundInfo(background_type="solid")),
        _block(block_id="b", bbox=bbox_b, mask=mask_b, background=BackgroundInfo(background_type="solid")),
    ]

    restored_image, results = restore_backgrounds(
        Image.fromarray(image),
        blocks,
        masks={"a": mask_a, "b": mask_b},
    )

    assert len(results) == 2
    assert all(item.success for item in results)
    restored = np.array(restored_image)
    assert np.mean(restored[bbox_a.y : bbox_a.y + bbox_a.height, bbox_a.x : bbox_a.x + bbox_a.width]) > 200
    assert np.mean(restored[bbox_b.y : bbox_b.y + bbox_b.height, bbox_b.x : bbox_b.x + bbox_b.width]) > 200


def test_overlapping_blocks_do_not_raise():
    image = np.full((80, 120, 3), 240, dtype=np.uint8)
    mask = np.zeros((30, 50), dtype=np.uint8)
    mask[10:20, 15:35] = MASK_FOREGROUND_VALUE
    bbox_a = BoundingBox(x=20, y=20, width=50, height=30)
    bbox_b = BoundingBox(x=40, y=25, width=50, height=30)
    image[bbox_a.y : bbox_a.y + bbox_a.height, bbox_a.x : bbox_a.x + bbox_a.width][mask > 0] = (0, 0, 0)
    image[bbox_b.y : bbox_b.y + bbox_b.height, bbox_b.x : bbox_b.x + bbox_b.width][mask > 0] = (0, 0, 0)

    blocks = [
        _block(block_id="a", bbox=bbox_a, mask=mask, background=BackgroundInfo(background_type="solid")),
        _block(block_id="b", bbox=bbox_b, mask=mask, background=BackgroundInfo(background_type="solid")),
    ]

    restored_image, results = restore_backgrounds(
        Image.fromarray(image),
        blocks,
        masks={"a": mask, "b": mask},
    )

    assert len(results) == 2
    assert restored_image.size == Image.fromarray(image).size


@pytest.mark.parametrize(
    ("x", "y"),
    [(0, 30), (140, 30), (70, 0), (70, 70)],
)
def test_restoration_at_image_edges(x: int, y: int):
    image = np.full((100, 180, 3), 230, dtype=np.uint8)
    w, h = 40, 24
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[6:18, 8:32] = MASK_FOREGROUND_VALUE
    image[y : y + h, x : x + w][mask > 0] = (0, 0, 0)
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    block = _block(bbox=bbox, mask=mask, background=BackgroundInfo(background_type="solid"))

    result = restore_text_background(Image.fromarray(image), block, mask=mask)

    assert result.success is True
    assert np.array(result.image).shape == image.shape


def test_save_clean_background_uses_project_relative_path(tmp_path: Path):
    output_dir = PROJECT_ROOT / "data" / "output"
    image = Image.new("RGB", (40, 20), color=(240, 240, 240))

    stored_path = save_clean_background(
        image,
        output_dir=output_dir,
        source_image_id="restore-demo-case",
    )

    try:
        assert stored_path.startswith("data/output/")
        assert stored_path.endswith("render_assets/clean_background.png")
        resolved = resolve_clean_background_path(stored_path)
        assert resolved.exists()
        loaded = Image.open(resolved)
        assert loaded.size == image.size
    finally:
        shutil.rmtree(output_dir / "restore-demo-case", ignore_errors=True)


def test_mask_dimension_mismatch_is_skipped():
    image_array, mask, bbox = _solid_image_with_bar()
    wrong_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    block = _block(bbox=bbox, mask=mask, background=BackgroundInfo(background_type="solid"))

    result = restore_text_background(Image.fromarray(image_array), block, mask=wrong_mask)

    assert result.success is False


def test_solid_blue_background_does_not_leave_white_residual():
    blue = (30, 80, 160)
    cream = (245, 240, 232)
    canvas = np.full((100, 180, 3), cream, dtype=np.uint8)
    bbox = BoundingBox(x=20, y=20, width=90, height=50)
    canvas[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = blue

    mask = np.zeros((bbox.height, bbox.width), dtype=np.uint8)
    mask[10:35, 18:72] = MASK_FOREGROUND_VALUE
    canvas[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width][mask > 0] = (240, 240, 240)
    mask[8:12, 24:30] = MASK_FOREGROUND_VALUE
    canvas[bbox.y + 8 : bbox.y + 12, bbox.x + 24 : bbox.x + 30] = (220, 220, 220)

    block = _block(
        bbox=bbox,
        mask=mask,
        background=BackgroundInfo(
            background_type="solid",
            dominant_color="#1E509F",
            preferred_restore_method="fill",
        ),
        text_height=40,
    )

    result = restore_text_background(Image.fromarray(canvas), block, mask=mask)
    expanded = expand_text_mask(mask, estimated_text_height=40)
    restored_patch = np.array(result.image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]

    assert result.success is True
    assert result.method == "fill"

    foreground_before = mask > 0
    restored_inside = restored_patch[expanded > 0]
    white_like = np.all(restored_inside > 230, axis=1)
    assert int(np.count_nonzero(white_like)) == 0

    blue_distance = np.linalg.norm(restored_inside.astype(np.float32) - np.array(blue, dtype=np.float32), axis=1)
    assert float(np.mean(blue_distance)) < 25.0


def test_solid_fill_color_ignores_outside_canvas_padding():
    blue = (30, 80, 160)
    cream = (245, 240, 232)
    patch = np.full((40, 60, 3), cream, dtype=np.uint8)
    patch[5:35, 10:50] = blue
    raw_mask = np.zeros((40, 60), dtype=np.uint8)
    raw_mask[13:27, 20:40] = MASK_FOREGROUND_VALUE
    patch[raw_mask > 0] = (240, 240, 240)
    expanded = expand_text_mask(raw_mask, estimated_text_height=30)

    fill_color = sample_solid_fill_color(patch, expanded, raw_mask, hint_rgb=blue)

    assert float(np.linalg.norm(fill_color - np.array(blue, dtype=np.float32))) < 20.0
    assert float(np.linalg.norm(fill_color - np.array(cream, dtype=np.float32))) > 40.0


def test_foreground_coverage_by_expanded_mask():
    image_array, mask, bbox = _solid_image_with_bar(bar=(20, 10, 5), bar_box=(40, 20, 80, 30))
    foreground = np.all(image_array[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] == (20, 10, 5), axis=2)
    expanded = expand_text_mask(mask, estimated_text_height=bbox.height)
    coverage = np.count_nonzero(expanded[foreground]) / max(1, np.count_nonzero(foreground))
    assert coverage >= 0.98
