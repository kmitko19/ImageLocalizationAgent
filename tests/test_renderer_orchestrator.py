"""Unit tests for Renderer R0.7 orchestrator."""

from __future__ import annotations

import json
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
from imageloc.renderer.background_restorer import resolve_clean_background_path
from imageloc.renderer.orchestrator import (
    BLOCK_STATUS_FAILED,
    BLOCK_STATUS_RENDERED,
    BLOCK_STATUS_SKIPPED,
    render_image_localization,
    resolve_renderer_result_path,
)
from imageloc.renderer.text_renderer import resolve_localized_image_path
from imageloc.utils.text_mask_analyzer import MASK_FOREGROUND_VALUE

pytestmark = pytest.mark.skipif(
    not (Path(__import__("os").environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf").is_file(),
    reason="Windows Unicode fonts required for renderer orchestrator tests",
)


def _block(
    *,
    block_id: str,
    bbox: BoundingBox,
    translated_text: str | None = "SALE",
    visual: VisualStyle | None = None,
    hints: RenderHints | None = None,
    background: BackgroundInfo | None = None,
    render_geometry: RenderGeometry | None = None,
) -> TextBlock:
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
        original_text=block_id,
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
        ocr=OCRData(raw_text=block_id, confidence=0.9),
        visual=style,
        render_hints=hints or RenderHints(available_width_px=bbox.width, available_height_px=bbox.height),
        review=ReviewInfo(),
        background=background
        or BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill"),
        render_geometry=render_geometry or RenderGeometry(mask_bbox=bbox, mask_confidence=0.8),
    )


def _blank_image(size=(320, 160), color=(240, 230, 220)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _mask_for_block(image: Image.Image, block: TextBlock, *, bright_text: bool = False) -> np.ndarray:
    bbox = block.bbox
    patch = np.array(image)[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    mask = np.zeros((bbox.height, bbox.width), dtype=np.uint8)
    if bright_text:
        mask[np.mean(patch, axis=2) > 200] = MASK_FOREGROUND_VALUE
    else:
        mask[np.mean(patch, axis=2) < 200] = MASK_FOREGROUND_VALUE
    return mask


def _restore_ready_block(
    image: Image.Image,
    *,
    block_id: str,
    rect: tuple[int, int, int, int],
    translated_text: str,
    fill: tuple[int, int, int],
    text_fill: tuple[int, int, int],
) -> tuple[TextBlock, np.ndarray]:
    draw = ImageDraw.Draw(image)
    draw.rectangle(rect, fill=fill)
    draw.text((rect[0] + 8, rect[1] + 8), block_id, fill=text_fill)
    x1, y1, x2, y2 = rect
    bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
    block = _block(
        block_id=block_id,
        bbox=bbox,
        translated_text=translated_text,
        background=BackgroundInfo(
            background_type="solid",
            dominant_color="#{:02x}{:02x}{:02x}".format(*fill),
            preferred_restore_method="fill",
        ),
        visual=VisualStyle(
            text_color="#FFFFFF" if text_fill[0] > 200 else "#141414",
            background_color="#{:02x}{:02x}{:02x}".format(*fill),
            estimated_text_height_px=bbox.height - 8,
            font_size_estimate=max(10, bbox.height - 10),
            alignment="center",
            vertical_alignment="center",
        ),
    )
    return block, _mask_for_block(image, block, bright_text=text_fill[0] > 200)


def test_successful_full_render():
    image = _blank_image()
    block, mask = _restore_ready_block(
        image,
        block_id="block_a",
        rect=(20, 20, 160, 60),
        translated_text="HELLO",
        fill=(30, 80, 160),
        text_fill=(240, 240, 240),
    )
    result = render_image_localization(image, [block], masks={block.id: mask})
    assert result.success is True
    assert result.rendered_block_count == 1
    assert result.block_results[0].status == BLOCK_STATUS_RENDERED


def test_multiple_blocks():
    image = _blank_image(size=(420, 160))
    block_a, mask_a = _restore_ready_block(
        image,
        block_id="block_a",
        rect=(20, 20, 160, 60),
        translated_text="ONE",
        fill=(30, 80, 160),
        text_fill=(240, 240, 240),
    )
    block_b, mask_b = _restore_ready_block(
        image,
        block_id="block_b",
        rect=(180, 20, 320, 60),
        translated_text="TWO",
        fill=(180, 60, 60),
        text_fill=(255, 255, 255),
    )
    result = render_image_localization(
        image,
        [block_a, block_b],
        masks={block_a.id: mask_a, block_b.id: mask_b},
    )
    assert result.success is True
    assert result.rendered_block_count == 2
    assert [item.block_id for item in result.block_results] == ["block_a", "block_b"]


def test_block_order_is_input_order():
    image = _blank_image(size=(420, 160))
    blocks = []
    masks = {}
    for block_id, rect, text in [
        ("first", (20, 20, 120, 55), "FIRST"),
        ("second", (140, 20, 240, 55), "SECOND"),
        ("third", (260, 20, 360, 55), "THIRD"),
    ]:
        block, mask = _restore_ready_block(
            image,
            block_id=block_id,
            rect=rect,
            translated_text=text,
            fill=(255, 255, 255),
            text_fill=(20, 20, 20),
        )
        blocks.append(block)
        masks[block.id] = mask
    result = render_image_localization(image, blocks, masks=masks)
    assert [item.block_id for item in result.block_results] == ["first", "second", "third"]


def test_empty_translated_text_is_skipped():
    bbox = BoundingBox(x=20, y=20, width=100, height=40)
    block = _block(block_id="empty_block", bbox=bbox, translated_text="   ")
    result = render_image_localization(_blank_image(), [block])
    assert result.success is True
    assert result.skipped_block_count == 1
    assert result.rendered_block_count == 0
    assert result.block_results[0].status == BLOCK_STATUS_SKIPPED
    assert any("empty translated_text" in warning for warning in result.block_results[0].warnings)


def test_one_failed_block_does_not_stop_others():
    image = _blank_image(size=(420, 160))
    ok_block, ok_mask = _restore_ready_block(
        image,
        block_id="ok_block",
        rect=(20, 20, 180, 60),
        translated_text="OK",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    fail_bbox = BoundingBox(x=220, y=20, width=70, height=24)
    fail_block = _block(
        block_id="fail_block",
        bbox=fail_bbox,
        translated_text="Too long text for tiny block",
        hints=RenderHints(
            available_width_px=70,
            available_height_px=24,
            max_line_count=1,
            font_scale_min=0.9,
            font_scale_max=1.0,
        ),
        visual=VisualStyle(
            text_color="#000000",
            font_size_estimate=22,
            estimated_text_height_px=20,
        ),
    )
    result = render_image_localization(
        image,
        [fail_block, ok_block],
        masks={ok_block.id: ok_mask},
    )
    assert result.success is True
    assert result.failed_block_ids == ["fail_block"]
    assert result.rendered_block_count == 1
    assert result.block_results[0].status == BLOCK_STATUS_FAILED
    assert result.block_results[1].status == BLOCK_STATUS_RENDERED


def test_warnings_aggregated():
    bbox = BoundingBox(x=20, y=20, width=100, height=40)
    block = _block(block_id="warn_block", bbox=bbox, translated_text="")
    result = render_image_localization(_blank_image(), [block])
    assert any("render[warn_block]" in warning for warning in result.warnings)


def test_failed_block_ids():
    bbox = BoundingBox(x=20, y=20, width=60, height=20)
    block = _block(
        block_id="tiny_fail",
        bbox=bbox,
        translated_text="Will not fit",
        hints=RenderHints(available_width_px=60, available_height_px=20, max_line_count=1, font_scale_min=0.9),
        visual=VisualStyle(text_color="#000000", font_size_estimate=20, estimated_text_height_px=18),
    )
    result = render_image_localization(_blank_image(size=(160, 100)), [block])
    assert result.failed_block_ids == ["tiny_fail"]


def test_rendered_and_skipped_counts():
    image = _blank_image(size=(420, 120))
    ok_block, ok_mask = _restore_ready_block(
        image,
        block_id="rendered",
        rect=(20, 20, 160, 60),
        translated_text="OK",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    skip_block = _block(block_id="skipped", bbox=BoundingBox(x=180, y=20, width=100, height=40), translated_text=None)
    result = render_image_localization(image, [ok_block, skip_block], masks={ok_block.id: ok_mask})
    assert result.rendered_block_count == 1
    assert result.skipped_block_count == 1


def test_rgb_preserved():
    image = _blank_image()
    block, mask = _restore_ready_block(
        image,
        block_id="rgb",
        rect=(20, 20, 160, 60),
        translated_text="RGB",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    result = render_image_localization(image, [block], masks={block.id: mask})
    assert result.image.mode == "RGB"


def test_rgba_alpha_preserved():
    image = Image.new("RGBA", (220, 120), (240, 230, 220, 128))
    block, mask = _restore_ready_block(
        image,
        block_id="rgba",
        rect=(20, 20, 160, 60),
        translated_text="RGBA",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    before = np.array(image.getchannel("A"))
    result = render_image_localization(image, [block], masks={block.id: mask})
    after = np.array(result.image.getchannel("A"))
    outside = np.ones(before.shape, dtype=bool)
    bbox = block.bbox
    outside[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width] = False
    assert np.array_equal(before[outside], after[outside])


def test_image_size_unchanged():
    image = _blank_image(size=(360, 180))
    block, mask = _restore_ready_block(
        image,
        block_id="size",
        rect=(20, 20, 160, 60),
        translated_text="SIZE",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    result = render_image_localization(image, [block], masks={block.id: mask})
    assert result.image.size == image.size
    assert result.clean_background.size == image.size


def test_deterministic_repeated_render():
    image = _blank_image(size=(320, 140))
    block, mask = _restore_ready_block(
        image,
        block_id="deterministic",
        rect=(20, 20, 180, 70),
        translated_text="SAME",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    first = np.array(render_image_localization(image.copy(), [block], masks={block.id: mask}).image)
    second = np.array(render_image_localization(image.copy(), [block], masks={block.id: mask}).image)
    assert np.array_equal(first, second)


def test_save_assets_false_creates_no_files(tmp_path: Path):
    image = _blank_image()
    block, mask = _restore_ready_block(
        image,
        block_id="no_save",
        rect=(20, 20, 160, 60),
        translated_text="NO SAVE",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    result = render_image_localization(
        image,
        [block],
        masks={block.id: mask},
        save_assets=False,
        output_dir=tmp_path,
    )
    assert result.success is True
    assert result.output_paths == {}
    assert not any(tmp_path.rglob("*"))


def test_save_assets_true_creates_expected_assets(tmp_path: Path):
    image = _blank_image()
    block, mask = _restore_ready_block(
        image,
        block_id="save_me",
        rect=(20, 20, 160, 60),
        translated_text="SAVE",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    source_image_id = "test_renderer_orchestrator_save"
    result = render_image_localization(
        image,
        [block],
        masks={block.id: mask},
        source_image_id=source_image_id,
        save_assets=True,
        output_dir=tmp_path,
    )
    try:
        assert result.success is True
        assert set(result.output_paths) == {"clean_background", "localized_image", "renderer_result"}
        clean_path = resolve_clean_background_path(result.output_paths["clean_background"])
        localized_path = resolve_localized_image_path(result.output_paths["localized_image"])
        result_path = resolve_renderer_result_path(result.output_paths["renderer_result"])
        assert clean_path.is_file()
        assert localized_path.is_file()
        assert result_path.is_file()
    finally:
        shutil.rmtree(tmp_path / source_image_id, ignore_errors=True)


def test_renderer_result_json_valid(tmp_path: Path):
    image = _blank_image()
    block = _block(block_id="json_block", bbox=BoundingBox(x=20, y=20, width=100, height=40), translated_text="   ")
    source_image_id = "test_renderer_orchestrator_json"
    result = render_image_localization(
        image,
        [block],
        source_image_id=source_image_id,
        save_assets=True,
        output_dir=tmp_path,
    )
    try:
        payload = json.loads(resolve_renderer_result_path(result.output_paths["renderer_result"]).read_text(encoding="utf-8"))
        assert payload["success"] is True
        assert payload["source_image_id"] == source_image_id
        assert payload["skipped_block_count"] == 1
        assert payload["block_results"][0]["status"] == BLOCK_STATUS_SKIPPED
        assert "image" not in payload
    finally:
        shutil.rmtree(tmp_path / source_image_id, ignore_errors=True)


def test_output_paths_are_project_relative(tmp_path: Path):
    image = _blank_image()
    block, mask = _restore_ready_block(
        image,
        block_id="paths",
        rect=(20, 20, 160, 60),
        translated_text="PATHS",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    source_image_id = "test_renderer_orchestrator_paths"
    result = render_image_localization(
        image,
        [block],
        masks={block.id: mask},
        source_image_id=source_image_id,
        save_assets=True,
        output_dir=tmp_path,
    )
    try:
        assert result.output_paths["clean_background"].endswith("render_assets/clean_background.png")
        assert result.output_paths["localized_image"].endswith("render_assets/localized_image.png")
        assert result.output_paths["renderer_result"].endswith("render_assets/renderer_result.json")
    finally:
        shutil.rmtree(tmp_path / source_image_id, ignore_errors=True)


def test_clean_background_readable(tmp_path: Path):
    image = _blank_image()
    block, mask = _restore_ready_block(
        image,
        block_id="clean",
        rect=(20, 20, 160, 60),
        translated_text="CLEAN",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    source_image_id = "test_renderer_orchestrator_clean"
    result = render_image_localization(
        image,
        [block],
        masks={block.id: mask},
        source_image_id=source_image_id,
        save_assets=True,
        output_dir=tmp_path,
    )
    try:
        loaded = Image.open(resolve_clean_background_path(result.output_paths["clean_background"]))
        assert loaded.size == image.size
    finally:
        shutil.rmtree(tmp_path / source_image_id, ignore_errors=True)


def test_localized_image_readable(tmp_path: Path):
    image = _blank_image()
    block, mask = _restore_ready_block(
        image,
        block_id="localized",
        rect=(20, 20, 160, 60),
        translated_text="LOCALIZED",
        fill=(255, 255, 255),
        text_fill=(20, 20, 20),
    )
    source_image_id = "test_renderer_orchestrator_localized"
    result = render_image_localization(
        image,
        [block],
        masks={block.id: mask},
        source_image_id=source_image_id,
        save_assets=True,
        output_dir=tmp_path,
    )
    try:
        loaded = Image.open(resolve_localized_image_path(result.output_paths["localized_image"]))
        assert loaded.size == image.size
    finally:
        shutil.rmtree(tmp_path / source_image_id, ignore_errors=True)


def test_invalid_source_image_id_with_save_assets():
    image = _blank_image()
    block = _block(block_id="invalid", bbox=BoundingBox(x=10, y=10, width=80, height=30))
    result = render_image_localization(
        image,
        [block],
        source_image_id="../bad/id",
        save_assets=True,
        output_dir=PROJECT_ROOT / "data/output",
    )
    assert result.success is False
    assert result.fatal_error is not None


def test_missing_source_image_id_with_save_assets():
    image = _blank_image()
    block = _block(block_id="missing", bbox=BoundingBox(x=10, y=10, width=80, height=30))
    result = render_image_localization(image, [block], save_assets=True)
    assert result.success is False
    assert "source_image_id is required" in (result.fatal_error or "")


def test_block_level_failure_has_controlled_result():
    bbox = BoundingBox(x=20, y=20, width=50, height=18)
    block = _block(
        block_id="controlled_fail",
        bbox=bbox,
        translated_text="Too long",
        hints=RenderHints(available_width_px=50, available_height_px=18, max_line_count=1, font_scale_min=0.95),
        visual=VisualStyle(text_color="#000000", font_size_estimate=18, estimated_text_height_px=16),
    )
    result = render_image_localization(_blank_image(size=(160, 100)), [block])
    assert result.success is True
    assert result.block_results[0].status == BLOCK_STATUS_FAILED
    assert result.block_results[0].failure_reason == "text_does_not_fit_at_minimum_font_size"
