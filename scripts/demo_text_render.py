"""Development demo for Renderer R0.4 text rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from imageloc.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402
from imageloc.models.raw_blocks import BoundingBox  # noqa: E402
from imageloc.models.text_block import (  # noqa: E402
    BackgroundInfo,
    OCRData,
    RenderGeometry,
    RenderHints,
    ReviewInfo,
    TextBlock,
    VisualStyle,
)
from imageloc.renderer.text_renderer import render_localized_image, save_localized_image  # noqa: E402
from imageloc.utils.text_mask_analyzer import MASK_FOREGROUND_VALUE  # noqa: E402


def _build_demo_image() -> tuple[Image.Image, list[TextBlock], dict[str, np.ndarray]]:
    canvas = Image.new("RGB", (720, 280), color=(245, 240, 232))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((40, 40, 220, 95), fill=(30, 80, 160))
    draw.text((95, 52), "SALE", fill=(240, 240, 240))

    draw.rectangle((260, 35, 470, 130), fill=(255, 255, 255))
    draw.text((270, 45), "SPECIAL", fill=(20, 20, 20))

    draw.rectangle((500, 45, 660, 110), fill=(180, 60, 60))
    draw.text((535, 58), "HOT", fill=(255, 240, 220))

    image = np.array(canvas)
    blocks: list[TextBlock] = []
    masks: dict[str, np.ndarray] = {}

    centered_bbox = BoundingBox(x=35, y=35, width=190, height=65)
    centered_mask = np.zeros((centered_bbox.height, centered_bbox.width), dtype=np.uint8)
    centered_patch = image[
        centered_bbox.y : centered_bbox.y + centered_bbox.height,
        centered_bbox.x : centered_bbox.x + centered_bbox.width,
    ]
    centered_mask[np.mean(centered_patch, axis=2) > 200] = MASK_FOREGROUND_VALUE
    blocks.append(
        TextBlock(
            id="demo_centered",
            original_text="SALE",
            translated_text="РАСПРОДАЖА",
            source_language="EN",
            confidence=0.9,
            bbox=centered_bbox,
            polygon=[
                (centered_bbox.x, centered_bbox.y),
                (centered_bbox.x + centered_bbox.width, centered_bbox.y),
                (centered_bbox.x + centered_bbox.width, centered_bbox.y + centered_bbox.height),
                (centered_bbox.x, centered_bbox.y + centered_bbox.height),
            ],
            ocr=OCRData(raw_text="SALE", confidence=0.9),
            visual=VisualStyle(
                background_color="#1E509F",
                text_color="#FFFFFF",
                estimated_text_height_px=36,
                font_size_estimate=30,
                alignment="center",
                vertical_alignment="center",
            ),
            render_hints=RenderHints(
                available_width_px=centered_bbox.width,
                available_height_px=centered_bbox.height,
                max_line_count=1,
            ),
            review=ReviewInfo(),
            background=BackgroundInfo(
                background_type="solid",
                dominant_color="#1E509F",
                preferred_restore_method="fill",
            ),
            render_geometry=RenderGeometry(mask_bbox=centered_bbox, mask_confidence=0.82),
        )
    )
    masks["demo_centered"] = centered_mask

    wrapped_bbox = BoundingBox(x=255, y=30, width=220, height=105)
    wrapped_mask = np.zeros((wrapped_bbox.height, wrapped_bbox.width), dtype=np.uint8)
    wrapped_patch = image[
        wrapped_bbox.y : wrapped_bbox.y + wrapped_bbox.height,
        wrapped_bbox.x : wrapped_bbox.x + wrapped_bbox.width,
    ]
    wrapped_mask[np.mean(wrapped_patch, axis=2) < 200] = MASK_FOREGROUND_VALUE
    blocks.append(
        TextBlock(
            id="demo_wrapped",
            original_text="SPECIAL",
            translated_text="Специальное предложение недели",
            source_language="EN",
            confidence=0.9,
            bbox=wrapped_bbox,
            polygon=[
                (wrapped_bbox.x, wrapped_bbox.y),
                (wrapped_bbox.x + wrapped_bbox.width, wrapped_bbox.y),
                (wrapped_bbox.x + wrapped_bbox.width, wrapped_bbox.y + wrapped_bbox.height),
                (wrapped_bbox.x, wrapped_bbox.y + wrapped_bbox.height),
            ],
            ocr=OCRData(raw_text="SPECIAL", confidence=0.9),
            visual=VisualStyle(
                background_color="#FFFFFF",
                text_color="#141414",
                estimated_text_height_px=28,
                font_size_estimate=22,
                alignment="left",
                vertical_alignment="top",
            ),
            render_hints=RenderHints(
                available_width_px=wrapped_bbox.width,
                available_height_px=wrapped_bbox.height,
                max_line_count=3,
                font_scale_min=0.55,
                font_scale_max=1.0,
            ),
            review=ReviewInfo(),
            background=BackgroundInfo(
                background_type="solid",
                dominant_color="#FFFFFF",
                preferred_restore_method="fill",
            ),
            render_geometry=RenderGeometry(mask_bbox=wrapped_bbox, mask_confidence=0.80),
        )
    )
    masks["demo_wrapped"] = wrapped_mask

    rotated_bbox = BoundingBox(x=495, y=40, width=170, height=75)
    rotated_mask = np.zeros((rotated_bbox.height, rotated_bbox.width), dtype=np.uint8)
    rotated_patch = image[
        rotated_bbox.y : rotated_bbox.y + rotated_bbox.height,
        rotated_bbox.x : rotated_bbox.x + rotated_bbox.width,
    ]
    rotated_mask[np.mean(rotated_patch, axis=2) > 200] = MASK_FOREGROUND_VALUE
    blocks.append(
        TextBlock(
            id="demo_rotated",
            original_text="HOT",
            translated_text="ГОРЯЧЕЕ",
            source_language="EN",
            confidence=0.9,
            bbox=rotated_bbox,
            polygon=[
                (rotated_bbox.x, rotated_bbox.y),
                (rotated_bbox.x + rotated_bbox.width, rotated_bbox.y),
                (rotated_bbox.x + rotated_bbox.width, rotated_bbox.y + rotated_bbox.height),
                (rotated_bbox.x, rotated_bbox.y + rotated_bbox.height),
            ],
            ocr=OCRData(raw_text="HOT", confidence=0.9),
            visual=VisualStyle(
                background_color="#B43C3C",
                text_color="#FFF0DC",
                estimated_text_height_px=28,
                font_size_estimate=24,
                alignment="center",
                vertical_alignment="center",
                rotation_deg=-18.0,
            ),
            render_hints=RenderHints(
                available_width_px=rotated_bbox.width,
                available_height_px=rotated_bbox.height,
                max_line_count=1,
            ),
            review=ReviewInfo(),
            background=BackgroundInfo(
                background_type="solid",
                dominant_color="#B43C3C",
                preferred_restore_method="fill",
            ),
            render_geometry=RenderGeometry(mask_bbox=rotated_bbox, mask_confidence=0.78),
        )
    )
    masks["demo_rotated"] = rotated_mask

    return Image.fromarray(image), blocks, masks


def main() -> None:
    image, blocks, masks = _build_demo_image()
    localized, restore_results, render_results = render_localized_image(image, blocks, masks=masks)

    output_dir = CONFIG_ROOT / "data" / "output"
    localized_path = save_localized_image(
        localized,
        output_dir=output_dir,
        source_image_id="demo_r0_4",
    )

    print("Text rendering demo complete.")
    for restore in restore_results:
        print(f"- restore {restore.method}: success={restore.success}, confidence={restore.confidence}")
    for render in render_results:
        print(
            f"- render {render.block_id}: success={render.success}, "
            f"font_size={render.font_size_px}, lines={render.rendered_lines}"
        )
    print(f"Localized image: {CONFIG_ROOT / localized_path}")


if __name__ == "__main__":
    main()
