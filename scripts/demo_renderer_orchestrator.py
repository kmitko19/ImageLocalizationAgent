"""Development demo for Renderer R0.7 orchestrator."""

from __future__ import annotations

import json
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
from imageloc.renderer.orchestrator import render_image_localization  # noqa: E402
from imageloc.utils.text_mask_analyzer import MASK_FOREGROUND_VALUE  # noqa: E402


def _mask_from_patch(patch: np.ndarray, *, bright_text: bool) -> np.ndarray:
    mask = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint8)
    if bright_text:
        corner_samples = np.concatenate(
            [
                patch[0:3, 0:3].reshape(-1, 3),
                patch[0:3, -3:].reshape(-1, 3),
                patch[-3:, 0:3].reshape(-1, 3),
                patch[-3:, -3:].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(corner_samples, axis=0).astype(np.float32)
        delta = patch.astype(np.float32) - background
        luminance_delta = 0.299 * delta[:, :, 0] + 0.587 * delta[:, :, 1] + 0.114 * delta[:, :, 2]
        mask[luminance_delta > 35.0] = MASK_FOREGROUND_VALUE
    else:
        mask[np.mean(patch, axis=2) < 200] = MASK_FOREGROUND_VALUE
    return mask


def _build_demo() -> tuple[Image.Image, list[TextBlock], dict[str, np.ndarray]]:
    canvas = Image.new("RGB", (920, 420), color=(245, 240, 232))
    draw = ImageDraw.Draw(canvas)

    specs = [
        ("normal", (20, 20, 220, 75), (255, 255, 255), "NORMAL"),
        ("multiline", (240, 20, 470, 95), (255, 255, 255), "MULTI"),
        ("rotated", (490, 20, 680, 75), (255, 255, 255), "ROT"),
        ("perspective", (700, 20, 890, 90), (255, 255, 255), "PERSP"),
        ("effects", (20, 130, 260, 200), (180, 60, 60), "FX"),
        ("empty", (280, 130, 430, 190), (255, 255, 255), "EMPTY"),
        ("failed", (450, 130, 540, 165), (255, 255, 255), "FAIL"),
    ]
    for _, rect, fill, label in specs:
        draw.rectangle(rect, fill=fill)
        draw.text((rect[0] + 10, rect[1] + 8), label, fill=(240, 240, 240) if fill != (255, 255, 255) else (30, 30, 30))

    image = np.array(canvas)
    blocks: list[TextBlock] = []
    masks: dict[str, np.ndarray] = {}

    def add_block(
        block_id: str,
        rect: tuple[int, int, int, int],
        *,
        translated_text: str | None,
        visual: VisualStyle,
        hints: RenderHints,
        background: BackgroundInfo,
        bright_text: bool = False,
        render_geometry: RenderGeometry | None = None,
        polygon: list[tuple[int, int]] | None = None,
    ) -> None:
        x1, y1, x2, y2 = rect
        bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        poly = polygon or [
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            (bbox.x, bbox.y + bbox.height),
        ]
        patch = image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
        blocks.append(
            TextBlock(
                id=block_id,
                original_text=block_id,
                translated_text=translated_text,
                source_language="EN",
                confidence=0.9,
                bbox=bbox,
                polygon=poly,
                ocr=OCRData(raw_text=block_id, confidence=0.9),
                visual=visual,
                render_hints=hints,
                review=ReviewInfo(),
                background=background,
                render_geometry=render_geometry or RenderGeometry(mask_bbox=bbox, mask_confidence=0.8),
            )
        )
        masks[block_id] = _mask_from_patch(patch, bright_text=bright_text)

    base_visual = dict(
        text_color="#141414",
        background_color="#FFFFFF",
        estimated_text_height_px=28,
        font_size_estimate=22,
        alignment="center",
        vertical_alignment="center",
    )
    base_bg = BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill")

    add_block(
        "demo_normal",
        specs[0][1],
        translated_text="Обычный блок",
        visual=VisualStyle(**base_visual),
        hints=RenderHints(available_width_px=180, available_height_px=45, max_line_count=1),
        background=base_bg,
    )
    add_block(
        "demo_multiline",
        specs[1][1],
        translated_text="Длинный перевод для нескольких строк",
        visual=VisualStyle(
            text_color="#141414",
            background_color="#FFFFFF",
            estimated_text_height_px=28,
            font_size_estimate=22,
            alignment="left",
            vertical_alignment="top",
            line_height_ratio=1.3,
        ),
        hints=RenderHints(available_width_px=210, available_height_px=70, max_line_count=3),
        background=base_bg,
    )
    add_block(
        "demo_rotated",
        specs[2][1],
        translated_text="Поворот +18",
        visual=VisualStyle(**base_visual, rotation_deg=18.0),
        hints=RenderHints(available_width_px=170, available_height_px=45, max_line_count=1),
        background=base_bg,
    )
    persp_bbox = BoundingBox(x=700, y=20, width=190, height=70)
    persp_quad = [(700, 20), (890, 28), (882, 90), (708, 82)]
    add_block(
        "demo_perspective",
        specs[3][1],
        translated_text="Перспектива",
        visual=VisualStyle(**base_visual),
        hints=RenderHints(available_width_px=180, available_height_px=60, max_line_count=1),
        background=base_bg,
        render_geometry=RenderGeometry(
            mask_bbox=persp_bbox,
            mask_confidence=0.8,
            perspective_detected=True,
            perspective_quad=persp_quad,
        ),
        polygon=persp_quad,
    )
    add_block(
        "demo_effects",
        specs[4][1],
        translated_text="Эффекты",
        visual=VisualStyle(
            text_color="#FFFFFF",
            background_color="#B43C3C",
            estimated_text_height_px=28,
            font_size_estimate=22,
            alignment="center",
            vertical_alignment="center",
            fill_type="gradient",
            gradient_start_color="#FFD166",
            gradient_end_color="#EF476F",
            gradient_angle_deg=90.0,
            gradient_confidence=0.9,
            shadow_color="#000000",
            shadow_opacity=0.5,
            shadow_offset_x_px=3.0,
            shadow_offset_y_px=3.0,
            shadow_blur_px=2.0,
            shadow_confidence=0.9,
        ),
        hints=RenderHints(available_width_px=220, available_height_px=60, max_line_count=1),
        background=BackgroundInfo(background_type="solid", dominant_color="#B43C3C", preferred_restore_method="fill"),
        bright_text=True,
    )
    add_block(
        "demo_empty",
        specs[5][1],
        translated_text="   ",
        visual=VisualStyle(**base_visual),
        hints=RenderHints(available_width_px=130, available_height_px=50, max_line_count=1),
        background=base_bg,
    )
    add_block(
        "demo_failed",
        specs[6][1],
        translated_text="Слишком длинный текст для крошечного блока",
        visual=VisualStyle(
            text_color="#141414",
            background_color="#FFFFFF",
            estimated_text_height_px=28,
            font_size_estimate=24,
            alignment="center",
            vertical_alignment="center",
        ),
        hints=RenderHints(
            available_width_px=80,
            available_height_px=30,
            max_line_count=1,
            font_scale_min=0.85,
            font_scale_max=1.0,
        ),
        background=base_bg,
    )

    return Image.fromarray(image), blocks, masks


def main() -> None:
    image, blocks, masks = _build_demo()
    result = render_image_localization(
        image,
        blocks,
        masks=masks,
        source_image_id="demo_r0_7",
        save_assets=True,
        output_dir=CONFIG_ROOT / "data" / "output",
    )

    print("Renderer orchestrator demo complete.")
    print(f"- success: {result.success}")
    print(f"- rendered: {result.rendered_block_count}")
    print(f"- skipped: {result.skipped_block_count}")
    print(f"- failed_block_ids: {result.failed_block_ids}")
    for block_result in result.block_results:
        print(
            f"- block {block_result.block_id}: status={block_result.status}, "
            f"restore={block_result.restore_method}, failure={block_result.failure_reason}"
        )
    for key, path in result.output_paths.items():
        print(f"- {key}: {CONFIG_ROOT / path}")

    result_json = CONFIG_ROOT / result.output_paths["renderer_result"]
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    print(f"- renderer_result.success in JSON: {payload['success']}")


if __name__ == "__main__":
    main()
