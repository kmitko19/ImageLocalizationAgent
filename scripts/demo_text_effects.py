"""Development demo for Renderer R0.6 geometry and visual effects."""

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
from imageloc.renderer.text_renderer import (  # noqa: E402
    compute_content_box,
    render_localized_image,
    save_localized_image,
)
from imageloc.utils.text_mask_analyzer import MASK_FOREGROUND_VALUE  # noqa: E402


def _mask_from_patch(patch: np.ndarray, *, bright_text: bool) -> np.ndarray:
    mask = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.uint8)
    if bright_text:
        mask[np.mean(patch, axis=2) > 200] = MASK_FOREGROUND_VALUE
    else:
        mask[np.mean(patch, axis=2) < 200] = MASK_FOREGROUND_VALUE
    return mask


def _build_demo() -> tuple[Image.Image, list[TextBlock], dict[str, np.ndarray], dict[str, str]]:
    canvas = Image.new("RGB", (960, 560), color=(245, 240, 232))
    draw = ImageDraw.Draw(canvas)

    specs = [
        ("normal", (20, 20, 200, 70), (255, 255, 255), "NORMAL"),
        ("rot_pos", (220, 20, 400, 70), (255, 255, 255), "+20"),
        ("rot_neg", (420, 20, 600, 70), (255, 255, 255), "-20"),
        ("perspective", (620, 20, 820, 90), (255, 255, 255), "PERSP"),
        ("persp_bad", (20, 120, 200, 190), (255, 255, 255), "BAD"),
        ("shadow", (220, 120, 420, 190), (255, 255, 255), "SHADOW"),
        ("gradient", (440, 120, 640, 190), (255, 255, 255), "GRAD"),
        ("stroke_shadow", (660, 120, 860, 190), (180, 60, 60), "STROKE"),
        ("rot_shadow", (20, 220, 220, 290), (255, 255, 255), "ROT+SH"),
        ("persp_grad", (240, 220, 460, 300), (255, 255, 255), "P+G"),
    ]
    labels = {name: label for name, rect, fill, label in specs}
    for _, rect, fill, label in specs:
        draw.rectangle(rect, fill=fill)
        draw.text((rect[0] + 10, rect[1] + 8), label, fill=(30, 30, 30))

    image = np.array(canvas)
    blocks: list[TextBlock] = []
    masks: dict[str, np.ndarray] = {}

    def add_block(
        block_id: str,
        rect: tuple[int, int, int, int],
        *,
        translated_text: str,
        visual: VisualStyle,
        render_geometry: RenderGeometry | None = None,
        polygon: list[tuple[int, int]] | None = None,
        bright_text: bool = False,
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
                render_hints=RenderHints(
                    available_width_px=bbox.width,
                    available_height_px=bbox.height,
                    max_line_count=2,
                ),
                review=ReviewInfo(),
                background=BackgroundInfo(
                    background_type="solid",
                    dominant_color="#FFFFFF",
                    preferred_restore_method="fill",
                ),
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

    add_block(
        "demo_normal",
        specs[0][1],
        translated_text="Обычный текст",
        visual=VisualStyle(**base_visual),
    )
    add_block(
        "demo_rot_pos",
        specs[1][1],
        translated_text="Поворот +20°",
        visual=VisualStyle(**base_visual, rotation_deg=20.0),
    )
    add_block(
        "demo_rot_neg",
        specs[2][1],
        translated_text="Поворот -20°",
        visual=VisualStyle(**base_visual, rotation_deg=-20.0),
    )
    persp_bbox = BoundingBox(x=620, y=20, width=200, height=70)
    persp_quad = [(620, 20), (820, 28), (812, 90), (628, 82)]
    add_block(
        "demo_perspective",
        specs[3][1],
        translated_text="Перспектива",
        visual=VisualStyle(**base_visual),
        render_geometry=RenderGeometry(
            mask_bbox=persp_bbox,
            mask_confidence=0.8,
            perspective_detected=True,
            perspective_quad=persp_quad,
        ),
        polygon=persp_quad,
    )
    bad_bbox = BoundingBox(x=20, y=120, width=180, height=70)
    bad_quad = [(20, 120), (22, 120), (21, 121), (20, 122)]
    add_block(
        "demo_persp_bad",
        specs[4][1],
        translated_text="Fallback",
        visual=VisualStyle(**base_visual),
        render_geometry=RenderGeometry(
            mask_bbox=bad_bbox,
            mask_confidence=0.8,
            perspective_detected=True,
            perspective_quad=bad_quad,
        ),
        polygon=bad_quad,
    )
    add_block(
        "demo_shadow",
        specs[5][1],
        translated_text="Тень",
        visual=VisualStyle(
            **base_visual,
            shadow_color="#000000",
            shadow_opacity=0.55,
            shadow_offset_x_px=4.0,
            shadow_offset_y_px=4.0,
            shadow_blur_px=3.0,
            shadow_confidence=0.9,
        ),
    )
    add_block(
        "demo_gradient",
        specs[6][1],
        translated_text="Градиент",
        visual=VisualStyle(
            **base_visual,
            fill_type="gradient",
            gradient_start_color="#E63946",
            gradient_end_color="#1D3557",
            gradient_angle_deg=90.0,
            gradient_confidence=0.9,
        ),
    )
    add_block(
        "demo_stroke_shadow",
        specs[7][1],
        translated_text="Обводка",
        visual=VisualStyle(
            text_color="#FFFFFF",
            background_color="#B43C3C",
            estimated_text_height_px=28,
            font_size_estimate=22,
            alignment="center",
            vertical_alignment="center",
            stroke_color="#FFFFFF",
            stroke_width_px=2.0,
            stroke_confidence=0.9,
            shadow_color="#000000",
            shadow_opacity=0.5,
            shadow_offset_x_px=3.0,
            shadow_offset_y_px=3.0,
            shadow_blur_px=2.0,
            shadow_confidence=0.9,
        ),
        bright_text=True,
    )
    add_block(
        "demo_rot_shadow",
        specs[8][1],
        translated_text="Поворот+тень",
        visual=VisualStyle(
            **base_visual,
            rotation_deg=15.0,
            shadow_color="#000000",
            shadow_opacity=0.5,
            shadow_offset_x_px=3.0,
            shadow_offset_y_px=3.0,
            shadow_blur_px=2.0,
            shadow_confidence=0.9,
        ),
    )
    pg_bbox = BoundingBox(x=240, y=220, width=220, height=80)
    pg_quad = [(240, 220), (460, 228), (452, 300), (248, 292)]
    add_block(
        "demo_persp_grad",
        specs[9][1],
        translated_text="Persp+Grad",
        visual=VisualStyle(
            **base_visual,
            fill_type="gradient",
            gradient_start_color="#06D6A0",
            gradient_end_color="#118AB2",
            gradient_angle_deg=0.0,
            gradient_confidence=0.9,
        ),
        render_geometry=RenderGeometry(
            mask_bbox=pg_bbox,
            mask_confidence=0.8,
            perspective_detected=True,
            perspective_quad=pg_quad,
        ),
        polygon=pg_quad,
    )

    return Image.fromarray(image), blocks, masks, labels


def _build_diagnostic(
    localized: Image.Image,
    blocks: list[TextBlock],
    labels: dict[str, str],
) -> Image.Image:
    diagnostic = localized.copy()
    draw = ImageDraw.Draw(diagnostic)
    for block in blocks:
        bbox = block.bbox
        draw.rectangle(
            (bbox.x, bbox.y, bbox.x + bbox.width - 1, bbox.y + bbox.height - 1),
            outline=(255, 64, 64),
            width=2,
        )
        _, content = compute_content_box(block)
        draw.rectangle(
            (
                bbox.x + content.x,
                bbox.y + content.y,
                bbox.x + content.x + content.width - 1,
                bbox.y + content.y + content.height - 1,
            ),
            outline=(64, 160, 255),
            width=1,
        )
        geometry = block.render_geometry
        if geometry and geometry.perspective_quad and len(geometry.perspective_quad) == 4:
            draw.polygon(geometry.perspective_quad, outline=(255, 180, 0), width=2)
        label = labels.get(block.id.replace("demo_", ""), block.id)
        draw.text((bbox.x + 4, bbox.y + 4), label, fill=(20, 20, 20))
    return diagnostic


def main() -> None:
    image, blocks, masks, labels = _build_demo()
    localized, restore_results, render_results = render_localized_image(image, blocks, masks=masks)

    output_dir = CONFIG_ROOT / "data" / "output"
    assets_dir = output_dir / "demo_r0_6" / "render_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    localized_path = save_localized_image(
        localized,
        output_dir=output_dir,
        source_image_id="demo_r0_6",
    )
    diagnostic = _build_diagnostic(localized, blocks, labels)
    diagnostic_path = assets_dir / "effects_diagnostic.png"
    diagnostic.save(diagnostic_path)

    print("Text effects demo complete.")
    for restore in restore_results:
        print(f"- restore {restore.method}: success={restore.success}")
    for render in render_results:
        print(
            f"- render {render.block_id}: rotation={render.applied_rotation}, "
            f"perspective={render.perspective_applied}, shadow={render.shadow_applied}, "
            f"gradient={render.gradient_applied}, warnings={render.warnings}"
        )
    print(f"Localized image: {CONFIG_ROOT / localized_path}")
    print(f"Diagnostic: {diagnostic_path}")


if __name__ == "__main__":
    main()
