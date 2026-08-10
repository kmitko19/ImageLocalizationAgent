"""Development demo for Renderer R0.5 layout and typography refinement."""

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


def _build_demo() -> tuple[Image.Image, list[TextBlock], dict[str, np.ndarray]]:
    canvas = Image.new("RGB", (900, 520), color=(245, 240, 232))
    draw = ImageDraw.Draw(canvas)

    specs = [
        ("centered", (30, 30, 210, 85), (30, 80, 160), "SALE", True),
        ("long", (250, 25, 430, 95), (255, 255, 255), "OFFER", False),
        ("multiline", (460, 25, 650, 120), (255, 255, 255), "DETAILS", False),
        ("left", (30, 150, 220, 210), (180, 60, 60), "LEFT", True),
        ("right", (250, 150, 430, 210), (180, 60, 60), "RIGHT", True),
        ("spacing", (460, 150, 650, 210), (255, 255, 255), "TRACK", False),
        ("lineheight", (30, 250, 260, 340), (255, 255, 255), "LINES", False),
        ("rotated", (680, 250, 860, 330), (180, 60, 60), "HOT", True),
    ]
    for _, rect, fill, label, bright in specs:
        draw.rectangle(rect, fill=fill)
        draw.text((rect[0] + 12, rect[1] + 10), label, fill=(240, 240, 240) if bright else (20, 20, 20))

    image = np.array(canvas)
    blocks: list[TextBlock] = []
    masks: dict[str, np.ndarray] = {}

    def add_block(
        block_id: str,
        rect: tuple[int, int, int, int],
        *,
        translated_text: str,
        visual: VisualStyle,
        hints: RenderHints,
        background: BackgroundInfo,
        bright_text: bool,
    ) -> None:
        x1, y1, x2, y2 = rect
        bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        patch = image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
        blocks.append(
            TextBlock(
                id=block_id,
                original_text=block_id,
                translated_text=translated_text,
                source_language="EN",
                confidence=0.9,
                bbox=bbox,
                polygon=[(bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), (bbox.x, bbox.y + bbox.height)],
                ocr=OCRData(raw_text=block_id, confidence=0.9),
                visual=visual,
                render_hints=hints,
                review=ReviewInfo(),
                background=background,
                render_geometry=RenderGeometry(mask_bbox=bbox, mask_confidence=0.8),
            )
        )
        masks[block_id] = _mask_from_patch(patch, bright_text=bright_text)

    add_block(
        "demo_centered",
        specs[0][1],
        translated_text="РАСПРОДАЖА",
        visual=VisualStyle(
            text_color="#FFFFFF",
            background_color="#1E509F",
            estimated_text_height_px=34,
            font_size_estimate=28,
            alignment="center",
            vertical_alignment="center",
        ),
        hints=RenderHints(available_width_px=180, available_height_px=55, max_line_count=1),
        background=BackgroundInfo(background_type="solid", dominant_color="#1E509F", preferred_restore_method="fill"),
        bright_text=True,
    )
    add_block(
        "demo_long",
        specs[1][1],
        translated_text="Очень длинный перевод для ограниченного блока",
        visual=VisualStyle(
            text_color="#141414",
            background_color="#FFFFFF",
            estimated_text_height_px=24,
            font_size_estimate=20,
            alignment="left",
            vertical_alignment="top",
        ),
        hints=RenderHints(available_width_px=180, available_height_px=70, max_line_count=3, font_scale_min=0.45, font_scale_max=1.0),
        background=BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill"),
        bright_text=False,
    )
    add_block(
        "demo_multiline",
        specs[2][1],
        translated_text="Специальное предложение недели",
        visual=VisualStyle(
            text_color="#141414",
            background_color="#FFFFFF",
            estimated_text_height_px=24,
            font_size_estimate=20,
            alignment="left",
            vertical_alignment="top",
        ),
        hints=RenderHints(available_width_px=190, available_height_px=95, max_line_count=3),
        background=BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill"),
        bright_text=False,
    )
    add_block(
        "demo_left",
        specs[3][1],
        translated_text="СЛЕВА",
        visual=VisualStyle(
            text_color="#FFF0DC",
            background_color="#B43C3C",
            estimated_text_height_px=28,
            font_size_estimate=24,
            alignment="left",
            vertical_alignment="center",
        ),
        hints=RenderHints(available_width_px=190, available_height_px=60, max_line_count=1),
        background=BackgroundInfo(background_type="solid", dominant_color="#B43C3C", preferred_restore_method="fill"),
        bright_text=True,
    )
    add_block(
        "demo_right",
        specs[4][1],
        translated_text="СПРАВА",
        visual=VisualStyle(
            text_color="#FFF0DC",
            background_color="#B43C3C",
            estimated_text_height_px=28,
            font_size_estimate=24,
            alignment="right",
            vertical_alignment="center",
        ),
        hints=RenderHints(available_width_px=180, available_height_px=60, max_line_count=1),
        background=BackgroundInfo(background_type="solid", dominant_color="#B43C3C", preferred_restore_method="fill"),
        bright_text=True,
    )
    add_block(
        "demo_spacing",
        specs[5][1],
        translated_text="ИНТЕРВАЛ",
        visual=VisualStyle(
            text_color="#141414",
            background_color="#FFFFFF",
            estimated_text_height_px=24,
            font_size_estimate=20,
            alignment="center",
            vertical_alignment="center",
            letter_spacing_px=5.0,
            letter_spacing_confidence=0.9,
        ),
        hints=RenderHints(available_width_px=190, available_height_px=60, max_line_count=1),
        background=BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill"),
        bright_text=False,
    )
    add_block(
        "demo_lineheight",
        specs[6][1],
        translated_text="Первая строка вторая строка",
        visual=VisualStyle(
            text_color="#141414",
            background_color="#FFFFFF",
            estimated_text_height_px=22,
            font_size_estimate=18,
            alignment="left",
            vertical_alignment="top",
            line_height_ratio=1.35,
            line_height_confidence=0.9,
        ),
        hints=RenderHints(available_width_px=230, available_height_px=90, max_line_count=2, font_scale_min=0.5, font_scale_max=1.0),
        background=BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill"),
        bright_text=False,
    )
    add_block(
        "demo_rotated",
        specs[7][1],
        translated_text="ГОРЯЧЕЕ",
        visual=VisualStyle(
            text_color="#FFF0DC",
            background_color="#B43C3C",
            estimated_text_height_px=26,
            font_size_estimate=22,
            alignment="center",
            vertical_alignment="center",
            rotation_deg=-16.0,
        ),
        hints=RenderHints(available_width_px=180, available_height_px=80, max_line_count=1),
        background=BackgroundInfo(background_type="solid", dominant_color="#B43C3C", preferred_restore_method="fill"),
        bright_text=True,
    )

    return Image.fromarray(image), blocks, masks


def _build_diagnostic(
    original: Image.Image,
    localized: Image.Image,
    blocks: list[TextBlock],
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
    return diagnostic


def main() -> None:
    image, blocks, masks = _build_demo()
    localized, restore_results, render_results = render_localized_image(image, blocks, masks=masks)

    output_dir = CONFIG_ROOT / "data" / "output"
    assets_dir = output_dir / "demo_r0_5" / "render_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    localized_path = save_localized_image(
        localized,
        output_dir=output_dir,
        source_image_id="demo_r0_5",
    )
    diagnostic = _build_diagnostic(image, localized, blocks)
    diagnostic_path = assets_dir / "layout_diagnostic.png"
    diagnostic.save(diagnostic_path)

    print("Text layout demo complete.")
    for restore in restore_results:
        print(f"- restore {restore.method}: success={restore.success}")
    for render in render_results:
        print(
            f"- render {render.block_id}: success={render.success}, "
            f"font_size={render.font_size_px}, padding={render.content_padding_px}, lines={render.rendered_lines}"
        )
    print(f"Localized image: {CONFIG_ROOT / localized_path}")
    print(f"Diagnostic: {diagnostic_path}")


if __name__ == "__main__":
    main()
