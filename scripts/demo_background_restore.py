"""Development demo for Renderer R0.3 background restoration.

Creates a synthetic image with trusted masks, restores backgrounds, and writes
`data/output/demo_r0_3/render_assets/clean_background.png` for manual inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
from imageloc.renderer.background_restorer import (  # noqa: E402
    _extract_patch,
    _mask_to_patch_coordinates,
    _patch_padding_px,
    restore_backgrounds,
    restore_text_background,
    sample_solid_fill_color,
    save_clean_background,
)
from imageloc.utils.text_mask_analyzer import MASK_FOREGROUND_VALUE, expand_text_mask  # noqa: E402

DIAGNOSTIC_BLOCK_ID = "demo_new"


def _build_demo_image() -> tuple[Image.Image, list[TextBlock], dict[str, np.ndarray]]:
    canvas = Image.new("RGB", (640, 240), color=(245, 240, 232))
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 40), "SALE", fill=(20, 20, 20))
    draw.rectangle((220, 35, 320, 85), fill=(30, 80, 160))
    draw.text((220, 45), "NEW", fill=(240, 240, 240))

    for y in range(120, 220):
        value = int(255 * (y - 120) / 99)
        for x in range(40, 320):
            canvas.putpixel((x, y), (value, value // 2, 255 - value))
    draw.text((60, 145), "GRADIENT", fill=(10, 10, 10))

    image = np.array(canvas)
    blocks: list[TextBlock] = []
    masks: dict[str, np.ndarray] = {}

    sale_bbox = BoundingBox(x=35, y=35, width=120, height=50)
    sale_mask = np.zeros((sale_bbox.height, sale_bbox.width), dtype=np.uint8)
    sale_patch = image[sale_bbox.y : sale_bbox.y + sale_bbox.height, sale_bbox.x : sale_bbox.x + sale_bbox.width]
    sale_mask[np.mean(sale_patch, axis=2) < 200] = MASK_FOREGROUND_VALUE
    blocks.append(
        TextBlock(
            id="demo_sale",
            original_text="SALE",
            translated_text=None,
            source_language="EN",
            confidence=0.9,
            bbox=sale_bbox,
            polygon=[
                (sale_bbox.x, sale_bbox.y),
                (sale_bbox.x + sale_bbox.width, sale_bbox.y),
                (sale_bbox.x + sale_bbox.width, sale_bbox.y + sale_bbox.height),
                (sale_bbox.x, sale_bbox.y + sale_bbox.height),
            ],
            ocr=OCRData(raw_text="SALE", confidence=0.9),
            visual=VisualStyle(
                background_color="#F5F0E8",
                text_color="#141414",
                estimated_text_height_px=40,
            ),
            render_hints=RenderHints(available_width_px=sale_bbox.width, available_height_px=sale_bbox.height),
            review=ReviewInfo(),
            background=BackgroundInfo(
                background_type="solid",
                dominant_color="#F5F0E8",
                preferred_restore_method="fill",
            ),
            render_geometry=RenderGeometry(mask_bbox=sale_bbox, mask_confidence=0.82),
        )
    )
    masks["demo_sale"] = sale_mask

    new_bbox = BoundingBox(x=215, y=35, width=110, height=55)
    new_mask = np.zeros((new_bbox.height, new_bbox.width), dtype=np.uint8)
    new_patch = image[new_bbox.y : new_bbox.y + new_bbox.height, new_bbox.x : new_bbox.x + new_bbox.width]
    new_mask[np.mean(new_patch, axis=2) > 200] = MASK_FOREGROUND_VALUE
    blocks.append(
        TextBlock(
            id=DIAGNOSTIC_BLOCK_ID,
            original_text="NEW",
            translated_text=None,
            source_language="EN",
            confidence=0.9,
            bbox=new_bbox,
            polygon=[
                (new_bbox.x, new_bbox.y),
                (new_bbox.x + new_bbox.width, new_bbox.y),
                (new_bbox.x + new_bbox.width, new_bbox.y + new_bbox.height),
                (new_bbox.x, new_bbox.y + new_bbox.height),
            ],
            ocr=OCRData(raw_text="NEW", confidence=0.9),
            visual=VisualStyle(
                background_color="#1E509F",
                text_color="#F0F0F0",
                estimated_text_height_px=40,
            ),
            render_hints=RenderHints(available_width_px=new_bbox.width, available_height_px=new_bbox.height),
            review=ReviewInfo(),
            background=BackgroundInfo(
                background_type="solid",
                dominant_color="#1E509F",
                preferred_restore_method="fill",
            ),
            render_geometry=RenderGeometry(mask_bbox=new_bbox, mask_confidence=0.80),
        )
    )
    masks[DIAGNOSTIC_BLOCK_ID] = new_mask

    grad_bbox = BoundingBox(x=55, y=135, width=210, height=55)
    grad_mask = np.zeros((grad_bbox.height, grad_bbox.width), dtype=np.uint8)
    grad_patch = image[grad_bbox.y : grad_bbox.y + grad_bbox.height, grad_bbox.x : grad_bbox.x + grad_bbox.width]
    grad_mask[np.mean(grad_patch, axis=2) < 120] = MASK_FOREGROUND_VALUE
    blocks.append(
        TextBlock(
            id="demo_gradient",
            original_text="GRADIENT",
            translated_text=None,
            source_language="EN",
            confidence=0.9,
            bbox=grad_bbox,
            polygon=[
                (grad_bbox.x, grad_bbox.y),
                (grad_bbox.x + grad_bbox.width, grad_bbox.y),
                (grad_bbox.x + grad_bbox.width, grad_bbox.y + grad_bbox.height),
                (grad_bbox.x, grad_bbox.y + grad_bbox.height),
            ],
            ocr=OCRData(raw_text="GRADIENT", confidence=0.9),
            visual=VisualStyle(
                background_color="#808080",
                text_color="#0A0A0A",
                estimated_text_height_px=40,
            ),
            render_hints=RenderHints(available_width_px=grad_bbox.width, available_height_px=grad_bbox.height),
            review=ReviewInfo(),
            background=BackgroundInfo(background_type="gradient", preferred_restore_method="gradient"),
            render_geometry=RenderGeometry(mask_bbox=grad_bbox, mask_confidence=0.78),
        )
    )
    masks["demo_gradient"] = grad_mask

    return Image.fromarray(image), blocks, masks


def _mask_panel(mask: np.ndarray, *, color=(255, 80, 80)) -> Image.Image:
    panel = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    panel[mask > 0] = color
    return Image.fromarray(panel)


def _label_panel(image: Image.Image, title: str) -> Image.Image:
    label_height = 24
    canvas = Image.new("RGB", (image.width, image.height + label_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill=(240, 240, 240))
    canvas.paste(image, (0, label_height))
    return canvas


def _build_diagnostic_artifact(
    image: Image.Image,
    block: TextBlock,
    raw_mask: np.ndarray,
    restored: Image.Image,
) -> Image.Image:
    bbox = block.render_geometry.mask_bbox
    assert bbox is not None
    text_height = block.visual.estimated_text_height_px or bbox.height
    expanded_mask = expand_text_mask(raw_mask, estimated_text_height=text_height)
    rgb = np.array(image.convert("RGB"))
    padding = _patch_padding_px(text_height)
    patch_rgb, _, _, patch_bbox = _extract_patch(rgb, bbox, padding=padding)
    patch_inpaint = _mask_to_patch_coordinates(expanded_mask, bbox, patch_bbox)
    patch_raw = _mask_to_patch_coordinates(raw_mask, bbox, patch_bbox)
    assert patch_inpaint is not None and patch_raw is not None

    fill_before_fix_reference = np.median(patch_rgb[patch_inpaint == 0], axis=0)
    fill_after = sample_solid_fill_color(
        patch_rgb,
        patch_inpaint,
        patch_raw,
        hint_rgb=(30, 80, 160),
    )

    original_panel = Image.fromarray(patch_rgb)
    raw_panel = _mask_panel(patch_raw, color=(255, 255, 255))
    expanded_panel = _mask_panel(patch_inpaint, color=(255, 128, 64))
    overlay = patch_rgb.copy()
    overlay[patch_inpaint > 0] = (overlay[patch_inpaint > 0] * 0.45 + np.array((255, 64, 64)) * 0.55).astype(np.uint8)
    overlay_panel = Image.fromarray(overlay)

    restored_rgb = np.array(restored.convert("RGB"))
    restored_patch = restored_rgb[patch_bbox.y : patch_bbox.y + patch_bbox.height, patch_bbox.x : patch_bbox.x + patch_bbox.width]
    restored_panel = Image.fromarray(restored_patch)

    panels = [
        _label_panel(original_panel, "ORIGINAL"),
        _label_panel(raw_panel, "RAW MASK"),
        _label_panel(expanded_panel, "EXPANDED MASK"),
        _label_panel(overlay_panel, "EXPANDED OVERLAY"),
        _label_panel(restored_panel, "RESTORED RESULT"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    composite = Image.new("RGB", (width, height + 28), color=(16, 16, 16))
    draw = ImageDraw.Draw(composite)
    draw.text(
        (8, 6),
        f"fill_old={fill_before_fix_reference.astype(int).tolist()} fill_new={fill_after.astype(int).tolist()}",
        fill=(220, 220, 220),
    )
    x = 0
    for panel in panels:
        composite.paste(panel, (x, 28))
        x += panel.width
    return composite


def main() -> None:
    image, blocks, masks = _build_demo_image()
    diagnostic_block = next(block for block in blocks if block.id == DIAGNOSTIC_BLOCK_ID)
    single_result = restore_text_background(image, diagnostic_block, mask=masks[DIAGNOSTIC_BLOCK_ID])
    restored, results = restore_backgrounds(image, blocks, masks=masks)

    output_dir = CONFIG_ROOT / "data" / "output"
    assets_dir = output_dir / "demo_r0_3" / "render_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    clean_path = save_clean_background(
        restored,
        output_dir=output_dir,
        source_image_id="demo_r0_3",
    )
    diagnostic = _build_diagnostic_artifact(
        image,
        diagnostic_block,
        masks[DIAGNOSTIC_BLOCK_ID],
        single_result.image,
    )
    diagnostic_path = assets_dir / "restoration_diagnostic.png"
    diagnostic.save(diagnostic_path)

    print("Background restoration demo complete.")
    for result in results:
        print(f"- {result.method}: success={result.success}, confidence={result.confidence}")
    print(f"Clean image: {CONFIG_ROOT / clean_path}")
    print(f"Diagnostic: {diagnostic_path}")


if __name__ == "__main__":
    main()
