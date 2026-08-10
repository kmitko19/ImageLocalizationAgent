"""Deterministic pipeline demo fixtures aligned with EasyOCR reading order."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from imageloc.config import PROJECT_ROOT
from imageloc.models.raw_blocks import RawOCRBlock, RawVisionBlock
from imageloc.providers.easyocr_provider import parse_easyocr_results
from imageloc.utils import image_loader

DEMO_SOURCE_IMAGE_ID = "demo_r0_8"

# Reading-order bboxes: row 1 left-to-right (SALE, DETAILS, HOT), row 2 (FX, EMPTY, FAIL).
DEMO_OCR_RESULTS = [
    ([[20.0, 20.0], [220.0, 20.0], [220.0, 80.0], [20.0, 80.0]], "SALE", 0.92),
    ([[240.0, 20.0], [470.0, 20.0], [470.0, 80.0], [240.0, 80.0]], "DETAILS", 0.88),
    ([[490.0, 20.0], [680.0, 20.0], [680.0, 80.0], [490.0, 80.0]], "HOT", 0.9),
    ([[20.0, 130.0], [260.0, 130.0], [260.0, 200.0], [20.0, 200.0]], "FX", 0.9),
    ([[280.0, 130.0], [430.0, 130.0], [430.0, 200.0], [280.0, 200.0]], "EMPTY", 0.85),
    ([[450.0, 130.0], [540.0, 130.0], [540.0, 200.0], [450.0, 200.0]], "FAIL", 0.85),
]

DEMO_IMAGE_SPECS: list[tuple[tuple[int, int, int, int], tuple[int, int, int], str]] = [
    ((20, 20, 220, 80), (255, 255, 255), "SALE"),
    ((240, 20, 470, 80), (255, 255, 255), "DETAILS"),
    ((490, 20, 680, 80), (180, 60, 60), "HOT"),
    ((20, 130, 260, 200), (180, 60, 60), "FX"),
    ((280, 130, 430, 200), (255, 255, 255), "EMPTY"),
    ((450, 130, 540, 200), (255, 255, 255), "FAIL"),
]

DEMO_TRANSLATIONS_BY_TEXT: dict[str, str] = {
    "SALE": "SALE",
    "DETAILS": "Details info",
    "HOT": "HOT",
    "FX": "Effects",
    "EMPTY": "   ",
    "FAIL": "Too long for tiny block",
}


def sorted_demo_ocr_blocks() -> list[RawOCRBlock]:
    """Return OCR blocks with IDs assigned by EasyOCR reading-order rules."""
    return parse_easyocr_results(DEMO_OCR_RESULTS)


def build_demo_vision_blocks(ocr_blocks: list[RawOCRBlock] | None = None) -> list[RawVisionBlock]:
    blocks = ocr_blocks or sorted_demo_ocr_blocks()
    return [
        RawVisionBlock(
            ocr_block_id=block.id,
            text=block.raw_text,
            corrected_text=block.raw_text,
            semantic_role="heading" if block.raw_text == "SALE" else "label",
            language="EN",
            confidence=0.9,
            alignment="center" if block.raw_text != "DETAILS" else "left",
        )
        for block in blocks
    ]


def build_demo_translation_payload(ocr_blocks: list[RawOCRBlock] | None = None) -> dict[str, Any]:
    blocks = ocr_blocks or sorted_demo_ocr_blocks()
    return {
        "translations": [
            {
                "id": block.id,
                "original_text": block.raw_text,
                "translated_text": DEMO_TRANSLATIONS_BY_TEXT[block.raw_text],
                "detected_language": "EN",
                "translation_confidence": 0.9,
            }
            for block in blocks
        ]
    }


def build_demo_translation_response_json(ocr_blocks: list[RawOCRBlock] | None = None) -> str:
    return json.dumps(build_demo_translation_payload(ocr_blocks))


def build_demo_source_image() -> Image.Image:
    """Build the deterministic RGB demo canvas used as pipeline input."""
    image = Image.new("RGB", (920, 420), color=(245, 240, 232))
    draw = ImageDraw.Draw(image)
    for rect, fill, label in DEMO_IMAGE_SPECS:
        draw.rectangle(rect, fill=fill)
        draw.text(
            (rect[0] + 10, rect[1] + 8),
            label,
            fill=(240, 240, 240) if fill != (255, 255, 255) else (20, 20, 20),
        )
    return image


def persist_source_image_asset(
    image: Image.Image,
    *,
    output_dir: Path,
    source_image_id: str,
) -> str:
    """Persist the exact pipeline input image under output_dir/source_image_id/source.png."""
    dest_dir = output_dir / source_image_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / "source.png"
    temp_path = target_path.with_suffix(".tmp.png")
    image.convert("RGB").save(temp_path, format="PNG")
    temp_path.replace(target_path)
    try:
        return str(target_path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return f"data/output/{source_image_id}/source.png"


def demo_required_asset_paths(source_image_id: str = DEMO_SOURCE_IMAGE_ID) -> dict[str, str]:
    """Relative project paths for the five required deterministic demo assets."""
    base = f"data/output/{source_image_id}"
    return {
        "source": f"{base}/source.png",
        "result": f"{base}/result.json",
        "clean_background": f"{base}/render_assets/clean_background.png",
        "localized_image": f"{base}/render_assets/localized_image.png",
        "renderer_result": f"{base}/render_assets/renderer_result.json",
    }


def resolve_demo_asset_path(relative_path: str, *, project_root: Path = PROJECT_ROOT) -> Path:
    return project_root / Path(relative_path.replace("/", "\\"))


def make_demo_load_patch(
    original_load: Callable[..., image_loader.LoadedImage],
    *,
    output_dir: Path,
    source_image_id: str = DEMO_SOURCE_IMAGE_ID,
) -> Callable[..., image_loader.LoadedImage]:
    """Patch load_image so demo assets use a stable id and persisted source.png."""

    def patched_load(input_path, output_dir=None, is_test_case=False, **_kwargs):
        loaded = original_load(input_path, output_dir, is_test_case)
        source_path = persist_source_image_asset(
            loaded.image,
            output_dir=output_dir,
            source_image_id=source_image_id,
        )
        return image_loader.LoadedImage(
            image=loaded.image,
            source_image_id=source_image_id,
            source_image=loaded.source_image.model_copy(
                update={
                    "path": source_path,
                    "filename": "source.png",
                    "width": loaded.image.width,
                    "height": loaded.image.height,
                    "format": "PNG",
                }
            ),
        )

    return patched_load
