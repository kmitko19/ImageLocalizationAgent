"""Deterministic visual analysis — color, estimated_text_height_px, rotation.

OpenCV/Pillow sample pixels inside each OCR bbox; optional Vision typography
(font_category, alignment, …) is merged in when the caller passes it — Vision
fields are not re-fetched here.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from imageloc.fusion.ocr_vision_fusion import FusedBlock
from imageloc.models.raw_blocks import BoundingBox, Point
from imageloc.models.text_block import RenderHints, VisualStyle

DECORATIVE_FONT_KEYWORDS = ("script", "decorative", "handwritten", "calligraphy")
DEFAULT_FONT_SIZE_RATIO = 0.75
DECORATIVE_FONT_SIZE_RATIO = 0.65


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def _clamp_bbox(bbox: BoundingBox, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(image_width, bbox.x + bbox.width)
    y2 = min(image_height, bbox.y + bbox.height)
    return x1, y1, x2, y2


def _rotation_from_polygon(polygon: Sequence[Point]) -> float:
    if len(polygon) < 2:
        return 0.0
    x0, y0 = polygon[0]
    x1, y1 = polygon[1]
    return round(math.degrees(math.atan2(y1 - y0, x1 - x0)), 2)


def estimated_text_height_px(polygon: Sequence[Point], bbox: BoundingBox) -> int:
    """Geometric text height from polygon (or bbox fallback). Not font size."""
    if polygon:
        ys = [point[1] for point in polygon]
        return max(1, max(ys) - min(ys))
    return max(1, bbox.height)


def sample_text_and_background_color(
    image_array: np.ndarray,
    bbox: BoundingBox,
) -> tuple[str, str]:
    """Sample dominant text and background colors inside a bbox region."""
    height, width = image_array.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return "#000000", "#FFFFFF"

    region = image_array[y1:y2, x1:x2]
    region_height, region_width = region.shape[:2]

    border_pixels = np.concatenate(
        [
            region[0, :, :].reshape(-1, 3),
            region[-1, :, :].reshape(-1, 3),
            region[:, 0, :].reshape(-1, 3),
            region[:, -1, :].reshape(-1, 3),
        ]
    )
    background = np.median(border_pixels, axis=0).astype(int)

    if region_height > 2 and region_width > 2:
        inner = region[1:-1, 1:-1, :].reshape(-1, 3)
    else:
        inner = region.reshape(-1, 3)

    distances = np.linalg.norm(inner.astype(float) - background.astype(float), axis=1)
    if len(distances) == 0:
        text = np.array([255, 255, 255]) - background
    else:
        threshold = max(30.0, float(np.percentile(distances, 60)))
        contrasting = inner[distances > threshold]
        if len(contrasting) == 0:
            text = np.array([255, 255, 255]) - background
        else:
            text = np.median(contrasting, axis=0).astype(int)

    text = np.clip(text, 0, 255).astype(int)
    return _rgb_to_hex(int(text[0]), int(text[1]), int(text[2])), _rgb_to_hex(
        int(background[0]), int(background[1]), int(background[2])
    )


def estimate_font_size(
    text_height_px: int,
    font_category: str | None = None,
) -> tuple[int, float]:
    """Approximate font size from geometric text height."""
    ratio = DEFAULT_FONT_SIZE_RATIO
    confidence = 0.7

    if font_category:
        lowered = font_category.lower()
        if any(keyword in lowered for keyword in DECORATIVE_FONT_KEYWORDS):
            ratio = DECORATIVE_FONT_SIZE_RATIO
            confidence = 0.45

    return max(1, int(text_height_px * ratio)), confidence


def analyze_visual_style(
    image: Image.Image,
    bbox: BoundingBox,
    polygon: list[Point],
    *,
    font_category: str | None = None,
    font_style: str | None = None,
    font_weight: str | None = None,
    alignment: str | None = None,
) -> VisualStyle:
    """Measure deterministic visual properties for one text block."""
    rgb_array = np.array(image.convert("RGB"))
    text_height = estimated_text_height_px(polygon, bbox)
    text_color, background_color = sample_text_and_background_color(rgb_array, bbox)
    font_size, font_size_confidence = estimate_font_size(text_height, font_category)

    return VisualStyle(
        text_color=text_color,
        background_color=background_color,
        estimated_text_height_px=text_height,
        font_size_estimate=font_size,
        font_size_confidence=font_size_confidence,
        font_category=font_category,
        font_style=font_style,
        font_weight=font_weight,
        alignment=alignment,
        rotation_deg=_rotation_from_polygon(polygon),
    )


def compute_render_hints(bbox: BoundingBox, text: str, visual: VisualStyle) -> RenderHints:
    """Derive layout hints for a future rendering stage."""
    line_height = visual.estimated_text_height_px or max(1, bbox.height)
    estimated_lines = max(1, round(bbox.height / line_height))

    return RenderHints(
        available_width_px=bbox.width,
        available_height_px=bbox.height,
        estimated_lines=estimated_lines,
        target_line_count=estimated_lines,
    )


def analyze_fused_block(
    image: Image.Image,
    block: FusedBlock,
    *,
    font_category: str | None = None,
    font_style: str | None = None,
    font_weight: str | None = None,
    alignment: str | None = None,
) -> tuple[VisualStyle, RenderHints]:
    """Analyze one fused block and return visual style + render hints."""
    visual = analyze_visual_style(
        image,
        block.bbox,
        block.polygon,
        font_category=font_category,
        font_style=font_style,
        font_weight=font_weight,
        alignment=alignment,
    )
    render_hints = compute_render_hints(block.bbox, block.text, visual)
    return visual, render_hints
