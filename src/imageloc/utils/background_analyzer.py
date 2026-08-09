"""Deterministic background analysis for future Renderer inpainting decisions."""

from __future__ import annotations

import math

import cv2
import numpy as np

from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import BackgroundInfo

# Minimum ring pixels required for a meaningful classification.
MIN_BACKGROUND_PIXELS = 64

# Margin around the text bbox used to sample surrounding background pixels.
BACKGROUND_MARGIN_RATIO = 0.20
MIN_BACKGROUND_MARGIN_PX = 4

# Solid backgrounds have low color dispersion and few edges.
SOLID_MAX_COLOR_STD = 12.0
SOLID_MAX_EDGE_DENSITY = 0.04

# Gradient backgrounds show directional luminance change with moderate dispersion.
GRADIENT_MIN_CORRELATION = 0.72
GRADIENT_MAX_COLOR_STD = 35.0

# Textured backgrounds show local variation without photo-like complexity.
TEXTURE_MIN_COLOR_STD = 14.0
TEXTURE_MIN_EDGE_DENSITY = 0.06
TEXTURE_MAX_COLOR_STD = 42.0

# Photo-like backgrounds combine high dispersion and edge structure.
PHOTO_MIN_COLOR_STD = 38.0
PHOTO_MIN_EDGE_DENSITY = 0.10

# Complex backgrounds exceed simple heuristics but are not safely classifiable.
COMPLEX_MIN_COLOR_STD = 48.0

# Correlation is undefined when either axis has near-zero variance.
CORRELATION_MIN_STD = 1e-6

RESTORE_METHOD_BY_TYPE = {
    "solid": "fill",
    "gradient": "gradient",
    "texture": "inpaint",
    "photo": "inpaint",
    "complex": "advanced_inpaint",
    "unknown": "auto",
}


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def _clamp_bbox(bbox: BoundingBox, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(image_width, bbox.x + bbox.width)
    y2 = min(image_height, bbox.y + bbox.height)
    return x1, y1, x2, y2


def _hex_to_rgb(color: str | None) -> tuple[int, int, int] | None:
    if not color or not color.startswith("#") or len(color) != 7:
        return None
    try:
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    except ValueError:
        return None


def _extract_background_ring(
    image_array: np.ndarray,
    bbox: BoundingBox,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the outer patch and a boolean ring mask excluding the inner text bbox."""
    height, width = image_array.shape[:2]
    inner_x1, inner_y1, inner_x2, inner_y2 = _clamp_bbox(bbox, width, height)
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        return None

    margin = max(
        MIN_BACKGROUND_MARGIN_PX,
        int(min(inner_x2 - inner_x1, inner_y2 - inner_y1) * BACKGROUND_MARGIN_RATIO),
    )
    outer_x1 = max(0, inner_x1 - margin)
    outer_y1 = max(0, inner_y1 - margin)
    outer_x2 = min(width, inner_x2 + margin)
    outer_y2 = min(height, inner_y2 + margin)

    outer = image_array[outer_y1:outer_y2, outer_x1:outer_x2]
    if outer.size == 0:
        return None

    mask = np.ones((outer.shape[0], outer.shape[1]), dtype=bool)
    inner_left = inner_x1 - outer_x1
    inner_top = inner_y1 - outer_y1
    inner_right = inner_x2 - outer_x1
    inner_bottom = inner_y2 - outer_y1
    mask[inner_top:inner_bottom, inner_left:inner_right] = False

    if not mask.any():
        return None
    return outer, mask


def _ring_pixels(outer: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return outer[mask].reshape(-1, 3)


def _dominant_and_secondary_colors(pixels: np.ndarray) -> tuple[str, str | None]:
    flattened = pixels.reshape(-1, 3).astype(np.float32)
    dominant = np.median(flattened, axis=0).astype(int)
    distances = np.linalg.norm(flattened - dominant, axis=1)
    contrasting = flattened[distances > max(18.0, float(np.percentile(distances, 70)))]
    secondary: str | None = None
    if len(contrasting) > 0:
        secondary_rgb = np.median(contrasting, axis=0).astype(int)
        secondary = _rgb_to_hex(int(secondary_rgb[0]), int(secondary_rgb[1]), int(secondary_rgb[2]))
    return _rgb_to_hex(int(dominant[0]), int(dominant[1]), int(dominant[2])), secondary


def _color_std(pixels: np.ndarray) -> float:
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)
    return float(np.mean(np.std(lab.reshape(-1, 3).astype(np.float32), axis=0)))


def _edge_density(outer: np.ndarray, mask: np.ndarray) -> float:
    if outer.shape[0] < 3 or outer.shape[1] < 3:
        return 0.0
    gray = cv2.cvtColor(outer.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 120)
    ring_edges = edges[mask]
    if ring_edges.size == 0:
        return 0.0
    return float(np.count_nonzero(ring_edges)) / float(ring_edges.size)


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return |correlation(a, b)|, or 0.0 when correlation is undefined.

    Fallback 0.0 keeps degenerate patches from passing GRADIENT_MIN_CORRELATION:
    constant luminance along an axis is not a systematic gradient.
    """
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return 0.0

    a_values = np.asarray(a, dtype=np.float64)
    b_values = np.asarray(b, dtype=np.float64)
    if not np.all(np.isfinite(a_values)) or not np.all(np.isfinite(b_values)):
        return 0.0

    if np.std(a_values) < CORRELATION_MIN_STD or np.std(b_values) < CORRELATION_MIN_STD:
        return 0.0

    corr = np.corrcoef(a_values, b_values)[0, 1]
    if math.isnan(corr):
        return 0.0
    return abs(float(corr))


def _gradient_correlation(outer: np.ndarray, mask: np.ndarray) -> float:
    if outer.shape[0] < 4 or outer.shape[1] < 4:
        return 0.0
    gray = cv2.cvtColor(outer.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    column_means = []
    for col in range(gray.shape[1]):
        values = gray[mask[:, col], col]
        if values.size:
            column_means.append(float(values.mean()))
    row_means = []
    for row in range(gray.shape[0]):
        values = gray[row, mask[row, :]]
        if values.size:
            row_means.append(float(values.mean()))
    x = np.arange(len(column_means), dtype=np.float32)
    y = np.arange(len(row_means), dtype=np.float32)

    column_corr = (
        _safe_correlation(x, np.array(column_means, dtype=np.float32))
        if column_means
        else 0.0
    )
    row_corr = (
        _safe_correlation(y, np.array(row_means, dtype=np.float32))
        if row_means
        else 0.0
    )
    return max(column_corr, row_corr)


def _unique_color_ratio(pixels: np.ndarray) -> float:
    quantized = (pixels.reshape(-1, 3) // 16).astype(np.uint8)
    unique = np.unique(quantized, axis=0)
    return len(unique) / max(1, len(quantized))


def _complexity_score(color_std: float, edge_density: float, unique_color_ratio: float) -> float:
    score = (
        min(color_std / 60.0, 1.0) * 0.45
        + min(edge_density / 0.25, 1.0) * 0.30
        + min(unique_color_ratio / 0.35, 1.0) * 0.25
    )
    return round(min(max(score, 0.0), 1.0), 3)


def _classify_background(
    *,
    patch_area: int,
    color_std: float,
    gradient_correlation: float,
    edge_density: float,
    unique_color_ratio: float,
) -> tuple[str, float]:
    if patch_area < MIN_BACKGROUND_PIXELS:
        return "unknown", 0.25

    if color_std <= SOLID_MAX_COLOR_STD and edge_density <= SOLID_MAX_EDGE_DENSITY:
        confidence = min(0.95, 0.75 + (SOLID_MAX_COLOR_STD - color_std) / 40.0)
        return "solid", round(confidence, 3)

    if (
        gradient_correlation >= GRADIENT_MIN_CORRELATION
        and color_std <= GRADIENT_MAX_COLOR_STD
        and edge_density <= 0.12
    ):
        confidence = min(0.90, 0.55 + gradient_correlation * 0.35)
        return "gradient", round(confidence, 3)

    if color_std >= PHOTO_MIN_COLOR_STD and edge_density >= PHOTO_MIN_EDGE_DENSITY:
        confidence = min(0.88, 0.45 + min(color_std / 70.0, 1.0) * 0.35 + edge_density * 1.5)
        return "photo", round(confidence, 3)

    if (
        TEXTURE_MIN_COLOR_STD <= color_std <= TEXTURE_MAX_COLOR_STD
        and edge_density >= TEXTURE_MIN_EDGE_DENSITY
    ):
        confidence = min(0.82, 0.40 + edge_density * 1.8)
        return "texture", round(confidence, 3)

    if color_std >= COMPLEX_MIN_COLOR_STD or (
        color_std >= TEXTURE_MIN_COLOR_STD and edge_density >= PHOTO_MIN_EDGE_DENSITY
    ):
        confidence = min(0.70, 0.35 + _complexity_score(color_std, edge_density, unique_color_ratio) * 0.4)
        return "complex", round(confidence, 3)

    return "unknown", 0.30


def _inpaint_required(background_type: str) -> bool:
    if background_type in {"texture", "photo", "complex"}:
        return True
    return False


def analyze_background(
    image_array: np.ndarray,
    bbox: BoundingBox,
    *,
    dominant_color_hint: str | None = None,
) -> BackgroundInfo | None:
    """Classify the background around a text block using deterministic pixel metrics."""
    ring = _extract_background_ring(image_array, bbox)
    if ring is None:
        if dominant_color_hint:
            return BackgroundInfo(
                background_type="unknown",
                dominant_color=dominant_color_hint,
                classification_confidence=0.20,
                preferred_restore_method="auto",
            )
        return None

    outer, mask = ring
    ring_pixels = _ring_pixels(outer, mask)
    if len(ring_pixels) < MIN_BACKGROUND_PIXELS:
        if dominant_color_hint:
            return BackgroundInfo(
                background_type="unknown",
                dominant_color=dominant_color_hint,
                classification_confidence=0.20,
                preferred_restore_method="auto",
            )
        return None

    dominant_color, secondary_color = _dominant_and_secondary_colors(ring_pixels)
    if dominant_color_hint:
        dominant_color = dominant_color_hint

    color_std = _color_std(ring_pixels)
    edge_density = _edge_density(outer, mask)
    gradient_correlation = _gradient_correlation(outer, mask)
    unique_color_ratio = _unique_color_ratio(ring_pixels)
    complexity_score = _complexity_score(color_std, edge_density, unique_color_ratio)

    background_type, classification_confidence = _classify_background(
        patch_area=len(ring_pixels),
        color_std=color_std,
        gradient_correlation=gradient_correlation,
        edge_density=edge_density,
        unique_color_ratio=unique_color_ratio,
    )

    return BackgroundInfo(
        background_type=background_type,
        dominant_color=dominant_color,
        secondary_color=secondary_color,
        complexity_score=complexity_score,
        classification_confidence=classification_confidence,
        inpaint_required=_inpaint_required(background_type),
        preferred_restore_method=RESTORE_METHOD_BY_TYPE[background_type],
    )
