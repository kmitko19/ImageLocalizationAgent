"""Deterministic text mask and render geometry analysis for future Renderer."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from imageloc.config import PROJECT_ROOT
from imageloc.fusion.ocr_vision_fusion import FusedBlock
from imageloc.models.raw_blocks import BoundingBox, Point
from imageloc.models.text_block import BackgroundInfo, RenderGeometry, RenderHints, VisualStyle

MASK_FOREGROUND_VALUE = 255
MASK_BACKGROUND_VALUE = 0

# Minimum patch size for meaningful mask analysis.
MIN_PATCH_WIDTH_PX = 4
MIN_PATCH_HEIGHT_PX = 4

# Minimum bbox area before analysis is attempted.
MIN_BBOX_AREA_PX = 16

# Confidence below which mask PNG is not persisted.
MASK_SAVE_MIN_CONFIDENCE = 0.45

# Minimum confidence to attempt VisualStyle refinement from mask pixels.
VISUAL_REFINE_MIN_CONFIDENCE = 0.55

# Coverage plausibility bounds for typical text masks.
COVERAGE_MIN_PLAUSIBLE = 0.015
COVERAGE_MAX_PLAUSIBLE = 0.85
COVERAGE_SUSPICIOUS_HIGH = 0.92

# Connected-component filtering relative to estimated text height.
MIN_PRIMARY_COMPONENT_HEIGHT_RATIO = 0.06
MIN_DETACHED_MARK_HEIGHT_RATIO = 0.035
DETACHED_MARK_PROXIMITY_HEIGHT_RATIO = 0.45
MIN_COMPONENT_AREA_PX = 4
MIN_DETACHED_MARK_AREA_PX = 2
MIN_DETACHED_MARK_PROXIMITY_PX = 3

# Mask expansion bounds relative to estimated text height.
EXPANSION_HEIGHT_RATIO = 0.08
MIN_EXPANSION_PX = 1
MAX_EXPANSION_PX = 5

# Perspective detection thresholds.
PERSPECTIVE_PARALLEL_TOLERANCE = 0.08
PERSPECTIVE_SIDE_LENGTH_RATIO = 1.18
PERSPECTIVE_ANGLE_TOLERANCE_DEG = 8.0

# Gradient/stroke refinement thresholds.
GRADIENT_MIN_CORRELATION = 0.82
GRADIENT_MIN_COLOR_SPREAD = 18.0
STROKE_MIN_RING_RATIO = 0.08
STROKE_MIN_COLOR_DISTANCE = 22.0

DEFAULT_WRITING_DIRECTION = "horizontal"
_SAFE_BLOCK_ID_PATTERN = re.compile(r"[^\w\-]+")


@dataclass
class TextMaskAnalysis:
    """Internal mask analysis result — not part of the public JSON contract."""

    mask: np.ndarray
    mask_bbox: BoundingBox
    confidence: float
    method: str
    coverage_ratio: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class TextMaskBlockResult:
    """Combined mask, geometry, and optional visual refinements for one block."""

    mask_analysis: TextMaskAnalysis | None
    render_geometry: RenderGeometry | None
    visual: VisualStyle


def _hex_to_rgb(color: str | None) -> tuple[int, int, int] | None:
    if not color or not color.startswith("#") or len(color) != 7:
        return None
    try:
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    except ValueError:
        return None


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def _clamp_bbox(bbox: BoundingBox, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(image_width, bbox.x + bbox.width)
    y2 = min(image_height, bbox.y + bbox.height)
    return x1, y1, x2, y2


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _safe_block_filename(block_id: str) -> str:
    sanitized = _SAFE_BLOCK_ID_PATTERN.sub("_", block_id.strip())
    return sanitized or "block"


def _extract_patch(
    image_array: np.ndarray,
    bbox: BoundingBox,
) -> tuple[np.ndarray, BoundingBox] | None:
    height, width = image_array.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
    patch_width = x2 - x1
    patch_height = y2 - y1
    if patch_width < MIN_PATCH_WIDTH_PX or patch_height < MIN_PATCH_HEIGHT_PX:
        return None
    if patch_width * patch_height < MIN_BBOX_AREA_PX:
        return None
    patch = image_array[y1:y2, x1:x2].copy()
    mask_bbox = BoundingBox(x=x1, y=y1, width=patch_width, height=patch_height)
    return patch, mask_bbox


def _polygon_mask_in_patch(
    polygon: list[Point],
    mask_bbox: BoundingBox,
    patch_shape: tuple[int, int],
) -> np.ndarray | None:
    if len(polygon) < 3:
        return None
    local_points = np.array(
        [[point[0] - mask_bbox.x, point[1] - mask_bbox.y] for point in polygon],
        dtype=np.int32,
    )
    mask = np.zeros(patch_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [local_points], 255)
    if not mask.any():
        return None
    return mask.astype(bool)


def normalize_quad_order(points: list[Point]) -> list[Point]:
    """Order four points clockwise starting from the top-left corner."""
    if len(points) != 4:
        return list(points)

    pts = np.array(points, dtype=np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    rolled = np.roll(ordered, -start, axis=0)
    return [(int(x), int(y)) for x, y in rolled]


def _vector(from_point: Point, to_point: Point) -> np.ndarray:
    return np.array([to_point[0] - from_point[0], to_point[1] - from_point[1]], dtype=np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return vector
    return vector / length


def _parallelism_deviation(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = _normalize(vector_a)
    b = _normalize(vector_b)
    if float(np.linalg.norm(a)) < 1e-6 or float(np.linalg.norm(b)) < 1e-6:
        return 1.0
    cross = abs(float(a[0] * b[1] - a[1] * b[0]))
    return cross


def _angle_between(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    a = _normalize(vector_a)
    b = _normalize(vector_b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _is_degenerate_quad(points: list[Point]) -> bool:
    if len(points) != 4:
        return True
    unique = {tuple(point) for point in points}
    if len(unique) < 4:
        return True
    pts = np.array(points, dtype=np.float32)
    area = cv2.contourArea(pts)
    return area < 4.0


def _is_rotated_rectangle(points: list[Point]) -> bool:
    if _is_degenerate_quad(points):
        return False

    ordered = normalize_quad_order(points)
    sides = [
        _vector(ordered[0], ordered[1]),
        _vector(ordered[1], ordered[2]),
        _vector(ordered[2], ordered[3]),
        _vector(ordered[3], ordered[0]),
    ]
    if any(float(np.linalg.norm(side)) < 1e-3 for side in sides):
        return False

    if (
        _parallelism_deviation(sides[0], sides[2]) > PERSPECTIVE_PARALLEL_TOLERANCE
        or _parallelism_deviation(sides[1], sides[3]) > PERSPECTIVE_PARALLEL_TOLERANCE
    ):
        return False

    corner_angles = [
        _angle_between(sides[0], sides[1]),
        _angle_between(sides[1], sides[2]),
        _angle_between(sides[2], sides[3]),
        _angle_between(sides[3], sides[0]),
    ]
    return all(abs(angle - 90.0) <= PERSPECTIVE_ANGLE_TOLERANCE_DEG for angle in corner_angles)


def analyze_perspective_quad(polygon: list[Point]) -> tuple[list[Point] | None, bool]:
    """Return normalized quad and whether conservative perspective is detected."""
    if len(polygon) != 4:
        return None, False

    normalized = normalize_quad_order(polygon)
    if _is_degenerate_quad(normalized):
        return normalized, False

    if _is_rotated_rectangle(normalized):
        return normalized, False

    ordered = normalized
    sides = [
        _vector(ordered[0], ordered[1]),
        _vector(ordered[1], ordered[2]),
        _vector(ordered[2], ordered[3]),
        _vector(ordered[3], ordered[0]),
    ]
    parallel_score = max(
        _parallelism_deviation(sides[0], sides[2]),
        _parallelism_deviation(sides[1], sides[3]),
    )
    lengths = [float(np.linalg.norm(side)) for side in sides]
    length_ratio = max(lengths) / max(min(lengths), 1e-3)
    perspective_detected = (
        parallel_score > PERSPECTIVE_PARALLEL_TOLERANCE
        or length_ratio > PERSPECTIVE_SIDE_LENGTH_RATIO
    )
    return normalized, perspective_detected


def _background_complexity_penalty(background: BackgroundInfo | None) -> float:
    if background is None:
        return 0.85
    penalties = {
        "solid": 1.0,
        "gradient": 0.92,
        "texture": 0.75,
        "photo": 0.65,
        "complex": 0.55,
        "unknown": 0.70,
    }
    base = penalties.get(background.background_type, 0.70)
    return base * (0.7 + 0.3 * background.classification_confidence)


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return 0.0
    a_values = np.asarray(a, dtype=np.float64)
    b_values = np.asarray(b, dtype=np.float64)
    if not np.all(np.isfinite(a_values)) or not np.all(np.isfinite(b_values)):
        return 0.0
    if np.std(a_values) < 1e-6 or np.std(b_values) < 1e-6:
        return 0.0
    corr = np.corrcoef(a_values, b_values)[0, 1]
    if math.isnan(corr):
        return 0.0
    return abs(float(corr))


def _compute_separation_maps(
    patch: np.ndarray,
    *,
    background_rgb: tuple[int, int, int],
    text_rgb: tuple[int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    patch_lab = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(
        np.array([[background_rgb]], dtype=np.uint8),
        cv2.COLOR_RGB2LAB,
    )[0, 0].astype(np.float32)
    color_distance = np.linalg.norm(patch_lab - bg_lab, axis=2)

    gray = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg_luminance = float(
        0.299 * background_rgb[0] + 0.587 * background_rgb[1] + 0.114 * background_rgb[2]
    )
    luminance_distance = np.abs(gray - bg_luminance)

    if text_rgb is not None:
        text_luminance = 0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]
        if text_luminance >= bg_luminance:
            signed_luminance = gray - bg_luminance
        else:
            signed_luminance = bg_luminance - gray
        signed_luminance = np.clip(signed_luminance, 0.0, None)
    else:
        signed_luminance = luminance_distance

    combined = 0.65 * color_distance + 0.35 * signed_luminance
    return combined, color_distance


def _threshold_mask(
    score_map: np.ndarray,
    search_mask: np.ndarray,
    *,
    background: BackgroundInfo | None,
) -> tuple[np.ndarray, str]:
    values = score_map[search_mask]
    if values.size == 0:
        return np.zeros(score_map.shape, dtype=np.uint8), "empty_search_region"

    if background and background.background_type == "solid":
        percentile = 68.0
        method = "color_distance_solid"
    else:
        percentile = 72.0
        method = "color_distance_adaptive"

    threshold = max(12.0, float(np.percentile(values, percentile)))
    binary = np.zeros(score_map.shape, dtype=np.uint8)
    binary[(score_map >= threshold) & search_mask] = MASK_FOREGROUND_VALUE

    if binary.any():
        return binary, method

    otsu_source = np.clip(score_map, 0, 255).astype(np.uint8)
    otsu_source[~search_mask] = 0
    if np.count_nonzero(search_mask) > 0 and np.unique(otsu_source[search_mask]).size > 1:
        otsu_value, _ = cv2.threshold(
            otsu_source,
            0,
            MASK_FOREGROUND_VALUE,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        binary[(score_map >= otsu_value) & search_mask] = MASK_FOREGROUND_VALUE
        method = "otsu_fallback"
    return binary, method


def resolve_mask_path(mask_path: str | Path) -> Path:
    """Resolve a stored mask path the same way as source_image.path in Result JSON."""
    path = Path(mask_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _component_bbox_distance(stats_a: np.ndarray, stats_b: np.ndarray) -> float:
    ax1 = int(stats_a[cv2.CC_STAT_LEFT])
    ay1 = int(stats_a[cv2.CC_STAT_TOP])
    ax2 = ax1 + int(stats_a[cv2.CC_STAT_WIDTH])
    ay2 = ay1 + int(stats_a[cv2.CC_STAT_HEIGHT])
    bx1 = int(stats_b[cv2.CC_STAT_LEFT])
    by1 = int(stats_b[cv2.CC_STAT_TOP])
    bx2 = bx1 + int(stats_b[cv2.CC_STAT_WIDTH])
    by2 = by1 + int(stats_b[cv2.CC_STAT_HEIGHT])
    dx = max(0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0, max(by1 - ay2, ay1 - by2))
    return float(math.hypot(dx, dy))


def _remove_isolated_single_pixel_noise(mask: np.ndarray) -> np.ndarray:
    """Remove only 1-pixel components; preserve multi-pixel detached marks."""
    if not mask.any():
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = mask.copy()
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) == 1:
            cleaned[labels == label] = MASK_BACKGROUND_VALUE
    return cleaned


def _filter_connected_components(
    mask: np.ndarray,
    *,
    estimated_text_height: int,
) -> np.ndarray:
    """Drop isolated noise while preserving detached marks near primary glyphs."""
    if not mask.any():
        return mask

    min_primary_area = max(
        MIN_COMPONENT_AREA_PX,
        int((estimated_text_height * MIN_PRIMARY_COMPONENT_HEIGHT_RATIO) ** 2),
    )
    min_detached_area = max(
        MIN_DETACHED_MARK_AREA_PX,
        int((estimated_text_height * MIN_DETACHED_MARK_HEIGHT_RATIO) ** 2),
    )
    max_proximity = max(
        MIN_DETACHED_MARK_PROXIMITY_PX,
        int(round(estimated_text_height * DETACHED_MARK_PROXIMITY_HEIGHT_RATIO)),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    primary_labels = [
        label
        for label in range(1, num_labels)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_primary_area
    ]

    filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_primary_area:
            filtered[labels == label] = MASK_FOREGROUND_VALUE
            continue
        if area < min_detached_area or not primary_labels:
            continue

        for primary_label in primary_labels:
            if _component_bbox_distance(stats[label], stats[primary_label]) <= max_proximity:
                filtered[labels == label] = MASK_FOREGROUND_VALUE
                break

    return filtered


def _coverage_score(coverage_ratio: float) -> float:
    if coverage_ratio <= 0.0:
        return 0.0
    if coverage_ratio < COVERAGE_MIN_PLAUSIBLE:
        return max(0.0, coverage_ratio / COVERAGE_MIN_PLAUSIBLE * 0.4)
    if coverage_ratio > COVERAGE_SUSPICIOUS_HIGH:
        return 0.15
    if coverage_ratio > COVERAGE_MAX_PLAUSIBLE:
        return max(0.2, 1.0 - (coverage_ratio - COVERAGE_MAX_PLAUSIBLE) / 0.15)
    return 1.0


def _contrast_score(color_distance: np.ndarray, mask: np.ndarray, search_mask: np.ndarray) -> float:
    if not mask.any() or not search_mask.any():
        return 0.0
    foreground = color_distance[mask > 0]
    background = color_distance[search_mask & (mask == 0)]
    if foreground.size == 0 or background.size == 0:
        return 0.0
    separation = float(foreground.mean() - background.mean())
    return min(max(separation / 35.0, 0.0), 1.0)


def _polygon_agreement_score(mask: np.ndarray, polygon_mask: np.ndarray | None) -> float:
    if polygon_mask is None or not mask.any():
        return 0.5
    foreground = mask > 0
    inside = polygon_mask
    if not foreground.any():
        return 0.0
    overlap = float(np.count_nonzero(foreground & inside)) / float(np.count_nonzero(foreground))
    outside = float(np.count_nonzero(foreground & ~inside)) / float(np.count_nonzero(foreground))
    return min(max(overlap * (1.0 - outside), 0.0), 1.0)


def _component_plausibility_score(mask: np.ndarray, bbox_area: int) -> float:
    if not mask.any():
        return 0.0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_count = max(0, num_labels - 1)
    if component_count == 0:
        return 0.0
    largest = max(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels))
    if largest / max(bbox_area, 1) > COVERAGE_SUSPICIOUS_HIGH:
        return 0.2
    if component_count > 24:
        return 0.45
    return min(1.0, 0.55 + min(component_count, 8) * 0.05)


def _local_background_variance_penalty(
    patch: np.ndarray,
    search_mask: np.ndarray,
    mask: np.ndarray,
) -> float:
    background_pixels = patch[(search_mask) & (mask == 0)]
    if background_pixels.size == 0:
        return 1.0
    color_std = float(np.mean(np.std(background_pixels.astype(np.float32), axis=0)))
    if color_std <= 8.0:
        return 1.0
    if color_std >= 40.0:
        return 0.55
    return 1.0 - ((color_std - 8.0) / 32.0) * 0.45


def _compute_mask_confidence(
    *,
    mask: np.ndarray,
    color_distance: np.ndarray,
    search_mask: np.ndarray,
    polygon_mask: np.ndarray | None,
    coverage_ratio: float,
    bbox_area: int,
    background: BackgroundInfo | None,
    patch: np.ndarray,
) -> float:
    base_confidence = (
        _contrast_score(color_distance, mask, search_mask) * 0.30
        + _coverage_score(coverage_ratio) * 0.25
        + _component_plausibility_score(mask, bbox_area) * 0.20
        + _polygon_agreement_score(mask, polygon_mask) * 0.15
        + 0.10
    )
    confidence = (
        base_confidence
        * _background_complexity_penalty(background)
        * _local_background_variance_penalty(patch, search_mask, mask)
    )
    return round(min(max(confidence, 0.0), 1.0), 3)


def analyze_text_mask(
    patch: np.ndarray,
    mask_bbox: BoundingBox,
    *,
    polygon: list[Point],
    visual: VisualStyle,
    background: BackgroundInfo | None,
    estimated_text_height: int,
) -> TextMaskAnalysis:
    """Build a conservative raw text mask inside one bbox patch."""
    warnings: list[str] = []
    patch_height, patch_width = patch.shape[:2]
    bbox_area = max(1, patch_width * patch_height)

    background_rgb = _hex_to_rgb(visual.background_color) or (255, 255, 255)
    text_rgb = _hex_to_rgb(visual.text_color)
    polygon_mask = _polygon_mask_in_patch(polygon, mask_bbox, (patch_height, patch_width))
    search_mask = polygon_mask if polygon_mask is not None else np.ones((patch_height, patch_width), dtype=bool)

    score_map, color_distance = _compute_separation_maps(
        patch,
        background_rgb=background_rgb,
        text_rgb=text_rgb,
    )
    raw_mask, method = _threshold_mask(score_map, search_mask, background=background)
    raw_mask = _remove_isolated_single_pixel_noise(raw_mask)
    raw_mask = _filter_connected_components(raw_mask, estimated_text_height=estimated_text_height)

    coverage_ratio = float(np.count_nonzero(raw_mask)) / float(bbox_area)
    if coverage_ratio >= COVERAGE_SUSPICIOUS_HIGH:
        warnings.append("coverage suspiciously high")
    if coverage_ratio <= 0.0:
        warnings.append("no foreground pixels detected")

    confidence = _compute_mask_confidence(
        mask=raw_mask,
        color_distance=color_distance,
        search_mask=search_mask,
        polygon_mask=polygon_mask,
        coverage_ratio=coverage_ratio,
        bbox_area=bbox_area,
        background=background,
        patch=patch,
    )

    return TextMaskAnalysis(
        mask=raw_mask,
        mask_bbox=mask_bbox,
        confidence=confidence,
        method=method,
        coverage_ratio=round(coverage_ratio, 4),
        warnings=warnings,
    )


def expand_text_mask(
    mask: np.ndarray,
    *,
    estimated_text_height: int,
) -> np.ndarray:
    """Expand a raw text mask for future inpainting without modifying the raw mask."""
    if mask.size == 0 or not mask.any():
        return mask.copy()
    dilation_px = int(round(estimated_text_height * EXPANSION_HEIGHT_RATIO))
    dilation_px = max(MIN_EXPANSION_PX, min(MAX_EXPANSION_PX, dilation_px))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_px + 1, 2 * dilation_px + 1),
    )
    expanded = cv2.dilate(mask, kernel, iterations=1)
    return expanded


def save_text_mask_png(
    mask: np.ndarray,
    *,
    output_dir: Path,
    source_image_id: str,
    block_id: str,
) -> str:
    """Persist mask PNG under output_dir/source_image_id/render_assets atomically."""
    safe_id = _safe_block_filename(block_id)
    assets_dir = output_dir / source_image_id / "render_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    target_path = assets_dir / f"{safe_id}_mask.png"
    temp_path = target_path.with_suffix(".tmp.png")

    Image.fromarray(mask).save(temp_path, format="PNG")
    temp_path.replace(target_path)
    return _relative_to_project(target_path)


def load_text_mask_png(mask_path: str | Path) -> np.ndarray:
    with Image.open(resolve_mask_path(mask_path)) as image:
        return np.array(image.convert("L"))


def _refine_visual_style_with_mask(
    visual: VisualStyle,
    *,
    patch: np.ndarray,
    mask: np.ndarray,
    confidence: float,
) -> VisualStyle:
    if confidence < VISUAL_REFINE_MIN_CONFIDENCE or not mask.any():
        return visual

    updates: dict[str, object] = {}
    foreground = mask > 0
    if np.count_nonzero(foreground) < 24:
        return visual

    patch_lab = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_lab = patch_lab[foreground]
    l_channel = fg_lab[:, 0]
    a_channel = fg_lab[:, 1]
    b_channel = fg_lab[:, 2]
    y_coords, x_coords = np.where(foreground)

    row_corr = _safe_correlation(y_coords.astype(np.float32), l_channel)
    col_corr = _safe_correlation(x_coords.astype(np.float32), l_channel)
    gradient_corr = max(row_corr, col_corr)
    color_spread = float(np.std(l_channel) + np.std(a_channel) + np.std(b_channel))

    if gradient_corr >= GRADIENT_MIN_CORRELATION and color_spread >= GRADIENT_MIN_COLOR_SPREAD:
        start_rgb = np.median(fg_lab[l_channel <= np.percentile(l_channel, 35)], axis=0)
        end_rgb = np.median(fg_lab[l_channel >= np.percentile(l_channel, 65)], axis=0)
        start = cv2.cvtColor(np.array([[start_rgb]], dtype=np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
        end = cv2.cvtColor(np.array([[end_rgb]], dtype=np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
        updates.update(
            {
                "fill_type": "gradient",
                "gradient_start_color": _rgb_to_hex(int(start[0]), int(start[1]), int(start[2])),
                "gradient_end_color": _rgb_to_hex(int(end[0]), int(end[1]), int(end[2])),
                "gradient_angle_deg": 90.0 if row_corr >= col_corr else 0.0,
                "gradient_confidence": round(min(0.85, 0.45 + gradient_corr * 0.4), 3),
            }
        )

    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    core = cv2.erode(mask, core_kernel, iterations=1)
    if core.any():
        ring = (cv2.dilate(core, core_kernel, iterations=1) > 0) & ~(core > 0)
        if np.count_nonzero(ring) >= max(12, int(np.count_nonzero(core) * STROKE_MIN_RING_RATIO)):
            core_rgb = patch[core > 0].astype(np.float32)
            ring_rgb = patch[ring].astype(np.float32)
            core_median = np.median(core_rgb, axis=0)
            ring_median = np.median(ring_rgb, axis=0)
            stroke_distance = float(np.linalg.norm(core_median - ring_median))
            if stroke_distance >= STROKE_MIN_COLOR_DISTANCE:
                stroke = np.clip(ring_median, 0, 255).astype(int)
                updates.update(
                    {
                        "stroke_color": _rgb_to_hex(int(stroke[0]), int(stroke[1]), int(stroke[2])),
                        "stroke_width_px": 1.0,
                        "stroke_confidence": round(min(0.75, 0.35 + stroke_distance / 80.0), 3),
                    }
                )

    if not updates:
        return visual
    return visual.model_copy(update=updates)


def build_render_geometry_from_analysis(
    *,
    visual: VisualStyle,
    render_hints: RenderHints,
    mask_analysis: TextMaskAnalysis | None,
    mask_path: str | None,
    perspective_quad: list[Point] | None,
    perspective_detected: bool,
) -> RenderGeometry | None:
    has_rotation = visual.rotation_deg != 0.0
    has_non_default_direction = render_hints.text_direction != DEFAULT_WRITING_DIRECTION
    has_mask = mask_analysis is not None and mask_analysis.confidence > 0.0
    has_perspective = perspective_quad is not None

    if not any([has_rotation, has_non_default_direction, has_mask, has_perspective]):
        return None

    return RenderGeometry(
        baseline_angle_deg=visual.rotation_deg if has_rotation else None,
        perspective_quad=perspective_quad,
        perspective_detected=perspective_detected,
        mask_bbox=mask_analysis.mask_bbox if mask_analysis else None,
        mask_path=mask_path,
        mask_confidence=mask_analysis.confidence if mask_analysis else 0.0,
        writing_direction=render_hints.text_direction,
    )


def analyze_text_mask_for_block(
    image: Image.Image,
    block: FusedBlock,
    *,
    visual: VisualStyle,
    render_hints: RenderHints,
    background: BackgroundInfo | None,
    source_image_id: str,
    output_dir: Path,
) -> TextMaskBlockResult:
    """Analyze text mask and render geometry for one fused block."""
    image_array = np.array(image.convert("RGB"))
    patch_data = _extract_patch(image_array, block.bbox)
    perspective_quad, perspective_detected = analyze_perspective_quad(block.polygon)

    if patch_data is None:
        geometry = build_render_geometry_from_analysis(
            visual=visual,
            render_hints=render_hints,
            mask_analysis=None,
            mask_path=None,
            perspective_quad=perspective_quad,
            perspective_detected=perspective_detected,
        )
        return TextMaskBlockResult(mask_analysis=None, render_geometry=geometry, visual=visual)

    patch, mask_bbox = patch_data
    estimated_text_height = visual.estimated_text_height_px or max(1, block.bbox.height)
    mask_analysis = analyze_text_mask(
        patch,
        mask_bbox,
        polygon=block.polygon,
        visual=visual,
        background=background,
        estimated_text_height=estimated_text_height,
    )

    mask_path: str | None = None
    if mask_analysis.confidence >= MASK_SAVE_MIN_CONFIDENCE and mask_analysis.mask.any():
        mask_path = save_text_mask_png(
            mask_analysis.mask,
            output_dir=output_dir,
            source_image_id=source_image_id,
            block_id=block.id,
        )

    refined_visual = _refine_visual_style_with_mask(
        visual,
        patch=patch,
        mask=mask_analysis.mask,
        confidence=mask_analysis.confidence,
    )
    geometry = build_render_geometry_from_analysis(
        visual=visual,
        render_hints=render_hints,
        mask_analysis=mask_analysis,
        mask_path=mask_path,
        perspective_quad=perspective_quad,
        perspective_detected=perspective_detected,
    )
    return TextMaskBlockResult(
        mask_analysis=mask_analysis,
        render_geometry=geometry,
        visual=refined_visual,
    )
