"""Deterministic background restoration for Renderer R0.3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from imageloc.config import PROJECT_ROOT
from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import BackgroundInfo, TextBlock
from imageloc.utils.text_mask_analyzer import (
    expand_text_mask,
    load_text_mask_png,
    resolve_mask_path,
)

MASK_FOREGROUND_VALUE = 255

# Minimum mask confidence required before mutating the image.
RESTORE_MIN_MASK_CONFIDENCE = 0.45

# Local patch padding around the mask bbox.
PATCH_PADDING_HEIGHT_RATIO = 0.35
MIN_PATCH_PADDING_PX = 6
MAX_PATCH_PADDING_PX = 32

# OpenCV inpaint radius bounds.
INPAINT_RADIUS_HEIGHT_RATIO = 0.12
MIN_INPAINT_RADIUS_PX = 2
MAX_INPAINT_RADIUS_PX = 8

# Minimum border samples required for gradient plane fitting.
GRADIENT_MIN_BORDER_SAMPLES = 12
GRADIENT_FALLBACK_MAX_BORDER_STD = 28.0

# Solid fill samples a ring outside the expanded mask, not the whole patch border.
SOLID_SAMPLE_RING_KERNEL_PX = 7
SOLID_MIN_RING_SAMPLES = 6
SOLID_FOREGROUND_EXCLUSION_DISTANCE = 22.0
SOLID_HINT_MAX_DISTANCE = 55.0

INPAINT_METHOD_STANDARD = cv2.INPAINT_TELEA
INPAINT_METHOD_ADVANCED = cv2.INPAINT_NS

CLEAN_BACKGROUND_FILENAME = "clean_background.png"


@dataclass
class BackgroundRestoreResult:
    """Internal result of one background restoration attempt."""

    image: Image.Image
    success: bool
    method: str
    confidence: float
    restored_bbox: BoundingBox | None = None
    warnings: list[str] = field(default_factory=list)


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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


def _estimated_text_height(block: TextBlock) -> int:
    if block.visual.estimated_text_height_px:
        return max(1, block.visual.estimated_text_height_px)
    if block.render_geometry and block.render_geometry.mask_bbox:
        return max(1, block.render_geometry.mask_bbox.height)
    return max(1, block.bbox.height)


def _patch_padding_px(text_height: int) -> int:
    padding = int(round(text_height * PATCH_PADDING_HEIGHT_RATIO))
    return max(MIN_PATCH_PADDING_PX, min(MAX_PATCH_PADDING_PX, padding))


def _inpaint_radius_px(text_height: int, *, advanced: bool = False) -> float:
    radius = text_height * INPAINT_RADIUS_HEIGHT_RATIO
    radius = max(MIN_INPAINT_RADIUS_PX, min(MAX_INPAINT_RADIUS_PX, radius))
    if advanced:
        radius = min(MAX_INPAINT_RADIUS_PX, radius + 1.0)
    return float(radius)


def select_restore_method(
    background: BackgroundInfo | None,
    *,
    mask_confidence: float,
    local_border_std: float | None = None,
) -> str:
    """Choose a deterministic restore method from background metadata."""
    if mask_confidence < RESTORE_MIN_MASK_CONFIDENCE:
        return "skip"

    if background is None:
        if local_border_std is not None and local_border_std >= GRADIENT_FALLBACK_MAX_BORDER_STD:
            return "inpaint"
        return "skip"

    preferred = background.preferred_restore_method
    background_type = background.background_type

    if preferred == "fill" or background_type == "solid":
        return "fill"
    if preferred == "gradient" or background_type == "gradient":
        return "gradient"
    if preferred == "advanced_inpaint" or background_type == "complex":
        return "advanced_inpaint"
    if preferred == "inpaint" or background_type in {"texture", "photo"}:
        return "inpaint"
    if background_type == "unknown":
        if (
            mask_confidence >= RESTORE_MIN_MASK_CONFIDENCE
            and local_border_std is not None
            and local_border_std >= GRADIENT_FALLBACK_MAX_BORDER_STD
        ):
            return "inpaint"
        return "skip"
    return "skip"


def _image_to_arrays(image: Image.Image) -> tuple[np.ndarray, np.ndarray | None, str]:
    if image.mode == "RGBA":
        rgba = np.array(image)
        return rgba[:, :, :3].copy(), rgba[:, :, 3].copy(), "RGBA"
    rgb = np.array(image.convert("RGB"))
    return rgb, None, "RGB"


def _arrays_to_image(rgb: np.ndarray, alpha: np.ndarray | None, mode: str) -> Image.Image:
    if mode == "RGBA" and alpha is not None:
        rgba = np.dstack([rgb, alpha])
        return Image.fromarray(rgba, mode="RGBA")
    return Image.fromarray(rgb, mode="RGB")


def _extract_patch(
    rgb: np.ndarray,
    mask_bbox: BoundingBox,
    *,
    padding: int,
) -> tuple[np.ndarray, int, int, BoundingBox]:
    height, width = rgb.shape[:2]
    x1 = max(0, mask_bbox.x - padding)
    y1 = max(0, mask_bbox.y - padding)
    x2 = min(width, mask_bbox.x + mask_bbox.width + padding)
    y2 = min(height, mask_bbox.y + mask_bbox.height + padding)
    patch = rgb[y1:y2, x1:x2].copy()
    patch_bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
    return patch, x1, y1, patch_bbox


def _mask_to_patch_coordinates(
    mask: np.ndarray,
    mask_bbox: BoundingBox,
    patch_bbox: BoundingBox,
) -> np.ndarray | None:
    if mask.shape != (mask_bbox.height, mask_bbox.width):
        return None
    patch_mask = np.zeros((patch_bbox.height, patch_bbox.width), dtype=np.uint8)
    offset_x = mask_bbox.x - patch_bbox.x
    offset_y = mask_bbox.y - patch_bbox.y
    y1 = max(0, offset_y)
    x1 = max(0, offset_x)
    y2 = min(patch_bbox.height, offset_y + mask.shape[0])
    x2 = min(patch_bbox.width, offset_x + mask.shape[1])
    src_y1 = max(0, -offset_y)
    src_x1 = max(0, -offset_x)
    src_y2 = src_y1 + (y2 - y1)
    src_x2 = src_x1 + (x2 - x1)
    if y2 <= y1 or x2 <= x1:
        return None
    patch_mask[y1:y2, x1:x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return patch_mask


def _border_sample_stats(patch_rgb: np.ndarray, inpaint_mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate local background variance from pixels outside the expanded mask."""
    background_mask = inpaint_mask == 0
    if not background_mask.any():
        return np.array([255, 255, 255], dtype=np.float32), 0.0
    pixels = patch_rgb[background_mask].astype(np.float32)
    color = np.median(pixels, axis=0)
    color_std = float(np.mean(np.std(pixels, axis=0)))
    return color, color_std


def _solid_sample_ring_mask(inpaint_mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (SOLID_SAMPLE_RING_KERNEL_PX, SOLID_SAMPLE_RING_KERNEL_PX),
    )
    dilated = cv2.dilate(inpaint_mask, kernel, iterations=1)
    ring = (dilated > 0) & (inpaint_mask == 0)
    if np.count_nonzero(ring) >= SOLID_MIN_RING_SAMPLES:
        return ring

    wider = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.dilate(inpaint_mask, wider, iterations=1)
    return (dilated > 0) & (inpaint_mask == 0)


def sample_solid_fill_color(
    patch_rgb: np.ndarray,
    inpaint_mask: np.ndarray,
    raw_mask: np.ndarray,
    *,
    hint_rgb: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Estimate a robust solid fill color from a ring outside the expanded mask."""
    ring_mask = _solid_sample_ring_mask(inpaint_mask)
    ring_mask &= raw_mask == 0
    candidates = patch_rgb[ring_mask].astype(np.float32)
    if candidates.size == 0:
        candidates = patch_rgb[inpaint_mask == 0].astype(np.float32)

    if raw_mask.any():
        foreground_median = np.median(patch_rgb[raw_mask > 0].astype(np.float32), axis=0)
        distances_from_text = np.linalg.norm(candidates - foreground_median, axis=1)
        filtered = candidates[distances_from_text >= SOLID_FOREGROUND_EXCLUSION_DISTANCE]
        if len(filtered) >= SOLID_MIN_RING_SAMPLES:
            candidates = filtered

    if hint_rgb is not None and len(candidates) > 0:
        hint = np.array(hint_rgb, dtype=np.float32)
        hint_distances = np.linalg.norm(candidates - hint, axis=1)
        hinted = candidates[hint_distances <= SOLID_HINT_MAX_DISTANCE]
        if len(hinted) >= SOLID_MIN_RING_SAMPLES:
            candidates = hinted

    if len(candidates) == 0:
        if hint_rgb is not None:
            return np.array(hint_rgb, dtype=np.float32)
        return np.array([255, 255, 255], dtype=np.float32)

    return np.median(candidates, axis=0)


def _restore_solid_fill(
    patch_rgb: np.ndarray,
    inpaint_mask: np.ndarray,
    raw_mask: np.ndarray,
    *,
    hint_rgb: tuple[int, int, int] | None,
) -> np.ndarray:
    border_color = sample_solid_fill_color(
        patch_rgb,
        inpaint_mask,
        raw_mask,
        hint_rgb=hint_rgb,
    )
    if hint_rgb is not None and not np.isfinite(border_color).all():
        border_color = np.array(hint_rgb, dtype=np.float32)
    restored = patch_rgb.copy()
    restored[inpaint_mask > 0] = np.clip(border_color, 0, 255).astype(np.uint8)
    return restored


def _fit_channel_plane(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, float] | None:
    if len(xs) < GRADIENT_MIN_BORDER_SAMPLES:
        return None
    design = np.column_stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(xs))])
    coeffs, _, rank, _ = np.linalg.lstsq(design, values.astype(np.float64), rcond=None)
    if rank < 3:
        return None
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def _restore_gradient_fill(
    patch_rgb: np.ndarray,
    inpaint_mask: np.ndarray,
    raw_mask: np.ndarray,
    *,
    hint_rgb: tuple[int, int, int] | None,
) -> tuple[np.ndarray, bool]:
    background_mask = inpaint_mask == 0
    if np.count_nonzero(background_mask) < GRADIENT_MIN_BORDER_SAMPLES:
        return _restore_solid_fill(patch_rgb, inpaint_mask, raw_mask, hint_rgb=hint_rgb), False

    ys, xs = np.where(background_mask)
    restored = patch_rgb.copy()
    fitted_channels = 0
    for channel in range(3):
        plane = _fit_channel_plane(xs, ys, patch_rgb[ys, xs, channel])
        if plane is None:
            continue
        slope_x, slope_y, intercept = plane
        fg_y, fg_x = np.where(inpaint_mask > 0)
        predicted = slope_x * fg_x + slope_y * fg_y + intercept
        restored[fg_y, fg_x, channel] = np.clip(predicted, 0, 255).astype(np.uint8)
        fitted_channels += 1

    if fitted_channels < 3:
        return _restore_solid_fill(patch_rgb, inpaint_mask, raw_mask, hint_rgb=hint_rgb), False
    return restored, True


def _restore_inpaint(
    patch_rgb: np.ndarray,
    inpaint_mask: np.ndarray,
    *,
    text_height: int,
    advanced: bool = False,
) -> np.ndarray:
    radius = _inpaint_radius_px(text_height, advanced=advanced)
    method = INPAINT_METHOD_ADVANCED if advanced else INPAINT_METHOD_STANDARD
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    restored_bgr = cv2.inpaint(patch_bgr, inpaint_mask, radius, method)
    return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)


def _apply_patch(
    rgb: np.ndarray,
    patch_rgb: np.ndarray,
    *,
    patch_x: int,
    patch_y: int,
    inpaint_mask: np.ndarray,
) -> None:
    region = rgb[patch_y : patch_y + patch_rgb.shape[0], patch_x : patch_x + patch_rgb.shape[1]]
    mask_bool = inpaint_mask > 0
    region[mask_bool] = patch_rgb[mask_bool]


def _load_block_mask(block: TextBlock, mask: np.ndarray | None) -> tuple[np.ndarray, BoundingBox] | None:
    if mask is not None:
        mask_bbox = block.render_geometry.mask_bbox if block.render_geometry and block.render_geometry.mask_bbox else block.bbox
        if mask.shape != (mask_bbox.height, mask_bbox.width):
            return None
        return mask, mask_bbox

    geometry = block.render_geometry
    if geometry is None or geometry.mask_path is None or geometry.mask_bbox is None:
        return None

    loaded = load_text_mask_png(geometry.mask_path)
    if loaded.shape != (geometry.mask_bbox.height, geometry.mask_bbox.width):
        return None
    return loaded, geometry.mask_bbox


def _restore_confidence(method: str, mask_confidence: float, *, gradient_fit_ok: bool = True) -> float:
    method_weight = {
        "fill": 0.90,
        "gradient": 0.82 if gradient_fit_ok else 0.65,
        "inpaint": 0.70,
        "advanced_inpaint": 0.62,
        "skip": 0.0,
    }
    return round(min(1.0, mask_confidence * method_weight.get(method, 0.5)), 3)


def restore_text_background(
    image: Image.Image,
    block: TextBlock,
    *,
    mask: np.ndarray | None = None,
    expanded_mask: np.ndarray | None = None,
) -> BackgroundRestoreResult:
    """Restore background under one text block using its trusted mask."""
    mask_confidence = block.render_geometry.mask_confidence if block.render_geometry else 0.0
    if mask_confidence < RESTORE_MIN_MASK_CONFIDENCE:
        return BackgroundRestoreResult(
            image=image.copy(),
            success=False,
            method="skip",
            confidence=0.0,
            warnings=["mask confidence below restore threshold"],
        )

    mask_data = _load_block_mask(block, mask)
    if mask_data is None:
        return BackgroundRestoreResult(
            image=image.copy(),
            success=False,
            method="skip",
            confidence=0.0,
            warnings=["trusted mask unavailable"],
        )

    raw_mask, mask_bbox = mask_data
    if not raw_mask.any():
        return BackgroundRestoreResult(
            image=image.copy(),
            success=False,
            method="skip",
            confidence=0.0,
            warnings=["empty mask"],
        )

    text_height = _estimated_text_height(block)
    inpaint_mask = (
        expanded_mask
        if expanded_mask is not None
        else expand_text_mask(raw_mask, estimated_text_height=text_height)
    )
    if expanded_mask is not None and expanded_mask.shape != raw_mask.shape:
        return BackgroundRestoreResult(
            image=image.copy(),
            success=False,
            method="skip",
            confidence=0.0,
            warnings=["expanded mask dimensions mismatch"],
        )

    rgb, alpha, mode = _image_to_arrays(image)
    padding = _patch_padding_px(text_height)
    patch_rgb, patch_x, patch_y, patch_bbox = _extract_patch(rgb, mask_bbox, padding=padding)
    patch_inpaint_mask = _mask_to_patch_coordinates(inpaint_mask, mask_bbox, patch_bbox)
    patch_raw_mask = _mask_to_patch_coordinates(raw_mask, mask_bbox, patch_bbox)
    if patch_inpaint_mask is None or not patch_inpaint_mask.any():
        return BackgroundRestoreResult(
            image=image.copy(),
            success=False,
            method="skip",
            confidence=0.0,
            warnings=["mask could not be mapped to image patch"],
        )
    if patch_raw_mask is None:
        patch_raw_mask = np.zeros_like(patch_inpaint_mask)

    _, border_std = _border_sample_stats(patch_rgb, patch_inpaint_mask)
    method = select_restore_method(
        block.background,
        mask_confidence=mask_confidence,
        local_border_std=border_std,
    )
    if method == "skip":
        return BackgroundRestoreResult(
            image=image.copy(),
            success=False,
            method="skip",
            confidence=0.0,
            warnings=["restore method selection skipped"],
        )

    hint_rgb = _hex_to_rgb(block.background.dominant_color if block.background else None)
    gradient_fit_ok = True
    if method == "fill":
        restored_patch = _restore_solid_fill(
            patch_rgb,
            patch_inpaint_mask,
            patch_raw_mask,
            hint_rgb=hint_rgb,
        )
    elif method == "gradient":
        restored_patch, gradient_fit_ok = _restore_gradient_fill(
            patch_rgb,
            patch_inpaint_mask,
            patch_raw_mask,
            hint_rgb=hint_rgb,
        )
        if not gradient_fit_ok and block.background and block.background.background_type == "gradient":
            method = "inpaint"
            restored_patch = _restore_inpaint(
                patch_rgb,
                patch_inpaint_mask,
                text_height=text_height,
                advanced=False,
            )
    elif method == "advanced_inpaint":
        restored_patch = _restore_inpaint(
            patch_rgb,
            patch_inpaint_mask,
            text_height=text_height,
            advanced=True,
        )
    else:
        restored_patch = _restore_inpaint(
            patch_rgb,
            patch_inpaint_mask,
            text_height=text_height,
            advanced=False,
        )

    _apply_patch(rgb, restored_patch, patch_x=patch_x, patch_y=patch_y, inpaint_mask=patch_inpaint_mask)
    restored_image = _arrays_to_image(rgb, alpha, mode)
    confidence = _restore_confidence(method, mask_confidence, gradient_fit_ok=gradient_fit_ok)
    return BackgroundRestoreResult(
        image=restored_image,
        success=True,
        method=method,
        confidence=confidence,
        restored_bbox=mask_bbox,
        warnings=[] if gradient_fit_ok or method != "gradient" else ["gradient plane fallback used"],
    )


def restore_backgrounds(
    image: Image.Image,
    blocks: list[TextBlock],
    *,
    masks: dict[str, np.ndarray] | None = None,
) -> tuple[Image.Image, list[BackgroundRestoreResult]]:
    """Restore multiple text blocks sequentially on one working image copy."""
    working = image.copy()
    results: list[BackgroundRestoreResult] = []
    masks = masks or {}

    for block in blocks:
        result = restore_text_background(working, block, mask=masks.get(block.id))
        results.append(result)
        if result.success:
            working = result.image

    return working, results


def save_clean_background(
    image: Image.Image,
    *,
    output_dir: Path,
    source_image_id: str,
) -> str:
    """Persist a clean background image under render_assets atomically."""
    assets_dir = output_dir / source_image_id / "render_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    target_path = assets_dir / CLEAN_BACKGROUND_FILENAME
    temp_path = target_path.with_suffix(".tmp.png")

    if image.mode == "RGBA":
        image.save(temp_path, format="PNG")
    else:
        image.convert("RGB").save(temp_path, format="PNG")
    temp_path.replace(target_path)
    return _relative_to_project(target_path)


def resolve_clean_background_path(clean_background_path: str | Path) -> Path:
    """Resolve a stored clean background path relative to PROJECT_ROOT."""
    path = Path(clean_background_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
