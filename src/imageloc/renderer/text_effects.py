"""Geometry transforms and visual effects for Renderer R0.6."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import TextBlock

SHADOW_MIN_CONFIDENCE = 0.5
GRADIENT_MIN_CONFIDENCE = 0.5
PERSPECTIVE_MIN_AREA_PX2 = 64.0
MAX_WARP_COORD = 100_000.0
ROTATION_EPSILON_DEG = 0.01

# Effect pipeline:
# layout alpha mask -> shadow -> gradient/solid fill -> stroke -> rotation -> perspective -> composite


@dataclass
class TextLayoutContext:
    """Deterministic glyph placement used for alpha, stroke, and effects."""

    font_path: str
    font_size_px: int
    lines: list[str]
    content_box: BoundingBox
    letter_spacing_px: float
    line_height_px: int
    y_start: float
    horizontal: str
    block_height: float


@dataclass
class EffectApplicationResult:
    """Result of geometry/effects processing on a local text layer."""

    layer: Image.Image
    applied_rotation: float = 0.0
    perspective_applied: bool = False
    shadow_applied: bool = False
    gradient_applied: bool = False
    paste_offset: tuple[int, int] = (0, 0)
    warnings: list[str] = field(default_factory=list)


def _hex_to_rgb(color: str | None) -> tuple[int, int, int] | None:
    if not color or not color.startswith("#") or len(color) != 7:
        return None
    try:
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    except ValueError:
        return None


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _is_convex_quad(points: list[tuple[float, float]]) -> bool:
    if len(points) != 4:
        return False

    signs: list[int] = []
    for index in range(4):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % 4]
        x3, y3 = points[(index + 2) % 4]
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if abs(cross) < 1e-6:
            continue
        signs.append(1 if cross > 0 else -1)
    return len(signs) >= 2 and len(set(signs)) == 1


def validate_perspective_quad(
    points: list[tuple[float, float]],
    *,
    max_width: int,
    max_height: int,
) -> tuple[bool, str | None]:
    """Validate a 4-point target quadrilateral for perspective warp."""
    if len(points) != 4:
        return False, "perspective quad must contain exactly 4 points"

    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            return False, "perspective quad contains non-finite coordinates"
        if abs(x) > MAX_WARP_COORD or abs(y) > MAX_WARP_COORD:
            return False, "perspective quad coordinates out of bounds"
        if x < -max_width or y < -max_height or x > max_width * 2 or y > max_height * 2:
            return False, "perspective quad coordinates outside local render region"

    if _polygon_area(points) < PERSPECTIVE_MIN_AREA_PX2:
        return False, "perspective quad area too small"

    if not _is_convex_quad(points):
        return False, "perspective quad is not convex"

    return True, None


def _local_quad_from_block(block: TextBlock) -> list[tuple[float, float]]:
    bbox = block.bbox
    geometry = block.render_geometry
    if geometry and geometry.perspective_quad and len(geometry.perspective_quad) == 4:
        return [(float(x - bbox.x), float(y - bbox.y)) for x, y in geometry.perspective_quad]

    if len(block.polygon) == 4:
        return [(float(x - bbox.x), float(y - bbox.y)) for x, y in block.polygon]

    return [
        (0.0, 0.0),
        (float(bbox.width), 0.0),
        (float(bbox.width), float(bbox.height)),
        (0.0, float(bbox.height)),
    ]


def resolve_perspective_target_quad(
    block: TextBlock,
) -> tuple[list[tuple[float, float]] | None, list[str]]:
    """Resolve and validate a local perspective target quad for a block."""
    warnings: list[str] = []
    geometry = block.render_geometry
    if geometry is None or not geometry.perspective_detected:
        return None, warnings

    points = _local_quad_from_block(block)
    valid, reason = validate_perspective_quad(points, max_width=block.bbox.width, max_height=block.bbox.height)
    if not valid:
        warnings.append(f"perspective fallback: {reason}")
        return None, warnings
    return points, warnings


def _measure_spaced_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    font: ImageFont.FreeTypeFont,
    *,
    letter_spacing_px: float,
) -> float:
    if letter_spacing_px <= 0:
        return float(draw.textlength(line, font=font))
    width = 0.0
    for index, char in enumerate(line):
        width += float(draw.textlength(char, font=font))
        if index < len(line) - 1:
            width += letter_spacing_px
    return width


def _draw_spaced_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    x: float,
    y: float,
    font: ImageFont.FreeTypeFont,
    *,
    fill,
    letter_spacing_px: float,
    stroke_width: float = 0.0,
    stroke_fill=None,
) -> None:
    if letter_spacing_px <= 0:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=fill,
            stroke_width=int(round(stroke_width)) if stroke_width > 0 else 0,
            stroke_fill=stroke_fill,
        )
        return

    cursor = x
    for char in line:
        draw.text(
            (cursor, y),
            char,
            font=font,
            fill=fill,
            stroke_width=int(round(stroke_width)) if stroke_width > 0 else 0,
            stroke_fill=stroke_fill,
        )
        cursor += float(draw.textlength(char, font=font)) + letter_spacing_px


def render_glyph_alpha_mask(
    block: TextBlock,
    layout: TextLayoutContext,
) -> Image.Image:
    """Render glyph alpha mask for a laid-out text block."""
    bbox = block.bbox
    mask = Image.new("L", (bbox.width, bbox.height), 0)
    draw = ImageDraw.Draw(mask)
    font = ImageFont.truetype(layout.font_path, layout.font_size_px)

    for index, line in enumerate(layout.lines):
        line_width = _measure_spaced_line(
            draw,
            line,
            font,
            letter_spacing_px=layout.letter_spacing_px,
        )
        if layout.horizontal == "left":
            x = float(layout.content_box.x)
        elif layout.horizontal == "right":
            x = float(layout.content_box.x) + max(0.0, layout.content_box.width - line_width)
        else:
            x = float(layout.content_box.x) + max(0.0, (layout.content_box.width - line_width) / 2.0)
        y = layout.y_start + index * layout.line_height_px
        _draw_spaced_line(
            draw,
            line,
            x,
            y,
            font,
            fill=255,
            letter_spacing_px=layout.letter_spacing_px,
        )
    return mask


def _shadow_settings(block: TextBlock) -> tuple[tuple[int, int, int], float, float, float, float] | None:
    visual = block.visual
    if visual.shadow_confidence < SHADOW_MIN_CONFIDENCE:
        return None
    color = _hex_to_rgb(visual.shadow_color) or (0, 0, 0)
    opacity = 0.45 if visual.shadow_opacity is None else max(0.0, min(1.0, float(visual.shadow_opacity)))
    offset_x = 0.0 if visual.shadow_offset_x_px is None else float(visual.shadow_offset_x_px)
    offset_y = 2.0 if visual.shadow_offset_y_px is None else float(visual.shadow_offset_y_px)
    blur = 2.0 if visual.shadow_blur_px is None else max(0.0, float(visual.shadow_blur_px))
    return color, opacity, offset_x, offset_y, blur


def _render_shadow_layer(
    alpha_mask: Image.Image,
    *,
    color: tuple[int, int, int],
    opacity: float,
    offset_x: float,
    offset_y: float,
    blur_radius: float,
) -> Image.Image:
    shadow = Image.new("RGBA", alpha_mask.size, (0, 0, 0, 0))
    shadow_alpha = alpha_mask.point(lambda value: int(value * opacity))
    shadow_rgb = Image.new("RGBA", alpha_mask.size, (*color, 0))
    shadow_rgb.putalpha(shadow_alpha)
    if blur_radius > 0:
        shadow_rgb = shadow_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    shadow.paste(shadow_rgb, (int(round(offset_x)), int(round(offset_y))), shadow_rgb)
    return shadow


def _linear_gradient_rgba(
    size: tuple[int, int],
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
    angle_deg: float,
) -> Image.Image:
    width, height = size
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    centered_x = grid_x - (width - 1) / 2.0
    centered_y = grid_y - (height - 1) / 2.0
    projection = centered_x * cos_a + centered_y * sin_a
    min_proj = float(projection.min())
    max_proj = float(projection.max())
    if max_proj - min_proj < 1e-6:
        ratio = np.zeros_like(projection)
    else:
        ratio = (projection - min_proj) / (max_proj - min_proj)
    start = np.array(start_rgb, dtype=np.float32)
    end = np.array(end_rgb, dtype=np.float32)
    rgb = start[None, None, :] + ratio[:, :, None] * (end - start)[None, None, :]
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def _resolve_fill_layer(
    alpha_mask: Image.Image,
    block: TextBlock,
    text_color: tuple[int, int, int],
    text_opacity: int,
) -> tuple[Image.Image, bool, list[str]]:
    visual = block.visual
    warnings: list[str] = []

    if (
        visual.fill_type == "gradient"
        and visual.gradient_confidence >= GRADIENT_MIN_CONFIDENCE
        and visual.gradient_start_color
        and visual.gradient_end_color
    ):
        start = _hex_to_rgb(visual.gradient_start_color)
        end = _hex_to_rgb(visual.gradient_end_color)
        if start is not None and end is not None:
            angle = 0.0 if visual.gradient_angle_deg is None else float(visual.gradient_angle_deg)
            gradient = _linear_gradient_rgba(alpha_mask.size, start, end, angle)
            gradient.putalpha(alpha_mask)
            return gradient, True, warnings

    if visual.fill_type == "gradient":
        warnings.append("gradient metadata insufficient; using solid text_color")

    solid = Image.new("RGBA", alpha_mask.size, (*text_color, text_opacity))
    solid.putalpha(alpha_mask)
    return solid, False, warnings


def _apply_stroke_layer(
    base_layer: Image.Image,
    block: TextBlock,
    layout: TextLayoutContext,
    *,
    stroke_width: float,
    stroke_color: tuple[int, int, int],
    text_opacity: int,
) -> Image.Image:
    stroke_layer = Image.new("RGBA", base_layer.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(stroke_layer)
    font = ImageFont.truetype(layout.font_path, layout.font_size_px)
    stroke_fill = (*stroke_color, text_opacity)

    for index, line in enumerate(layout.lines):
        line_width = _measure_spaced_line(
            draw,
            line,
            font,
            letter_spacing_px=layout.letter_spacing_px,
        )
        if layout.horizontal == "left":
            x = float(layout.content_box.x)
        elif layout.horizontal == "right":
            x = float(layout.content_box.x) + max(0.0, layout.content_box.width - line_width)
        else:
            x = float(layout.content_box.x) + max(0.0, (layout.content_box.width - line_width) / 2.0)
        y = layout.y_start + index * layout.line_height_px
        _draw_spaced_line(
            draw,
            line,
            x,
            y,
            font,
            fill=(0, 0, 0, 0),
            letter_spacing_px=layout.letter_spacing_px,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    return Image.alpha_composite(base_layer, stroke_layer)


def _compute_rotation_paste_offset(
    layer_size: tuple[int, int],
    rotation_deg: float,
    *,
    center: tuple[float, float],
) -> tuple[int, int]:
    """Compute layer-local paste offset so rotation center stays fixed in image space."""
    width, height = layer_size
    cx, cy = center
    angle_rad = math.radians(-rotation_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated_points: list[tuple[float, float]] = []
    for x, y in ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))):
        dx = x - cx
        dy = y - cy
        rotated_points.append(
            (
                cx + dx * cos_a - dy * sin_a,
                cy + dx * sin_a + dy * cos_a,
            )
        )

    min_x = min(point[0] for point in rotated_points)
    min_y = min(point[1] for point in rotated_points)
    return int(math.floor(min_x)), int(math.floor(min_y))


def rotate_layer(
    layer: Image.Image,
    rotation_deg: float,
    *,
    center: tuple[float, float],
    preserve_ink: bool = True,
) -> tuple[Image.Image, tuple[int, int]]:
    """Rotate a layer around center, optionally preserving full transformed ink."""
    if abs(rotation_deg) <= ROTATION_EPSILON_DEG:
        return layer, (0, 0)

    if preserve_ink:
        rotated = layer.rotate(
            -rotation_deg,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            center=center,
        )
        paste_offset = _compute_rotation_paste_offset(layer.size, rotation_deg, center=center)
        return rotated, paste_offset

    rotated = layer.rotate(
        -rotation_deg,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        center=center,
    )
    return rotated, (0, 0)


def warp_perspective_layer(
    layer: Image.Image,
    target_quad: list[tuple[float, float]],
) -> Image.Image:
    width, height = layer.size
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    destination = np.float32(target_quad)
    matrix = cv2.getPerspectiveTransform(source, destination)
    rgba = np.array(layer.convert("RGBA"))
    warped = cv2.warpPerspective(
        rgba,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped, mode="RGBA")


def apply_text_effects(
    alpha_mask: Image.Image,
    block: TextBlock,
    layout: TextLayoutContext,
    *,
    text_color: tuple[int, int, int],
    text_opacity: int,
    rotation_deg: float,
    stroke_width: float = 0.0,
    stroke_color: tuple[int, int, int] | None = None,
) -> EffectApplicationResult:
    """Apply shadow, fill, stroke, rotation, and perspective to a glyph mask."""
    warnings: list[str] = []
    bbox = block.bbox
    working = Image.new("RGBA", alpha_mask.size, (0, 0, 0, 0))

    shadow_applied = False
    shadow_settings = _shadow_settings(block)
    if shadow_settings is not None:
        color, opacity, offset_x, offset_y, blur = shadow_settings
        shadow_layer = _render_shadow_layer(
            alpha_mask,
            color=color,
            opacity=opacity,
            offset_x=offset_x,
            offset_y=offset_y,
            blur_radius=blur,
        )
        working = Image.alpha_composite(working, shadow_layer)
        shadow_applied = True

    fill_layer, gradient_applied, fill_warnings = _resolve_fill_layer(
        alpha_mask,
        block,
        text_color,
        text_opacity,
    )
    warnings.extend(fill_warnings)
    working = Image.alpha_composite(working, fill_layer)

    if stroke_width > 0 and stroke_color is not None:
        working = _apply_stroke_layer(
            working,
            block,
            layout,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            text_opacity=text_opacity,
        )

    applied_rotation = float(rotation_deg or 0.0)
    perspective_applied = False
    paste_offset = (0, 0)
    target_quad, perspective_warnings = resolve_perspective_target_quad(block)
    warnings.extend(perspective_warnings)

    working, paste_offset = rotate_layer(
        working,
        applied_rotation,
        center=(bbox.width / 2.0, bbox.height / 2.0),
        preserve_ink=target_quad is None,
    )

    if target_quad is not None:
        working = warp_perspective_layer(working, target_quad)
        perspective_applied = True

    return EffectApplicationResult(
        layer=working,
        applied_rotation=applied_rotation,
        perspective_applied=perspective_applied,
        shadow_applied=shadow_applied,
        gradient_applied=gradient_applied,
        paste_offset=paste_offset,
        warnings=warnings,
    )
