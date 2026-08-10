"""Deterministic translated text rendering for Renderer R0.4/R0.5."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imageloc.config import PROJECT_ROOT
from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import TextBlock
from imageloc.renderer.background_restorer import BackgroundRestoreResult, restore_backgrounds

FONT_MIN_SIZE_PX = 6
FONT_MAX_SIZE_PX = 256
FONT_SIZE_SEARCH_MAX_ITERATIONS = 16
STROKE_MIN_CONFIDENCE = 0.5
DEFAULT_LINE_HEIGHT_RATIO = 1.2
LINE_HEIGHT_CONFIDENCE_MIN = 0.4
DEFAULT_HORIZONTAL_ALIGNMENT = "center"
DEFAULT_VERTICAL_ALIGNMENT = "center"
LOCALIZED_IMAGE_FILENAME = "localized_image.png"

CONTENT_PADDING_MIN_PX = 2
CONTENT_PADDING_MAX_PX = 16
CONTENT_PADDING_BBOX_RATIO = 0.06
CONTENT_PADDING_FONT_RATIO = 0.18
CONTENT_MIN_INNER_PX = 8

WINDOWS_FONTS_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"

FONT_CATEGORY_FILES: dict[str, list[str]] = {
    "sans-serif": ["segoeui.ttf", "arial.ttf", "calibri.ttf", "tahoma.ttf"],
    "serif": ["times.ttf", "georgia.ttf", "timesbd.ttf"],
    "monospace": ["consola.ttf", "cour.ttf", "lucon.ttf"],
}

FONT_VARIANTS: dict[str, dict[str, str]] = {
    "segoeui.ttf": {
        "normal": "segoeui.ttf",
        "bold": "segoeuib.ttf",
        "italic": "segoeuii.ttf",
        "bold_italic": "segoeuiz.ttf",
    },
    "arial.ttf": {
        "normal": "arial.ttf",
        "bold": "arialbd.ttf",
        "italic": "ariali.ttf",
        "bold_italic": "arialbi.ttf",
    },
    "calibri.ttf": {
        "normal": "calibri.ttf",
        "bold": "calibrib.ttf",
        "italic": "calibrii.ttf",
        "bold_italic": "calibriz.ttf",
    },
    "times.ttf": {
        "normal": "times.ttf",
        "bold": "timesbd.ttf",
        "italic": "timesi.ttf",
        "bold_italic": "timesbi.ttf",
    },
    "consola.ttf": {
        "normal": "consola.ttf",
        "bold": "consolab.ttf",
        "italic": "consolai.ttf",
        "bold_italic": "consolaz.ttf",
    },
    "cour.ttf": {
        "normal": "cour.ttf",
        "bold": "courbd.ttf",
        "italic": "couri.ttf",
        "bold_italic": "courbi.ttf",
    },
}

CYRILLIC_PROBE = "АБВ"
UNICODE_FALLBACK_FILES = ["segoeui.ttf", "arial.ttf", "calibri.ttf", "tahoma.ttf"]


@dataclass
class TextRenderResult:
    """Internal result of one text rendering attempt."""

    image: Image.Image
    success: bool
    block_id: str
    font_path: str | None = None
    font_size_px: int | None = None
    rendered_lines: list[str] = field(default_factory=list)
    rendered_bbox: BoundingBox | None = None
    content_bbox: BoundingBox | None = None
    content_padding_px: int = 0
    fitted_width_px: float | None = None
    fitted_height_px: float | None = None
    failure_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _hex_to_rgb(color: str | None) -> tuple[int, int, int] | None:
    if not color or not color.startswith("#") or len(color) != 7:
        return None
    try:
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    except ValueError:
        return None


def _normalize_horizontal_alignment(value: str | None) -> str:
    normalized = (value or DEFAULT_HORIZONTAL_ALIGNMENT).strip().lower()
    if normalized in {"left", "center", "centre", "middle"}:
        return "left" if normalized == "left" else "center"
    if normalized in {"right"}:
        return "right"
    return DEFAULT_HORIZONTAL_ALIGNMENT


def _normalize_vertical_alignment(value: str | None) -> str:
    normalized = (value or DEFAULT_VERTICAL_ALIGNMENT).strip().lower()
    if normalized in {"top"}:
        return "top"
    if normalized in {"bottom"}:
        return "bottom"
    return DEFAULT_VERTICAL_ALIGNMENT


def _normalize_weight(value: str | None) -> str:
    normalized = (value or "normal").strip().lower()
    if normalized in {"bold", "700", "800", "900"}:
        return "bold"
    return "normal"


def _normalize_style(value: str | None) -> str:
    normalized = (value or "normal").strip().lower()
    if normalized in {"italic", "oblique"}:
        return "italic"
    return "normal"


def compute_content_padding_px(block: TextBlock) -> int:
    """Compute deterministic inner padding for a text block bbox."""
    bbox = block.bbox
    visual = block.visual
    hints = block.render_hints
    outer_w = min(bbox.width, hints.available_width_px)
    outer_h = min(bbox.height, hints.available_height_px)
    font_hint = visual.font_size_estimate or visual.estimated_text_height_px or min(outer_w, outer_h)
    from_bbox = int(round(min(outer_w, outer_h) * CONTENT_PADDING_BBOX_RATIO))
    from_font = int(round(max(1, font_hint) * CONTENT_PADDING_FONT_RATIO))

    compact = min(outer_w, outer_h) < 56
    if compact:
        padding = max(CONTENT_PADDING_MIN_PX, min(from_bbox, from_font))
    else:
        padding = max(from_bbox, from_font, CONTENT_PADDING_MIN_PX)
    padding = min(padding, CONTENT_PADDING_MAX_PX)

    max_pad_x = max(0, (outer_w - CONTENT_MIN_INNER_PX) // 2)
    max_pad_y = max(0, (outer_h - CONTENT_MIN_INNER_PX) // 2)
    if max_pad_x == 0 or max_pad_y == 0:
        return 0
    return min(padding, max_pad_x, max_pad_y)


def compute_content_box(block: TextBlock) -> tuple[int, BoundingBox]:
    """Return padding and inner content box in local bbox coordinates."""
    padding = compute_content_padding_px(block)
    bbox = block.bbox
    hints = block.render_hints
    outer_w = min(bbox.width, hints.available_width_px)
    outer_h = min(bbox.height, hints.available_height_px)
    inner_w = max(1, outer_w - 2 * padding)
    inner_h = max(1, outer_h - 2 * padding)
    return padding, BoundingBox(x=padding, y=padding, width=inner_w, height=inner_h)


def _font_file_exists(filename: str) -> bool:
    return (WINDOWS_FONTS_DIR / filename).is_file()


def _supports_cyrillic(filename: str) -> bool:
    path = WINDOWS_FONTS_DIR / filename
    if not path.is_file():
        return False
    try:
        font = ImageFont.truetype(str(path), size=24)
        probe = Image.new("RGBA", (96, 48), (255, 255, 255, 0))
        draw = ImageDraw.Draw(probe)
        draw.text((4, 4), CYRILLIC_PROBE, font=font, fill=(0, 0, 0, 255))
        alpha = np.array(probe)[:, :, 3]
        return int(np.count_nonzero(alpha)) > 0
    except OSError:
        return False


def _resolve_variant_filename(base_filename: str, *, weight: str, style: str) -> str:
    variants = FONT_VARIANTS.get(base_filename.lower())
    if variants is None:
        return base_filename
    key = "bold_italic" if weight == "bold" and style == "italic" else weight if weight == "bold" else style
    if key not in variants:
        key = "normal"
    candidate = variants[key]
    if _font_file_exists(candidate):
        return candidate
    if _font_file_exists(variants["normal"]):
        return variants["normal"]
    return base_filename


def _candidate_font_files(font_family_candidate: str) -> list[str]:
    candidate_path = Path(font_family_candidate)
    if candidate_path.is_file():
        return [candidate_path.name]

    lowered = font_family_candidate.strip().lower()
    if lowered.endswith(".ttf") or lowered.endswith(".otf"):
        return [Path(lowered).name]

    name_map = {
        "segoe ui": "segoeui.ttf",
        "arial": "arial.ttf",
        "calibri": "calibri.ttf",
        "times new roman": "times.ttf",
        "consolas": "consola.ttf",
        "courier new": "cour.ttf",
        "tahoma": "tahoma.ttf",
    }
    if lowered in name_map:
        return [name_map[lowered]]
    return []


def resolve_font(
    block: TextBlock,
) -> tuple[str | None, list[str]]:
    """Resolve a deterministic local font path for a TextBlock."""
    warnings: list[str] = []
    visual = block.visual
    weight = _normalize_weight(visual.font_weight)
    style = _normalize_style(visual.font_style)

    candidates: list[str] = []
    if visual.font_family_candidate and block.render_hints.allow_font_substitution:
        candidates.extend(_candidate_font_files(visual.font_family_candidate))
    elif visual.font_family_candidate and not block.render_hints.allow_font_substitution:
        candidates.extend(_candidate_font_files(visual.font_family_candidate))

    category = (visual.font_category or "sans-serif").strip().lower()
    candidates.extend(FONT_CATEGORY_FILES.get(category, []))
    candidates.extend(UNICODE_FALLBACK_FILES)

    seen: set[str] = set()
    for filename in candidates:
        if filename in seen:
            continue
        seen.add(filename)
        resolved = _resolve_variant_filename(filename, weight=weight, style=style)
        if not _font_file_exists(resolved):
            continue
        if not _supports_cyrillic(resolved):
            warnings.append(f"font '{resolved}' failed Cyrillic probe; trying next fallback")
            continue
        return str(WINDOWS_FONTS_DIR / resolved), warnings

    warnings.append("no Unicode-capable system font found for Cyrillic text")
    return None, warnings


def _apply_text_transform(text: str, transform: str | None) -> str:
    if not transform:
        return text
    normalized = transform.strip().lower()
    if normalized == "uppercase":
        return text.upper()
    if normalized == "lowercase":
        return text.lower()
    if normalized in {"capitalize", "title"}:
        return text.title()
    return text


def _fallback_text_color(image: Image.Image, bbox: BoundingBox) -> tuple[int, int, int]:
    region = np.array(image.crop((bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)))
    if region.ndim == 3 and region.shape[2] >= 3:
        avg = region[:, :, :3].mean(axis=(0, 1))
    else:
        avg = np.array([255, 255, 255], dtype=np.float32)
    luminance = 0.299 * avg[0] + 0.587 * avg[1] + 0.114 * avg[2]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


def _resolve_text_color(image: Image.Image, block: TextBlock) -> tuple[tuple[int, int, int], list[str]]:
    warnings: list[str] = []
    if block.visual.fill_type == "gradient":
        warnings.append("gradient text fill not supported; using solid text_color")
    rgb = _hex_to_rgb(block.visual.text_color)
    if rgb is not None:
        return rgb, warnings
    warnings.append("text_color missing; using contrast fallback")
    return _fallback_text_color(image, block.bbox), warnings


def _resolve_line_height_ratio(block: TextBlock, font: ImageFont.FreeTypeFont | None = None) -> float:
    visual = block.visual
    if visual.line_height_ratio is not None and visual.line_height_confidence >= LINE_HEIGHT_CONFIDENCE_MIN:
        return max(0.8, min(2.5, float(visual.line_height_ratio)))
    if font is not None:
        ascent, descent = font.getmetrics()
        if ascent + descent > 0:
            return DEFAULT_LINE_HEIGHT_RATIO
    return DEFAULT_LINE_HEIGHT_RATIO


def _line_height_px(font: ImageFont.FreeTypeFont, line_height_ratio: float) -> int:
    ascent, descent = font.getmetrics()
    return max(1, int(round((ascent + descent) * line_height_ratio)))


def _stroke_margin_px(stroke_width: float) -> float:
    if stroke_width <= 0:
        return 0.0
    return float(stroke_width)


def _measure_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    font: ImageFont.FreeTypeFont,
    *,
    letter_spacing_px: float = 0.0,
    stroke_width: float = 0.0,
) -> float:
    if not line:
        return 0.0
    if letter_spacing_px <= 0:
        width = float(draw.textlength(line, font=font))
    else:
        width = 0.0
        for index, char in enumerate(line):
            width += float(draw.textlength(char, font=font))
            if index < len(line) - 1:
                width += letter_spacing_px
    return width + _stroke_margin_px(stroke_width) * 2.0


def _split_long_word(
    word: str,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    max_width: float,
    *,
    letter_spacing_px: float = 0.0,
    stroke_width: float = 0.0,
) -> list[str]:
    parts: list[str] = []
    current = ""
    for char in word:
        candidate = current + char
        if (
            _measure_line(
                draw,
                candidate,
                font,
                letter_spacing_px=letter_spacing_px,
                stroke_width=stroke_width,
            )
            <= max_width
            or not current
        ):
            current = candidate
        else:
            parts.append(current)
            current = char
    if current:
        parts.append(current)
    return parts or [word[:1]]


def _score_line_layout(
    lines: list[str],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: float,
    letter_spacing_px: float = 0.0,
    stroke_width: float = 0.0,
) -> float:
    if not lines:
        return float("inf")
    widths = [
        _measure_line(
            draw,
            line,
            font,
            letter_spacing_px=letter_spacing_px,
            stroke_width=stroke_width,
        )
        for line in lines
    ]
    if any(width > max_width for width in widths):
        return float("inf")
    average = sum(widths) / len(widths)
    variance = sum((width - average) ** 2 for width in widths) / len(widths)
    short_last = 0.0
    if len(widths) > 1 and widths[-1] < average * 0.55:
        short_last = (average * 0.55 - widths[-1]) ** 2
    return variance + short_last + len(lines) * 0.05


def _wrap_greedy(
    words: list[str],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: float,
    max_lines: int,
    letter_spacing_px: float,
    stroke_width: float,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if (
            _measure_line(
                draw,
                candidate,
                font,
                letter_spacing_px=letter_spacing_px,
                stroke_width=stroke_width,
            )
            <= max_width
        ):
            current = candidate
            continue

        if current:
            lines.append(current)
            current = word
        else:
            lines.extend(
                _split_long_word(
                    word,
                    draw,
                    font,
                    max_width,
                    letter_spacing_px=letter_spacing_px,
                    stroke_width=stroke_width,
                )
            )
            current = ""

        if len(lines) >= max_lines:
            break

    if current and len(lines) < max_lines:
        if (
            _measure_line(
                draw,
                current,
                font,
                letter_spacing_px=letter_spacing_px,
                stroke_width=stroke_width,
            )
            <= max_width
        ):
            lines.append(current)
        else:
            for part in _split_long_word(
                current,
                draw,
                font,
                max_width,
                letter_spacing_px=letter_spacing_px,
                stroke_width=stroke_width,
            ):
                if len(lines) >= max_lines:
                    break
                lines.append(part)

    return lines[:max_lines]


def _wrap_balanced(
    words: list[str],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: float,
    max_lines: int,
    letter_spacing_px: float,
    stroke_width: float,
) -> list[str]:
    word_count = len(words)
    if word_count == 0:
        return []

    best_lines: list[str] | None = None
    best_score = float("inf")
    target_width = max_width * 0.72

    for line_count in range(1, min(max_lines, word_count) + 1):
        dp_score = [float("inf")] * (word_count + 1)
        dp_prev = [-1] * (word_count + 1)
        dp_lines = [0] * (word_count + 1)
        dp_score[0] = 0.0

        for start in range(word_count):
            if dp_score[start] == float("inf"):
                continue
            if dp_lines[start] >= line_count:
                continue
            for end in range(start + 1, word_count + 1):
                if dp_lines[start] + 1 > line_count:
                    break
                line = " ".join(words[start:end])
                line_width = _measure_line(
                    draw,
                    line,
                    font,
                    letter_spacing_px=letter_spacing_px,
                    stroke_width=stroke_width,
                )
                if line_width > max_width:
                    break
                segment_cost = (line_width - target_width) ** 2
                total = dp_score[start] + segment_cost
                if total < dp_score[end]:
                    dp_score[end] = total
                    dp_prev[end] = start
                    dp_lines[end] = dp_lines[start] + 1

        if dp_score[word_count] == float("inf"):
            continue

        lines: list[str] = []
        index = word_count
        while index > 0:
            start = dp_prev[index]
            if start < 0:
                lines = []
                break
            lines.append(" ".join(words[start:index]))
            index = start
        if not lines:
            continue
        lines.reverse()
        score = _score_line_layout(
            lines,
            draw,
            font,
            max_width=max_width,
            letter_spacing_px=letter_spacing_px,
            stroke_width=stroke_width,
        )
        if score < best_score:
            best_score = score
            best_lines = lines

    if best_lines is not None:
        return best_lines
    return _wrap_greedy(
        words,
        draw,
        font,
        max_width=max_width,
        max_lines=max_lines,
        letter_spacing_px=letter_spacing_px,
        stroke_width=stroke_width,
    )


def wrap_text_lines(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: int,
    max_line_count: int | None,
    letter_spacing_px: float = 0.0,
    stroke_width: float = 0.0,
) -> list[str]:
    """Wrap text deterministically using measured glyph widths and balanced scoring."""
    probe = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    max_lines = max_line_count if max_line_count and max_line_count > 0 else 10_000
    words = text.split()
    if not words:
        return []

    if len(words) == 1:
        word = words[0]
        if (
            _measure_line(
                draw,
                word,
                font,
                letter_spacing_px=letter_spacing_px,
                stroke_width=stroke_width,
            )
            <= max_width
        ):
            return [word]
        return _split_long_word(
            word,
            draw,
            font,
            float(max_width),
            letter_spacing_px=letter_spacing_px,
            stroke_width=stroke_width,
        )[:max_lines]

    return _wrap_balanced(
        words,
        draw,
        font,
        max_width=float(max_width),
        max_lines=max_lines,
        letter_spacing_px=letter_spacing_px,
        stroke_width=stroke_width,
    )


def _text_fully_represented(source: str, lines: list[str]) -> bool:
    rendered = "".join(lines).replace(" ", "")
    return rendered == source.replace(" ", "")


def _block_ink_height(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    *,
    line_height: int,
    stroke_width: float = 0.0,
) -> float:
    if not lines:
        return 0.0
    stroke_margin = _stroke_margin_px(stroke_width)
    first_box = font.getbbox(lines[0])
    if len(lines) == 1:
        return float(first_box[3] - first_box[1]) + stroke_margin * 2.0
    last_box = font.getbbox(lines[-1])
    return float((len(lines) - 1) * line_height + last_box[3] - first_box[1]) + stroke_margin * 2.0


def _measure_text_block(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    *,
    letter_spacing_px: float = 0.0,
    line_height_ratio: float,
    stroke_width: float = 0.0,
) -> tuple[float, float]:
    probe = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    line_height = _line_height_px(font, line_height_ratio)
    max_width = 0.0
    for line in lines:
        max_width = max(
            max_width,
            _measure_line(
                draw,
                line,
                font,
                letter_spacing_px=letter_spacing_px,
                stroke_width=stroke_width,
            ),
        )
    total_height = _block_ink_height(lines, font, line_height=line_height, stroke_width=stroke_width)
    return max_width, total_height


def _font_size_bounds(block: TextBlock) -> tuple[int, int]:
    hints = block.render_hints
    visual = block.visual
    base = visual.font_size_estimate or visual.estimated_text_height_px or hints.available_height_px
    base = max(1, int(base))
    min_size = max(FONT_MIN_SIZE_PX, int(round(base * hints.font_scale_min)))
    max_size = min(
        FONT_MAX_SIZE_PX,
        int(round(base * hints.font_scale_max)),
        hints.available_height_px,
        hints.available_width_px,
    )
    if min_size > max_size:
        min_size = max(FONT_MIN_SIZE_PX, max_size)
    return min_size, max_size


def _layout_fits(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    *,
    source_text: str,
    content_box: BoundingBox,
    max_line_count: int | None,
    letter_spacing_px: float,
    line_height_ratio: float,
    stroke_width: float,
) -> tuple[bool, float, float]:
    width, height = _measure_text_block(
        lines,
        font,
        letter_spacing_px=letter_spacing_px,
        line_height_ratio=line_height_ratio,
        stroke_width=stroke_width,
    )
    fits = width <= content_box.width and height <= content_box.height and bool(lines)
    if max_line_count is not None:
        fits = fits and len(lines) <= max_line_count
    fits = fits and _text_fully_represented(source_text, lines)
    return fits, width, height


def fit_font_size_and_lines(
    text: str,
    font_path: str,
    block: TextBlock,
    *,
    content_box: BoundingBox,
    letter_spacing_px: float = 0.0,
    line_height_ratio: float | None = None,
    stroke_width: float = 0.0,
) -> tuple[int | None, list[str], float, float]:
    """Binary-search the largest font size that fits inside the content box."""
    hints = block.render_hints
    min_size, max_size = _font_size_bounds(block)
    best_size: int | None = None
    best_lines: list[str] = []
    best_width = 0.0
    best_height = 0.0

    low = min_size
    high = max_size
    iterations = 0
    while low <= high and iterations < FONT_SIZE_SEARCH_MAX_ITERATIONS:
        iterations += 1
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, mid)
        ratio = line_height_ratio if line_height_ratio is not None else _resolve_line_height_ratio(block, font)
        lines = wrap_text_lines(
            text,
            font,
            max_width=content_box.width,
            max_line_count=hints.max_line_count,
            letter_spacing_px=letter_spacing_px,
            stroke_width=stroke_width,
        )
        fits, width, height = _layout_fits(
            lines,
            font,
            source_text=text,
            content_box=content_box,
            max_line_count=hints.max_line_count,
            letter_spacing_px=letter_spacing_px,
            line_height_ratio=ratio,
            stroke_width=stroke_width,
        )
        if fits:
            best_size = mid
            best_lines = lines
            best_width = width
            best_height = height
            low = mid + 1
        else:
            high = mid - 1

    if best_size is not None:
        return best_size, best_lines, best_width, best_height

    font = ImageFont.truetype(font_path, min_size)
    ratio = line_height_ratio if line_height_ratio is not None else _resolve_line_height_ratio(block, font)
    lines = wrap_text_lines(
        text,
        font,
        max_width=content_box.width,
        max_line_count=hints.max_line_count,
        letter_spacing_px=letter_spacing_px,
        stroke_width=stroke_width,
    )
    fits, width, height = _layout_fits(
        lines,
        font,
        source_text=text,
        content_box=content_box,
        max_line_count=hints.max_line_count,
        letter_spacing_px=letter_spacing_px,
        line_height_ratio=ratio,
        stroke_width=stroke_width,
    )
    if fits:
        return min_size, lines, width, height
    return None, lines, width, height


def _draw_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    x: float,
    y: float,
    font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int, int],
    letter_spacing_px: float = 0.0,
    stroke_width: float = 0.0,
    stroke_fill: tuple[int, int, int, int] | None = None,
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
    for index, char in enumerate(line):
        draw.text(
            (cursor, y),
            char,
            font=font,
            fill=fill,
            stroke_width=int(round(stroke_width)) if stroke_width > 0 else 0,
            stroke_fill=stroke_fill,
        )
        cursor += float(draw.textlength(char, font=font)) + letter_spacing_px


def _render_block_layer(
    block: TextBlock,
    *,
    font_path: str,
    font_size_px: int,
    lines: list[str],
    text_color: tuple[int, int, int],
    content_box: BoundingBox,
    letter_spacing_px: float,
    line_height_ratio: float,
    stroke_width: float,
    stroke_color: tuple[int, int, int] | None,
    rotation_deg: float,
) -> Image.Image:
    bbox = block.bbox
    opacity = block.visual.text_opacity
    alpha = 255 if opacity is None else int(max(0, min(1.0, opacity)) * 255)
    fill = (*text_color, alpha)
    stroke_fill = (*stroke_color, alpha) if stroke_color is not None else None

    layer = Image.new("RGBA", (bbox.width, bbox.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.truetype(font_path, font_size_px)
    line_height = _line_height_px(font, line_height_ratio)
    block_height = _block_ink_height(
        lines,
        font,
        line_height=line_height,
        stroke_width=stroke_width,
    )
    first_box = font.getbbox(lines[0]) if lines else (0, 0, 0, 0)

    horizontal = _normalize_horizontal_alignment(block.visual.alignment)
    vertical = _normalize_vertical_alignment(block.visual.vertical_alignment)

    if vertical == "top":
        y_start = content_box.y - first_box[1]
    elif vertical == "bottom":
        y_start = content_box.y + max(0.0, content_box.height - block_height) - first_box[1]
    else:
        y_start = content_box.y + max(0.0, (content_box.height - block_height) / 2.0) - first_box[1]

    for index, line in enumerate(lines):
        line_width = _measure_line(
            draw,
            line,
            font,
            letter_spacing_px=letter_spacing_px,
            stroke_width=stroke_width,
        )
        if horizontal == "left":
            x = float(content_box.x)
        elif horizontal == "right":
            x = float(content_box.x) + max(0.0, content_box.width - line_width)
        else:
            x = float(content_box.x) + max(0.0, (content_box.width - line_width) / 2.0)
        y = y_start + index * line_height
        _draw_line(
            draw,
            line,
            x,
            y,
            font,
            fill=fill,
            letter_spacing_px=letter_spacing_px,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    if abs(rotation_deg) > 0.01:
        layer = layer.rotate(
            -rotation_deg,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            center=(bbox.width / 2.0, bbox.height / 2.0),
        )
    return layer


def _paste_layer_into_image(base: Image.Image, layer: Image.Image, bbox: BoundingBox) -> Image.Image:
    working = base.convert("RGBA") if base.mode != "RGBA" else base.copy()
    region = working.crop((bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height))
    composited = Image.alpha_composite(region, layer)
    working.paste(composited, (bbox.x, bbox.y))
    if base.mode == "RGBA":
        return working
    return working.convert("RGB")


def render_text_block(
    image: Image.Image,
    block: TextBlock,
) -> TextRenderResult:
    """Render one translated TextBlock onto a clean/restored image."""
    warnings: list[str] = []
    translated = block.translated_text
    if translated is None or not str(translated).strip():
        return TextRenderResult(
            image=image.copy(),
            success=True,
            block_id=block.id,
            warnings=["empty translated_text; skipped"],
        )

    if block.render_geometry and block.render_geometry.perspective_detected:
        warnings.append("perspective detected; rendering without perspective warp")

    font_path, font_warnings = resolve_font(block)
    warnings.extend(font_warnings)
    if font_path is None:
        return TextRenderResult(
            image=image.copy(),
            success=False,
            block_id=block.id,
            failure_reason="font_resolution_failed",
            warnings=warnings,
        )

    padding_px, content_box = compute_content_box(block)
    text = _apply_text_transform(str(translated).strip(), block.visual.text_transform)
    text_color, color_warnings = _resolve_text_color(image, block)
    warnings.extend(color_warnings)

    letter_spacing_px = 0.0
    if block.visual.letter_spacing_px is not None:
        letter_spacing_px = max(0.0, float(block.visual.letter_spacing_px))

    stroke_width = 0.0
    stroke_color: tuple[int, int, int] | None = None
    if (
        block.visual.stroke_confidence >= STROKE_MIN_CONFIDENCE
        and block.visual.stroke_width_px > 0
        and block.visual.stroke_color
    ):
        parsed_stroke = _hex_to_rgb(block.visual.stroke_color)
        if parsed_stroke is not None:
            stroke_width = float(block.visual.stroke_width_px)
            stroke_color = parsed_stroke

    line_height_ratio = _resolve_line_height_ratio(block)

    font_size_px, lines, fitted_width, fitted_height = fit_font_size_and_lines(
        text,
        font_path,
        block,
        content_box=content_box,
        letter_spacing_px=letter_spacing_px,
        line_height_ratio=line_height_ratio,
        stroke_width=stroke_width,
    )
    if font_size_px is None or not lines:
        failure_reason = "text_does_not_fit_at_minimum_font_size"
        warnings.append("text does not fit at minimum font size")
        return TextRenderResult(
            image=image.copy(),
            success=False,
            block_id=block.id,
            font_path=font_path,
            rendered_lines=lines,
            rendered_bbox=block.bbox,
            content_bbox=content_box,
            content_padding_px=padding_px,
            fitted_width_px=fitted_width,
            fitted_height_px=fitted_height,
            failure_reason=failure_reason,
            warnings=warnings,
        )

    ratio = _resolve_line_height_ratio(block, ImageFont.truetype(font_path, font_size_px))
    rotation_deg = float(block.visual.rotation_deg or 0.0)
    layer = _render_block_layer(
        block,
        font_path=font_path,
        font_size_px=font_size_px,
        lines=lines,
        text_color=text_color,
        content_box=content_box,
        letter_spacing_px=letter_spacing_px,
        line_height_ratio=ratio,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        rotation_deg=rotation_deg,
    )
    rendered = _paste_layer_into_image(image, layer, block.bbox)
    return TextRenderResult(
        image=rendered,
        success=True,
        block_id=block.id,
        font_path=font_path,
        font_size_px=font_size_px,
        rendered_lines=lines,
        rendered_bbox=block.bbox,
        content_bbox=content_box,
        content_padding_px=padding_px,
        fitted_width_px=fitted_width,
        fitted_height_px=fitted_height,
        warnings=warnings,
    )


def render_text_blocks(
    image: Image.Image,
    blocks: list[TextBlock],
) -> tuple[Image.Image, list[TextRenderResult]]:
    """Render multiple TextBlocks sequentially on one working image copy."""
    working = image.copy()
    results: list[TextRenderResult] = []
    for block in blocks:
        result = render_text_block(working, block)
        results.append(result)
        if result.success and result.rendered_lines:
            working = result.image
    return working, results


def save_localized_image(
    image: Image.Image,
    *,
    output_dir: Path,
    source_image_id: str,
) -> str:
    """Persist a localized image under render_assets atomically."""
    assets_dir = output_dir / source_image_id / "render_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    target_path = assets_dir / "localized_image.png"
    temp_path = target_path.with_suffix(".tmp.png")

    if image.mode == "RGBA":
        image.save(temp_path, format="PNG")
    else:
        image.convert("RGB").save(temp_path, format="PNG")
    temp_path.replace(target_path)
    return _relative_to_project(target_path)


def resolve_localized_image_path(localized_image_path: str | Path) -> Path:
    """Resolve a stored localized image path relative to PROJECT_ROOT."""
    path = Path(localized_image_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def render_localized_image(
    original_image: Image.Image,
    blocks: list[TextBlock],
    *,
    masks: dict[str, np.ndarray] | None = None,
) -> tuple[Image.Image, list[BackgroundRestoreResult], list[TextRenderResult]]:
    """End-to-end Renderer API: restore backgrounds, then draw translated text."""
    clean_image, _restore_results = restore_backgrounds(original_image, blocks, masks=masks)
    localized_image, render_results = render_text_blocks(clean_image, blocks)
    return localized_image, _restore_results, render_results
