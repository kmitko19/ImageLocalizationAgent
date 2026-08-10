"""Deterministic post-translation render metadata finalization.

Populates renderer-oriented fields that can be derived from existing
TextBlock data without AI, image segmentation, or font recognition.
"""

from __future__ import annotations

from imageloc.models.text_block import (
    BackgroundInfo,
    RenderGeometry,
    RenderHints,
    TextBlock,
    VisualStyle,
)

DEFAULT_WRITING_DIRECTION = "horizontal"


def _char_count(text: str | None) -> int:
    if not text:
        return 0
    return len(text)


def _length_ratio(original_count: int, translated_count: int) -> float | None:
    if original_count == 0:
        return None
    return translated_count / original_count


def finalize_render_hints(
    render_hints: RenderHints,
    *,
    original_text: str | None,
    translated_text: str | None,
) -> RenderHints:
    """Fill translation-dependent render hint fields after text is known."""
    original_char_count = _char_count(original_text)
    translated_char_count = _char_count(translated_text)
    length_ratio = _length_ratio(original_char_count, translated_char_count)
    max_line_count = max(render_hints.target_line_count, render_hints.estimated_lines)

    return render_hints.model_copy(
        update={
            "original_char_count": original_char_count,
            "translated_char_count": translated_char_count,
            "length_ratio": length_ratio,
            "max_line_count": max_line_count,
        }
    )


def build_render_geometry(
    visual: VisualStyle,
    render_hints: RenderHints,
) -> RenderGeometry | None:
    """Build render geometry only when non-default direction or rotation exists."""
    writing_direction = render_hints.text_direction
    has_non_default_direction = writing_direction != DEFAULT_WRITING_DIRECTION
    has_rotation = visual.rotation_deg != 0.0

    if not has_non_default_direction and not has_rotation:
        return None

    return RenderGeometry(
        baseline_angle_deg=visual.rotation_deg if has_rotation else None,
        writing_direction=writing_direction,
    )


def build_background_info(visual: VisualStyle) -> BackgroundInfo | None:
    """Initialize background metadata from a sampled background color only."""
    if not visual.background_color:
        return None

    return BackgroundInfo(dominant_color=visual.background_color)


def finalize_text_block_metadata(block: TextBlock) -> TextBlock:
    """Apply all deterministic post-translation render metadata to one block."""
    render_hints = finalize_render_hints(
        block.render_hints,
        original_text=block.original_text,
        translated_text=block.translated_text,
    )
    render_geometry = (
        block.render_geometry
        if block.render_geometry is not None
        else build_render_geometry(block.visual, render_hints)
    )
    background = block.background if block.background is not None else build_background_info(block.visual)

    return block.model_copy(
        update={
            "render_hints": render_hints,
            "render_geometry": render_geometry,
            "background": background,
        }
    )


def refresh_translation_dependent_render_hints(block: TextBlock) -> TextBlock:
    """Recalculate render_hints fields that depend on current block texts."""
    return block.model_copy(
        update={
            "render_hints": finalize_render_hints(
                block.render_hints,
                original_text=block.original_text,
                translated_text=block.translated_text,
            )
        }
    )
