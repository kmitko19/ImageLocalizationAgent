"""Renderer-layer modules for background restoration and text drawing."""

from imageloc.renderer.background_restorer import (
    BackgroundRestoreResult,
    restore_backgrounds,
    restore_text_background,
    save_clean_background,
    select_restore_method,
)
from imageloc.renderer.text_effects import (
    EffectApplicationResult,
    TextLayoutContext,
    apply_text_effects,
    render_glyph_alpha_mask,
    resolve_perspective_target_quad,
    validate_perspective_quad,
)
from imageloc.renderer.text_renderer import (
    TextRenderResult,
    compute_content_box,
    compute_content_padding_px,
    render_localized_image,
    render_text_block,
    render_text_blocks,
    resolve_font,
    resolve_localized_image_path,
    save_localized_image,
    wrap_text_lines,
)

__all__ = [
    "BackgroundRestoreResult",
    "EffectApplicationResult",
    "TextLayoutContext",
    "TextRenderResult",
    "apply_text_effects",
    "compute_content_box",
    "compute_content_padding_px",
    "render_glyph_alpha_mask",
    "render_localized_image",
    "render_text_block",
    "render_text_blocks",
    "resolve_font",
    "resolve_localized_image_path",
    "resolve_perspective_target_quad",
    "restore_backgrounds",
    "restore_text_background",
    "save_clean_background",
    "save_localized_image",
    "select_restore_method",
    "validate_perspective_quad",
    "wrap_text_lines",
]
