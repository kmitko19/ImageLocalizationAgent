"""Renderer-layer modules for background restoration and text drawing."""

from imageloc.renderer.background_restorer import (
    BackgroundRestoreResult,
    restore_backgrounds,
    restore_text_background,
    save_clean_background,
    select_restore_method,
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
    "TextRenderResult",
    "compute_content_box",
    "compute_content_padding_px",
    "render_localized_image",
    "render_text_block",
    "render_text_blocks",
    "resolve_font",
    "resolve_localized_image_path",
    "restore_backgrounds",
    "restore_text_background",
    "save_clean_background",
    "save_localized_image",
    "select_restore_method",
    "wrap_text_lines",
]
