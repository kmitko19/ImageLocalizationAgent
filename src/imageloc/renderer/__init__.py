"""Renderer-layer modules for background restoration and future text drawing."""

from imageloc.renderer.background_restorer import (
    BackgroundRestoreResult,
    restore_backgrounds,
    restore_text_background,
    save_clean_background,
    sample_solid_fill_color,
    select_restore_method,
)

__all__ = [
    "BackgroundRestoreResult",
    "restore_backgrounds",
    "restore_text_background",
    "save_clean_background",
    "select_restore_method",
]
