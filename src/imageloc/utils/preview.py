"""Preview rendering — draw bbox outlines and block IDs on the source image."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from imageloc.models.raw_blocks import BoundingBox


class PreviewBlock(Protocol):
    id: str
    bbox: BoundingBox


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk for preview rendering."""
    with Image.open(image_path) as image:
        image.load()
        return image.convert("RGB")


def draw_preview(
    image: Image.Image,
    blocks: Sequence[PreviewBlock],
    *,
    outline_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2,
) -> Image.Image:
    """Draw bounding boxes and block IDs on a copy of the image."""
    rgb = np.array(image.convert("RGB"))
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for block in blocks:
        bbox = block.bbox
        top_left = (bbox.x, bbox.y)
        bottom_right = (bbox.x + bbox.width, bbox.y + bbox.height)
        cv2.rectangle(canvas, top_left, bottom_right, outline_color, line_thickness)

        label_y = max(bbox.y - 6, 12)
        cv2.putText(
            canvas,
            block.id,
            (bbox.x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )

    result_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def save_preview(
    image: Image.Image,
    blocks: Sequence[PreviewBlock],
    output_path: str | Path,
    *,
    outline_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (0, 255, 0),
) -> Path:
    """Render preview with bbox overlays and save to disk."""
    preview = draw_preview(
        image,
        blocks,
        outline_color=outline_color,
        text_color=text_color,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path)
    return path


def save_preview_from_path(
    image_path: str | Path,
    blocks: Sequence[PreviewBlock],
    output_path: str | Path,
) -> Path:
    """Load source image, draw preview overlays, and save the result."""
    image = load_image(image_path)
    return save_preview(image, blocks, output_path)
