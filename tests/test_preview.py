"""Unit tests for preview rendering — mock OpenCV drawing where useful."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from imageloc.models.raw_blocks import BoundingBox
from imageloc.utils.preview import draw_preview, load_image, save_preview, save_preview_from_path


@dataclass
class _PreviewBlock:
    id: str
    bbox: BoundingBox


SAMPLE_BLOCKS = [
    _PreviewBlock(id="block_001", bbox=BoundingBox(x=10, y=20, width=80, height=30)),
    _PreviewBlock(id="block_002", bbox=BoundingBox(x=10, y=70, width=120, height=25)),
]


def test_draw_preview_returns_image_with_same_dimensions():
    image = Image.new("RGB", (200, 120), color="white")

    preview = draw_preview(image, SAMPLE_BLOCKS)

    assert isinstance(preview, Image.Image)
    assert preview.size == (200, 120)


@patch("imageloc.utils.preview.cv2.rectangle")
@patch("imageloc.utils.preview.cv2.putText")
def test_draw_preview_draws_each_block(mock_put_text: MagicMock, mock_rectangle: MagicMock):
    image = Image.new("RGB", (200, 120), color="white")

    draw_preview(image, SAMPLE_BLOCKS)

    assert mock_rectangle.call_count == 2
    assert mock_put_text.call_count == 2
    assert mock_put_text.call_args_list[0].args[1] == "block_001"
    assert mock_put_text.call_args_list[1].args[1] == "block_002"


def test_save_preview_writes_file(tmp_path):
    image = Image.new("RGB", (200, 120), color="white")
    output_path = tmp_path / "preview.png"

    saved_path = save_preview(image, SAMPLE_BLOCKS, output_path)

    assert saved_path.exists()
    with Image.open(saved_path) as saved_image:
        assert saved_image.size == (200, 120)


def test_save_preview_from_path_loads_image_and_saves_result(tmp_path):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "preview.png"
    Image.new("RGB", (180, 90), color="white").save(source_path)

    saved_path = save_preview_from_path(source_path, SAMPLE_BLOCKS, output_path)

    assert saved_path.exists()
    loaded = load_image(source_path)
    assert loaded.size == (180, 90)


@patch("imageloc.utils.preview.draw_preview")
def test_save_preview_delegates_to_draw_preview(mock_draw_preview: MagicMock, tmp_path):
    image = Image.new("RGB", (100, 100), color="white")
    preview_image = Image.new("RGB", (100, 100), color="black")
    mock_draw_preview.return_value = preview_image
    output_path = tmp_path / "out.png"

    save_preview(image, SAMPLE_BLOCKS, output_path)

    mock_draw_preview.assert_called_once_with(image, SAMPLE_BLOCKS, outline_color=(0, 255, 0), text_color=(0, 255, 0))
    assert output_path.exists()
