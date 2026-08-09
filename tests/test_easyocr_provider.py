"""Unit tests for EasyOCRProvider — no real OCR model loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from imageloc.models.raw_blocks import RawOCRBlock
from imageloc.providers.easyocr_provider import EasyOCRProvider, parse_easyocr_results


SAMPLE_RESULTS = [
    (
        [[10.0, 200.0], [110.0, 200.0], [110.0, 240.0], [10.0, 240.0]],
        "Нижний текст",
        0.81,
    ),
    (
        [[10.0, 20.0], [210.0, 20.0], [210.0, 80.0], [10.0, 80.0]],
        "Заголовок",
        0.93,
    ),
    (
        [[10.0, 100.0], [160.0, 100.0], [160.0, 140.0], [10.0, 140.0]],
        "Подпись",
        0.77,
    ),
]


def test_parse_easyocr_results_builds_blocks_with_bbox_and_confidence():
    blocks = parse_easyocr_results(SAMPLE_RESULTS)

    assert len(blocks) == 3
    assert blocks[0].raw_text == "Заголовок"
    assert blocks[0].confidence == pytest.approx(0.93)
    assert blocks[0].bbox.x == 10
    assert blocks[0].bbox.y == 20
    assert blocks[0].bbox.width == 200
    assert blocks[0].bbox.height == 60


def test_parse_easyocr_results_sorts_by_reading_order_top_to_bottom():
    blocks = parse_easyocr_results(SAMPLE_RESULTS)

    assert [block.raw_text for block in blocks] == [
        "Заголовок",
        "Подпись",
        "Нижний текст",
    ]
    assert [block.id for block in blocks] == ["block_001", "block_002", "block_003"]


def test_parse_easyocr_results_preserves_polygon_points():
    blocks = parse_easyocr_results(SAMPLE_RESULTS)

    assert blocks[0].polygon == [(10, 20), (210, 20), (210, 80), (10, 80)]


def test_parse_easyocr_results_skips_empty_text():
    results = [
        (
            [[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]],
            "   ",
            0.5,
        ),
        (
            [[60.0, 0.0], [120.0, 0.0], [120.0, 20.0], [60.0, 20.0]],
            "Есть текст",
            0.6,
        ),
    ]

    blocks = parse_easyocr_results(results)

    assert len(blocks) == 1
    assert blocks[0].raw_text == "Есть текст"
    assert blocks[0].id == "block_001"


@patch("easyocr.Reader")
def test_detect_uses_mocked_reader_without_loading_models(mock_reader_cls: MagicMock):
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = SAMPLE_RESULTS
    mock_reader_cls.return_value = mock_reader

    provider = EasyOCRProvider(languages=["ru"])
    image = Image.new("RGB", (300, 300), color="white")

    blocks = provider.detect(image)

    mock_reader_cls.assert_called_once_with(["ru"], gpu=False)
    mock_reader.readtext.assert_called_once()
    assert len(blocks) == 3
    assert isinstance(blocks[0], RawOCRBlock)
    assert blocks[0].raw_text == "Заголовок"


@patch("easyocr.Reader")
def test_detect_from_path_loads_image_and_delegates_to_detect(
    mock_reader_cls: MagicMock, tmp_path
):
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = SAMPLE_RESULTS
    mock_reader_cls.return_value = mock_reader

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (300, 300), color="white").save(image_path)

    provider = EasyOCRProvider()
    blocks = provider.detect_from_path(image_path)

    mock_reader.readtext.assert_called_once()
    assert len(blocks) == 3
    assert blocks[-1].raw_text == "Нижний текст"
