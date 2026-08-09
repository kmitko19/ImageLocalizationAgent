"""EasyOCR provider — локальная детекция текстовых областей и первичный OCR.

Returns bbox, polygon, raw_text and confidence for each block.
Geometry from this provider is the single authoritative source of
coordinates in the pipeline.

Reading order is encoded as list order and sequential block IDs
(block_001, block_002, …) sorted top-to-bottom, left-to-right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from imageloc.models.raw_blocks import BoundingBox, Point, RawOCRBlock

EasyOCRResult = tuple[list[list[float]], str, float]


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _points_to_bbox(polygon: Sequence[Sequence[float]]) -> BoundingBox:
    xs = [int(round(point[0])) for point in polygon]
    ys = [int(round(point[1])) for point in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return BoundingBox(
        x=x_min,
        y=y_min,
        width=max(1, x_max - x_min),
        height=max(1, y_max - y_min),
    )


def _points_to_polygon(polygon: Sequence[Sequence[float]]) -> list[Point]:
    return [(int(round(point[0])), int(round(point[1]))) for point in polygon]


def _reading_order_key(result: EasyOCRResult) -> tuple[float, float]:
    polygon, _, _ = result
    center_x = sum(point[0] for point in polygon) / len(polygon)
    center_y = sum(point[1] for point in polygon) / len(polygon)
    return (center_y, center_x)


def parse_easyocr_results(results: list[EasyOCRResult]) -> list[RawOCRBlock]:
    """Convert raw EasyOCR readtext() output into sorted RawOCRBlock list."""
    sorted_results = sorted(results, key=_reading_order_key)

    blocks: list[RawOCRBlock] = []
    block_index = 0
    for polygon, text, confidence in sorted_results:
        cleaned_text = text.strip()
        if not cleaned_text:
            continue

        block_index += 1
        blocks.append(
            RawOCRBlock(
                id=f"block_{block_index:03d}",
                raw_text=cleaned_text,
                confidence=_clamp_confidence(confidence),
                bbox=_points_to_bbox(polygon),
                polygon=_points_to_polygon(polygon),
            )
        )

    return blocks


class EasyOCRProvider:
    """Local OCR provider backed by EasyOCR.

    Implements OCRProvider.detect(image). Use detect_from_path() when you
    have a file path instead of a PIL image.
    """

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        self._languages = languages or ["ru", "en"]
        self._gpu = gpu
        self._reader: Any | None = None

    def _get_reader(self) -> Any:
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        return self._reader

    def detect(self, image: Image.Image) -> list[RawOCRBlock]:
        """Run EasyOCR on a PIL image and return detected text blocks."""
        reader = self._get_reader()
        image_array = np.array(image.convert("RGB"))
        raw_results: list[EasyOCRResult] = reader.readtext(image_array)
        return parse_easyocr_results(raw_results)

    def detect_from_path(self, image_path: str | Path) -> list[RawOCRBlock]:
        """Load an image from disk and run detect()."""
        with Image.open(image_path) as image:
            image.load()
            return self.detect(image.convert("RGB"))
