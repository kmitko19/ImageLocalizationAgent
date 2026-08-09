"""Unit tests for deterministic background analysis."""

from __future__ import annotations

import math
import warnings

import numpy as np
from PIL import Image

from imageloc.models.raw_blocks import BoundingBox
from imageloc.utils.background_analyzer import _safe_correlation, analyze_background


def _solid_image(size=(200, 120), color=(240, 230, 220)) -> np.ndarray:
    image = Image.new("RGB", size, color=color)
    return np.array(image)


def _gradient_image(size=(200, 120)) -> np.ndarray:
    width, height = size
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        value = int(255 * y / max(1, height - 1))
        gradient[y, :, :] = (value, value // 2, 255 - value)
    return gradient


def _texture_image(size=(200, 120), seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return noise


def test_solid_background_classification():
    image = _solid_image()
    bbox = BoundingBox(x=60, y=40, width=80, height=30)

    background = analyze_background(image, bbox)

    assert background is not None
    assert background.background_type == "solid"
    assert background.classification_confidence >= 0.75
    assert background.preferred_restore_method == "fill"
    assert background.inpaint_required is False
    assert background.dominant_color is not None


def test_gradient_background_classification():
    image = _gradient_image()
    bbox = BoundingBox(x=60, y=40, width=80, height=30)

    background = analyze_background(image, bbox)

    assert background is not None
    assert background.background_type == "gradient"
    assert background.preferred_restore_method == "gradient"
    assert background.inpaint_required is False


def test_texture_background_is_not_classified_as_solid():
    image = _texture_image()
    bbox = BoundingBox(x=60, y=40, width=80, height=30)

    background = analyze_background(image, bbox)

    assert background is not None
    assert background.background_type != "solid"


def test_complex_patch_does_not_raise():
    image = _texture_image(seed=99)
    noisy = image.copy()
    noisy[20:100, 20:180] = np.clip(noisy[20:100, 20:180].astype(np.int16) + 80, 0, 255).astype(np.uint8)
    bbox = BoundingBox(x=70, y=45, width=60, height=30)

    background = analyze_background(noisy, bbox)

    assert background is not None
    assert background.background_type in {"texture", "photo", "complex", "unknown"}


def test_tiny_bbox_falls_back_to_unknown():
    image = _solid_image()
    bbox = BoundingBox(x=196, y=116, width=4, height=4)

    background = analyze_background(image, bbox, dominant_color_hint="#F0E6DC")

    assert background is not None
    assert background.background_type == "unknown"
    assert background.classification_confidence <= 0.30
    assert background.preferred_restore_method == "auto"


def _collect_runtime_warnings(callable_obj):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = callable_obj()
    runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    return result, runtime_warnings


def test_uniform_patch_has_no_corrcoef_runtime_warning():
    image = _solid_image()
    bbox = BoundingBox(x=60, y=40, width=80, height=30)

    background, runtime_warnings = _collect_runtime_warnings(
        lambda: analyze_background(image, bbox)
    )

    assert not runtime_warnings
    assert background is not None
    assert background.background_type == "solid"
    assert math.isfinite(background.classification_confidence)
    assert math.isfinite(background.complexity_score)


def test_safe_correlation_zero_variance_is_finite_without_warning():
    axis = np.arange(10, dtype=np.float32)
    constant_values = np.full(10, 128.0, dtype=np.float32)

    result, runtime_warnings = _collect_runtime_warnings(
        lambda: _safe_correlation(axis, constant_values)
    )

    assert not runtime_warnings
    assert result == 0.0
    assert math.isfinite(result)


def test_gradient_background_still_detected_without_corrcoef_warning():
    image = _gradient_image()
    bbox = BoundingBox(x=60, y=40, width=80, height=30)

    background, runtime_warnings = _collect_runtime_warnings(
        lambda: analyze_background(image, bbox)
    )

    assert not runtime_warnings
    assert background is not None
    assert background.background_type == "gradient"
    assert background.preferred_restore_method == "gradient"
