"""Integration test for the OCR portion of the pipeline.

Flow under test (no real EasyOCR models, no OpenAI):

    ImageLoader → EasyOCRProvider (mock) → OCRVisionFusion → PipelineResult → JSON
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from imageloc.config import FusionSettings
from imageloc.fusion.ocr_vision_fusion import FusedBlock, FusionResult, fuse
from imageloc.models.raw_blocks import RawVisionBlock
from imageloc.models.result import PipelineResult, Stats
from imageloc.models.text_block import RenderHints, TextBlock, VisualStyle
from imageloc.providers.easyocr_provider import EasyOCRProvider
from imageloc.utils.image_loader import load_image

FUSION_SETTINGS = FusionSettings(
    iou_threshold=0.25,
    text_similarity_threshold=0.55,
    text_mismatch_threshold=0.5,
    low_ocr_confidence_threshold=0.5,
    low_font_size_confidence_threshold=0.4,
)

MOCK_OCR_RESULTS = [
    (
        [[10.0, 20.0], [210.0, 20.0], [210.0, 80.0], [10.0, 80.0]],
        "Новая коллекцiя",
        0.72,
    ),
    (
        [[10.0, 100.0], [160.0, 100.0], [160.0, 140.0], [10.0, 140.0]],
        "Скидка 50%",
        0.91,
    ),
]

MOCK_VISION_BLOCKS = [
    RawVisionBlock(
        ocr_block_id="block_001",
        text="Новая коллекцiя",
        corrected_text="Новая коллекция",
        semantic_role="heading",
        language="RU",
        confidence=0.94,
        font_category="decorative_script",
    ),
    RawVisionBlock(
        ocr_block_id="block_002",
        text="Скидка 50%",
        corrected_text="Скидка 50%",
        semantic_role="label",
        language="RU",
        confidence=0.92,
    ),
]


def _fused_block_to_text_block(fused: FusedBlock) -> TextBlock:
    """Minimal TextBlock assembly for the OCR-only integration slice.

    Visual analysis and translation are stubbed with bbox-derived placeholders
    until their dedicated pipeline stages are implemented.
    """
    return TextBlock(
        id=fused.id,
        original_text=fused.text,
        translated_text=None,
        source_language=fused.source_language,
        confidence=fused.confidence,
        bbox=fused.bbox,
        polygon=fused.polygon,
        ocr=fused.ocr,
        vision=fused.vision,
        visual=VisualStyle(estimated_text_height_px=fused.bbox.height),
        render_hints=RenderHints(
            available_width_px=fused.bbox.width,
            available_height_px=fused.bbox.height,
        ),
        review=fused.review,
    )


def _build_pipeline_result(
    *,
    source_image_id: str,
    source_image,
    fusion_result: FusionResult,
    ocr_block_count: int,
    vision_block_count: int,
    target_language: str = "EN",
) -> PipelineResult:
    text_blocks = [_fused_block_to_text_block(block) for block in fusion_result.fused_blocks]
    blocks_requiring_review = sum(1 for block in text_blocks if block.review.requires_review)
    avg_confidence = (
        sum(block.confidence for block in text_blocks) / len(text_blocks) if text_blocks else 0.0
    )

    return PipelineResult(
        source_image_id=source_image_id,
        source_image=source_image,
        source_language="RU",
        source_language_mode="auto",
        target_language=target_language,
        processed_at=datetime.now(timezone.utc),
        text_blocks=text_blocks,
        unmatched_vision_blocks=fusion_result.unmatched_vision_blocks,
        stats=Stats(
            total_blocks=len(text_blocks),
            blocks_requiring_review=blocks_requiring_review,
            avg_confidence=avg_confidence,
            ocr_blocks=ocr_block_count,
            vision_blocks=vision_block_count,
        ),
    )


@patch("easyocr.Reader")
def test_ocr_pipeline_produces_valid_pipeline_result_and_json(mock_reader_cls: MagicMock, tmp_path):
    # --- Arrange: test image on disk ---
    image_path = tmp_path / "catalog_page.png"
    Image.new("RGB", (400, 200), color="white").save(image_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    # --- Act: ImageLoader → EasyOCR (mock) → Fusion → PipelineResult ---
    loaded = load_image(image_path, output_dir=output_dir, is_test_case=True)

    ocr_provider = EasyOCRProvider(languages=["ru"])
    ocr_blocks = ocr_provider.detect(loaded.image)

    fusion_result = fuse(ocr_blocks, MOCK_VISION_BLOCKS, FUSION_SETTINGS)

    result = _build_pipeline_result(
        source_image_id=loaded.source_image_id,
        source_image=loaded.source_image,
        fusion_result=fusion_result,
        ocr_block_count=len(ocr_blocks),
        vision_block_count=len(MOCK_VISION_BLOCKS),
    )

    json_str = result.model_dump_json(indent=2)
    restored = PipelineResult.model_validate_json(json_str)

    # --- Assert: PipelineResult structure ---
    assert result.version == "1.2"
    assert result.source_image_id == loaded.source_image_id
    assert result.source_image.filename == "catalog_page.png"
    assert result.source_image.width == 400
    assert result.source_image.height == 200
    assert len(result.text_blocks) == 2

    heading = result.text_blocks[0]
    assert heading.id == "block_001"
    assert heading.original_text == "Новая коллекция"
    assert heading.ocr.raw_text == "Новая коллекцiя"
    assert heading.ocr.confidence == pytest.approx(0.72)
    assert heading.vision is not None
    assert heading.vision.corrected_text == "Новая коллекция"
    assert heading.vision.semantic_role == "heading"
    assert heading.bbox.width == 200
    assert heading.bbox.height == 60

    label = result.text_blocks[1]
    assert label.id == "block_002"
    assert label.original_text == "Скидка 50%"
    assert label.vision.semantic_role == "label"

    assert result.stats.total_blocks == 2
    assert result.stats.ocr_blocks == 2
    assert result.stats.vision_blocks == 2
    assert result.stats.blocks_requiring_review >= 1
    assert "decorative_text" in heading.review.reasons

    # --- Assert: JSON round-trip ---
    assert restored == result
    assert "OPENAI_API_KEY" not in json_str
    assert "source_image_id" in json_str
    assert restored.text_blocks[0].review.user_confirmed is None

    mock_reader_cls.assert_called_once_with(["ru"], gpu=False)
    mock_reader.readtext.assert_called_once()
