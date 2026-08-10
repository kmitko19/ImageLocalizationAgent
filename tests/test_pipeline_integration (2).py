"""Full pipeline integration test — image → orchestrator → JSON export.

Uses mock OCR, Vision and Translation providers; no real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from imageloc.config import FusionSettings
from imageloc.models.raw_blocks import RawVisionBlock
from imageloc.pipeline.orchestrator import PipelineOrchestrator
from imageloc.providers.easyocr_provider import EasyOCRProvider
from imageloc.providers.openai_translation_provider import OpenAITranslationProvider
from imageloc.utils.json_export import load_result, save_result, validate_json_export

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
        alignment="center",
    ),
    RawVisionBlock(
        ocr_block_id="block_002",
        text="Скидка 50%",
        corrected_text="Скидка 50%",
        semantic_role="label",
        language="RU",
        confidence=0.92,
        alignment="left",
    ),
]

TRANSLATION_RESPONSE_JSON = json.dumps(
    {
        "translations": [
            {
                "id": "block_001",
                "original_text": "Новая коллекция",
                "translated_text": "New Collection",
                "detected_language": "RU",
                "translation_confidence": 0.94,
                "render_notes": "Short heading; fits one line in 200px width.",
            },
            {
                "id": "block_002",
                "original_text": "Скидка 50%",
                "translated_text": "50% Off",
                "detected_language": "RU",
                "translation_confidence": 0.91,
                "render_notes": "Label shortened to fit narrow bbox.",
            },
        ]
    },
    ensure_ascii=False,
)


def _make_pipeline_image(path) -> None:
    """Synthetic image with dark text regions on a light background."""
    image = Image.new("RGB", (400, 200), color=(240, 230, 220))
    pixels = image.load()
    regions = [
        (10, 20, 200, 60),
        (10, 100, 150, 40),
    ]
    for x, y, w, h in regions:
        for row in range(y, y + h):
            for col in range(x, x + w):
                pixels[col, row] = (20, 10, 5)
    image.save(path)


class MockVisionProvider:
    def analyze(self, image, ocr_blocks):
        return MOCK_VISION_BLOCKS


def _mock_openai_client(response_content: str) -> MagicMock:
    client = MagicMock()
    message = MagicMock()
    message.content = response_content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


@patch("easyocr.Reader")
def test_full_pipeline_from_image_to_json(mock_reader_cls: MagicMock, tmp_path):
    image_path = tmp_path / "catalog_page.png"
    _make_pipeline_image(image_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    translation_client = _mock_openai_client(TRANSLATION_RESPONSE_JSON)
    orchestrator = PipelineOrchestrator(
        ocr_provider=EasyOCRProvider(languages=["ru"]),
        vision_provider=MockVisionProvider(),
        translation_provider=OpenAITranslationProvider(api_key="test-key", client=translation_client),
        fusion_settings=FUSION_SETTINGS,
        output_dir=output_dir,
    )

    result = orchestrator.run(
        image_path,
        target_language="EN",
        source_language="RU",
        is_test_case=True,
    )

    json_path = save_result(result, output_dir)
    restored = load_result(json_path)
    exported = validate_json_export(result)

    assert result.version == "1.2"
    assert len(result.text_blocks) == 2
    assert result.source_language == "RU"
    assert result.target_language == "EN"
    assert result.stats.total_blocks == 2
    assert result.stats.ocr_blocks == 2
    assert result.stats.vision_blocks == 2

    heading = result.text_blocks[0]
    assert heading.id == "block_001"
    assert heading.original_text == "Новая коллекция"
    assert heading.translated_text == "New Collection"
    assert heading.confidence == pytest.approx(0.72)
    assert heading.bbox.width == 200
    assert heading.bbox.height == 60
    assert heading.ocr.raw_text == "Новая коллекцiя"
    assert heading.vision is not None
    assert heading.vision.semantic_role == "heading"
    assert heading.visual is not None
    assert heading.visual.estimated_text_height_px is not None
    assert heading.visual.text_color is not None
    assert heading.render_hints.available_width_px == 200
    assert heading.render_hints.available_height_px == 60
    assert heading.render_hints.original_char_count == len("Новая коллекция")
    assert heading.render_hints.translated_char_count == len("New Collection")
    assert heading.render_hints.length_ratio == pytest.approx(
        len("New Collection") / len("Новая коллекция")
    )
    assert heading.render_hints.max_line_count == max(
        heading.render_hints.target_line_count,
        heading.render_hints.estimated_lines,
    )
    assert heading.background is not None
    assert heading.background.dominant_color == heading.visual.background_color
    assert heading.background.background_type in {"solid", "unknown"}
    if heading.background.background_type == "solid":
        assert heading.background.classification_confidence >= 0.75
        assert heading.background.preferred_restore_method == "fill"
        assert heading.background.inpaint_required is False
    assert heading.visual.text_transform in {"normal", "uppercase", "lowercase"}
    assert heading.review.requires_review is True
    assert "decorative_text" in heading.review.reasons
    assert heading.review.user_confirmed is None

    label = result.text_blocks[1]
    assert label.id == "block_002"
    assert label.original_text == "Скидка 50%"
    assert label.translated_text == "50% Off"
    assert label.vision.semantic_role == "label"
    assert label.render_hints.available_width_px == 150
    assert label.review.requires_review is False
    assert label.review.user_confirmed is True
    assert label.review.user_action == "approved"

    assert restored == result
    assert json_path.exists()
    assert json_path.parent.name == result.source_image_id

    block_export = exported["text_blocks"][0]
    assert block_export["original_text"] == "Новая коллекция"
    assert block_export["translated_text"] == "New Collection"
    assert block_export["bbox"]["width"] == 200
    assert block_export["visual"]["text_color"] is not None
    assert block_export["render_hints"]["available_width_px"] == 200
    assert block_export["confidence"] == pytest.approx(0.72)

    json_text = json_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in json_text
    assert "test-key" not in json_text

    mock_reader.readtext.assert_called_once()
    translation_client.chat.completions.create.assert_called_once()


HEART_OCR_RESULTS = [
    (
        [[135.0, 549.0], [267.0, 549.0], [267.0, 709.0], [135.0, 709.0]],
        "~",
        0.262,
    ),
]

HEART_VISION_BLOCKS = [
    RawVisionBlock(
        ocr_block_id="block_001",
        text="~",
        corrected_text="~",
        semantic_role="other",
        language=None,
        confidence=0.99,
    ),
]

HEART_TRANSLATION_JSON = json.dumps({"translations": []}, ensure_ascii=False)


@patch("easyocr.Reader")
def test_orchestrator_skips_translation_for_probable_non_text(mock_reader_cls: MagicMock, tmp_path):
    image_path = tmp_path / "heart_icon.png"
    Image.new("RGB", (400, 800), color="white").save(image_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = HEART_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    translation_client = _mock_openai_client(HEART_TRANSLATION_JSON)
    orchestrator = PipelineOrchestrator(
        ocr_provider=EasyOCRProvider(languages=["ru"]),
        vision_provider=MockVisionProvider(),
        translation_provider=OpenAITranslationProvider(api_key="test-key", client=translation_client),
        fusion_settings=FUSION_SETTINGS,
        output_dir=output_dir,
    )

    class HeartVisionProvider(MockVisionProvider):
        def analyze(self, image, ocr_blocks):
            return HEART_VISION_BLOCKS

    orchestrator.vision_provider = HeartVisionProvider()

    result = orchestrator.run(image_path, target_language="EN", is_test_case=True)

    assert len(result.text_blocks) == 1
    block = result.text_blocks[0]
    assert block.original_text == "~"
    assert block.translated_text is None
    assert "probable_non_text" in block.review.reasons
    translation_client.chat.completions.create.assert_not_called()
