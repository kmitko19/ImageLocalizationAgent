"""Unit tests for OpenAIVisionProvider — no real API calls, no API key."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from PIL import Image

from imageloc.models.raw_blocks import BoundingBox, RawOCRBlock, RawVisionBlock
from imageloc.providers.openai_vision_provider import (
    OpenAIVisionProvider,
    VISION_SCOPE_FALLBACK_CONFIDENCE,
    apply_anchor_scope_guard,
    build_messages,
    correction_similarity,
    is_plausible_anchor_correction,
    normalize_vision_blocks,
    parse_vision_response,
)

SAMPLE_OCR_BLOCKS = [
    RawOCRBlock(
        id="block_001",
        raw_text="Новая коллекцiя",
        confidence=0.72,
        bbox=BoundingBox(x=10, y=20, width=200, height=60),
        polygon=[(10, 20), (210, 20), (210, 80), (10, 80)],
    ),
    RawOCRBlock(
        id="block_002",
        raw_text="Скидка 50%",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=100, width=150, height=40),
        polygon=[(10, 100), (160, 100), (160, 140), (10, 140)],
    ),
]

SAMPLE_RESPONSE_JSON = json.dumps(
    {
        "blocks": [
            {
                "ocr_block_id": "block_001",
                "text": "Новая коллекцiя",
                "corrected_text": "Новая коллекция",
                "semantic_role": "heading",
                "language": "RU",
                "font_category": "decorative_script",
                "font_style": "italic",
                "font_weight": "regular",
                "alignment": "center",
                "confidence": 0.94,
                "bbox": None,
            },
            {
                "ocr_block_id": "block_002",
                "text": "Скидка 50%",
                "corrected_text": "Скидка 50%",
                "semantic_role": "label",
                "language": "RU",
                "font_category": None,
                "font_style": None,
                "font_weight": None,
                "alignment": None,
                "confidence": 0.92,
                "bbox": {"x": 10, "y": 100, "width": 150, "height": 40},
            },
        ]
    },
    ensure_ascii=False,
)


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


def test_parse_vision_response_builds_raw_vision_blocks():
    blocks = parse_vision_response(SAMPLE_RESPONSE_JSON)

    assert len(blocks) == 2
    assert all(isinstance(b, RawVisionBlock) for b in blocks)

    heading = blocks[0]
    assert heading.ocr_block_id == "block_001"
    assert heading.corrected_text == "Новая коллекция"
    assert heading.semantic_role == "heading"
    assert heading.language == "RU"
    assert heading.font_category == "decorative_script"
    assert heading.confidence == pytest.approx(0.94)
    assert heading.bbox is None


def test_parse_vision_response_includes_optional_bbox_when_present():
    blocks = parse_vision_response(SAMPLE_RESPONSE_JSON)

    label = blocks[1]
    assert label.bbox is not None
    assert label.bbox.x == 10
    assert label.bbox.width == 150


def test_parse_vision_response_skips_malformed_entries():
    content = json.dumps(
        {
            "blocks": [
                {"not_a_valid_block": True},
                {
                    "ocr_block_id": "block_002",
                    "text": "Скидка 50%",
                    "corrected_text": "Скидка 50%",
                    "semantic_role": "label",
                    "language": "RU",
                    "font_category": None,
                    "font_style": None,
                    "font_weight": None,
                    "alignment": None,
                    "confidence": 0.9,
                    "bbox": None,
                },
            ]
        }
    )

    blocks = parse_vision_response(content)

    assert len(blocks) == 1
    assert blocks[0].ocr_block_id == "block_002"


def test_parse_vision_response_raises_clear_error_on_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_vision_response("not json at all")


def test_parse_vision_response_handles_empty_blocks_list():
    assert parse_vision_response(json.dumps({"blocks": []})) == []


def test_build_messages_includes_ocr_anchors_and_image():
    image = Image.new("RGB", (300, 300), color="white")

    messages = build_messages(image, SAMPLE_OCR_BLOCKS)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    user_content = messages[1]["content"]
    text_part = next(part for part in user_content if part["type"] == "text")
    image_part = next(part for part in user_content if part["type"] == "image_url")

    assert "block_001" in text_part["text"]
    assert "Новая коллекцiя" in text_part["text"]
    assert "one Vision block per anchor" in text_part["text"]
    assert "Do not merge anchors" in text_part["text"]
    assert "corrected_text must come only from that anchor's bbox" in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_messages_with_no_ocr_blocks():
    image = Image.new("RGB", (100, 100), color="white")

    messages = build_messages(image, [])

    text_part = next(part for part in messages[1]["content"] if part["type"] == "text")
    assert "[]" in text_part["text"]
    assert "one Vision block per anchor" in text_part["text"]


def test_normalize_vision_blocks_keeps_best_block_per_anchor():
    blocks = [
        RawVisionBlock(
            ocr_block_id="block_001",
            text="low",
            corrected_text="low",
            semantic_role="label",
            confidence=0.4,
        ),
        RawVisionBlock(
            ocr_block_id="block_001",
            text="high",
            corrected_text="Новая коллекция",
            semantic_role="label",
            confidence=0.95,
        ),
        RawVisionBlock(
            ocr_block_id="block_002",
            text="Скидка 50%",
            corrected_text="Скидка 50%",
            semantic_role="label",
            language="RU",
            confidence=0.9,
        ),
        RawVisionBlock(
            ocr_block_id="unknown",
            text="extra",
            corrected_text="extra",
            semantic_role="paragraph",
            confidence=0.8,
        ),
    ]

    normalized = normalize_vision_blocks(blocks, SAMPLE_OCR_BLOCKS)

    anchor_blocks = [block for block in normalized if block.ocr_block_id is not None]
    unmatched = [block for block in normalized if block.ocr_block_id is None]

    assert len(anchor_blocks) == 2
    assert anchor_blocks[0].ocr_block_id == "block_001"
    assert anchor_blocks[0].corrected_text == "Новая коллекция"
    assert len(unmatched) == 1
    assert unmatched[0].ocr_block_id is None


def test_is_plausible_anchor_correction_accepts_local_typo_fix():
    assert is_plausible_anchor_correction("Биречь", "Беречь") is True
    assert correction_similarity("Биречь", "Беречь") >= 0.8


def test_is_plausible_anchor_correction_rejects_neighbor_paragraph():
    ocr_text = "шербне"
    neighbor_text = "между собой над эскизами. Этот кропотливый и пло-"

    assert is_plausible_anchor_correction(ocr_text, neighbor_text) is False


def test_is_plausible_anchor_correction_accepts_unreadable_fallback_to_raw():
    unreadable = "gЧкg■ чinerчvgрu"
    assert is_plausible_anchor_correction(unreadable, unreadable) is True


def test_apply_anchor_scope_guard_falls_back_to_ocr_raw_text():
    ocr = SAMPLE_OCR_BLOCKS[0]
    vision = RawVisionBlock(
        ocr_block_id="block_001",
        text="ignored",
        corrected_text="между собой над эскизами",
        semantic_role="paragraph",
        language="ru",
        confidence=0.98,
    )

    guarded = apply_anchor_scope_guard(vision, ocr)

    assert guarded.corrected_text == ocr.raw_text
    assert guarded.text == ocr.raw_text
    assert guarded.confidence == VISION_SCOPE_FALLBACK_CONFIDENCE


def test_normalize_rejects_neighbor_scope_bleed_while_preserving_anchor_ids():
    blocks = [
        RawVisionBlock(
            ocr_block_id="block_001",
            text="wrong",
            corrected_text="между собой над эскизами",
            semantic_role="paragraph",
            language="ru",
            confidence=0.98,
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

    normalized = normalize_vision_blocks(blocks, SAMPLE_OCR_BLOCKS)
    by_id = {block.ocr_block_id: block for block in normalized if block.ocr_block_id}

    assert set(by_id) == {"block_001", "block_002"}
    assert by_id["block_001"].corrected_text == SAMPLE_OCR_BLOCKS[0].raw_text
    assert by_id["block_001"].confidence == VISION_SCOPE_FALLBACK_CONFIDENCE
    assert by_id["block_002"].corrected_text == "Скидка 50%"


def test_analyze_uses_injected_mock_client_without_real_api_call():
    mock_client = _mock_openai_client(SAMPLE_RESPONSE_JSON)
    provider = OpenAIVisionProvider(api_key="test-key-not-real", model="gpt-4o", client=mock_client)
    image = Image.new("RGB", (300, 300), color="white")

    blocks = provider.analyze(image, SAMPLE_OCR_BLOCKS)

    assert len(blocks) == 2
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["response_format"]["type"] == "json_schema"


def test_analyze_returns_empty_list_when_model_returns_no_content():
    mock_client = _mock_openai_client(None)
    provider = OpenAIVisionProvider(api_key="test-key-not-real", client=mock_client)
    image = Image.new("RGB", (100, 100), color="white")

    blocks = provider.analyze(image, [])

    assert blocks == []


def test_analyze_wraps_client_errors_and_sanitizes_message(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-value")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError(
        "Auth failed for key sk-super-secret-value"
    )
    provider = OpenAIVisionProvider(api_key="sk-super-secret-value", client=mock_client)
    image = Image.new("RGB", (100, 100), color="white")

    with pytest.raises(RuntimeError) as exc_info:
        provider.analyze(image, [])

    assert "sk-super-secret-value" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_provider_never_hardcodes_or_reads_env_for_api_key():
    # The provider must receive the key explicitly; it must not read
    # OPENAI_API_KEY itself (that responsibility belongs to imageloc.config).
    import inspect

    from imageloc.providers import openai_vision_provider as module

    source = inspect.getsource(module)
    assert "os.environ" not in source
    assert "getenv" not in source
