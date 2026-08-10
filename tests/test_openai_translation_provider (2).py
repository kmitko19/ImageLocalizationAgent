"""Unit tests for OpenAITranslationProvider — no real API calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from imageloc.fusion.ocr_vision_fusion import FusedBlock
from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import OCRData, RenderHints, ReviewInfo, TextBlock, VisualStyle
from imageloc.providers.base import TranslationInput
from imageloc.providers.openai_translation_provider import (
    OpenAITranslationProvider,
    TranslationContext,
    build_messages,
    build_translation_context_from_fused,
    build_translation_context_from_text_block,
    parse_translation_response,
)

SAMPLE_RESPONSE_JSON = json.dumps(
    {
        "translations": [
            {
                "id": "block_001",
                "original_text": "Новая коллекция",
                "translated_text": "New Collection",
                "detected_language": "RU",
                "translation_confidence": 0.94,
                "render_notes": "Short heading; fits one line in 680px width.",
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


def test_parse_translation_response_builds_detail_objects():
    results = parse_translation_response(SAMPLE_RESPONSE_JSON)

    assert len(results) == 2
    assert results[0].original_text == "Новая коллекция"
    assert results[0].translated_text == "New Collection"
    assert results[0].detected_language == "RU"
    assert results[0].translation_confidence == pytest.approx(0.94)
    assert "680px" in (results[0].render_notes or "")


def test_parse_translation_response_raises_on_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_translation_response("not-json")


def test_build_messages_includes_layout_constraints():
    contexts = [
        TranslationContext(
            input=TranslationInput(
                id="block_001",
                text="Новая коллекция",
                semantic_role="heading",
                available_width_px=680,
                available_height_px=95,
                font_category="decorative_script",
            ),
            estimated_lines=1,
            target_line_count=1,
            preserve_layout=True,
        )
    ]

    messages = build_messages(contexts, target_lang="EN", source_lang="RU")

    user_content = messages[1]["content"]
    assert "Target language: EN" in user_content
    assert "Source language: RU" in user_content
    assert "available_width_px" in user_content
    assert "estimated_lines" in user_content
    assert "decorative_script" in user_content


def test_build_translation_context_from_fused_block():
    fused = FusedBlock(
        id="block_001",
        text="Заголовок",
        source_language="RU",
        confidence=0.9,
        bbox=BoundingBox(x=0, y=0, width=200, height=40),
        polygon=[(0, 0), (200, 0), (200, 40), (0, 40)],
        ocr=OCRData(raw_text="Заголовок", confidence=0.9),
        vision=None,
        review=ReviewInfo(),
    )
    hints = RenderHints(available_width_px=200, available_height_px=40, estimated_lines=1)

    ctx = build_translation_context_from_fused(fused, hints)

    assert ctx.input.text == "Заголовок"
    assert ctx.input.available_width_px == 200
    assert ctx.estimated_lines == 1


def test_build_translation_context_from_text_block():
    block = TextBlock(
        id="block_002",
        original_text="Скидка 50%",
        confidence=0.9,
        bbox=BoundingBox(x=0, y=0, width=150, height=30),
        polygon=[(0, 0), (150, 0), (150, 30), (0, 30)],
        ocr=OCRData(raw_text="Скидка 50%", confidence=0.9),
        visual=VisualStyle(font_category="sans_serif"),
        render_hints=RenderHints(
            available_width_px=150,
            available_height_px=30,
            estimated_lines=1,
            target_line_count=1,
        ),
        review=ReviewInfo(),
    )

    ctx = build_translation_context_from_text_block(block)

    assert ctx.input.semantic_role == "other"
    assert ctx.input.font_category == "sans_serif"
    assert ctx.preserve_layout is True


def test_translate_details_uses_mock_client_without_real_api():
    mock_client = _mock_openai_client(SAMPLE_RESPONSE_JSON)
    provider = OpenAITranslationProvider(api_key="test-key", client=mock_client)
    contexts = [
        TranslationContext(
            input=TranslationInput(
                id="block_001",
                text="Новая коллекция",
                semantic_role="heading",
                available_width_px=680,
                available_height_px=95,
            ),
            estimated_lines=1,
        )
    ]

    results = provider.translate_details(contexts, target_lang="EN", source_lang="RU")

    assert len(results) == 2
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"]["type"] == "json_schema"


def test_translate_protocol_returns_translated_block_list():
    mock_client = _mock_openai_client(SAMPLE_RESPONSE_JSON)
    provider = OpenAITranslationProvider(api_key="test-key", client=mock_client)
    blocks = [
        TranslationInput(
            id="block_001",
            text="Новая коллекция",
            semantic_role="heading",
            available_width_px=680,
            available_height_px=95,
        )
    ]

    translated = provider.translate(blocks, target_lang="EN", source_lang="RU")

    assert len(translated) == 2
    assert translated[0].id == "block_001"
    assert translated[0].translated_text == "New Collection"


def test_translate_fused_blocks_passes_render_hints_to_api_payload():
    mock_client = _mock_openai_client(SAMPLE_RESPONSE_JSON)
    provider = OpenAITranslationProvider(api_key="test-key", client=mock_client)
    fused = FusedBlock(
        id="block_001",
        text="Новая коллекция",
        source_language="RU",
        confidence=0.9,
        bbox=BoundingBox(x=0, y=0, width=680, height=95),
        polygon=[(0, 0), (680, 0), (680, 95), (0, 95)],
        ocr=OCRData(raw_text="Новая коллекция", confidence=0.9),
        vision=None,
        review=ReviewInfo(),
    )
    hints = RenderHints(available_width_px=680, available_height_px=95, estimated_lines=1)

    results = provider.translate_fused_blocks([(fused, hints)], target_lang="EN")

    assert results[0].translated_text == "New Collection"
    user_content = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "680" in user_content


def test_translate_wraps_client_errors_and_sanitizes_message(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Auth failed sk-secret-key")
    provider = OpenAITranslationProvider(api_key="sk-secret-key", client=mock_client)
    contexts = [
        TranslationContext(
            input=TranslationInput(
                id="block_001",
                text="Test",
                semantic_role="label",
                available_width_px=100,
                available_height_px=20,
            )
        )
    ]

    with pytest.raises(RuntimeError) as exc_info:
        provider.translate_details(contexts, target_lang="EN")

    assert "sk-secret-key" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)
