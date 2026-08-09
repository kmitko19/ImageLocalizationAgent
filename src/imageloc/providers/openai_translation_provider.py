"""OpenAI Translation provider — context-aware batch translation.

Considers semantic role, available bbox footprint (width, height, lines) and
whether the translation may need shortening to preserve layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from imageloc.fusion.ocr_vision_fusion import FusedBlock
from imageloc.models.text_block import RenderHints, TextBlock
from imageloc.providers.base import TranslatedBlock, TranslationInput
from imageloc.utils.security import sanitize_message

TRANSLATION_JSON_SCHEMA: dict[str, Any] = {
    "name": "translations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "original_text": {"type": "string"},
                        "translated_text": {"type": "string"},
                        "detected_language": {"type": ["string", "null"]},
                        "translation_confidence": {"type": "number"},
                        "render_notes": {"type": ["string", "null"]},
                    },
                    "required": [
                        "id",
                        "original_text",
                        "translated_text",
                        "detected_language",
                        "translation_confidence",
                        "render_notes",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You translate text blocks for an image localization pipeline. "
    "Each block has a semantic role (heading, label, paragraph, etc.) and "
    "a limited pixel footprint on the image. Produce translations that fit "
    "the available width and line count. Prefer concise wording for labels "
    "and headings; preserve meaning for paragraphs. When the target language "
    "text would be significantly longer, shorten it naturally and explain "
    "the constraint in render_notes. Return one translation per input block "
    "with the same id."
)


@dataclass
class TranslationContext:
    """Full translation request for one block, including render constraints."""

    input: TranslationInput
    estimated_lines: int = 1
    target_line_count: int = 1
    preserve_layout: bool = True


@dataclass
class TranslationResultDetail:
    """Rich translation result returned by OpenAITranslationProvider."""

    id: str
    original_text: str
    translated_text: str
    detected_language: str | None
    translation_confidence: float
    render_notes: str | None = None


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_translation_context_from_fused(
    block: FusedBlock,
    render_hints: RenderHints,
) -> TranslationContext:
    semantic_role = block.vision.semantic_role if block.vision else "other"
    return TranslationContext(
        input=TranslationInput(
            id=block.id,
            text=block.text,
            semantic_role=semantic_role,
            available_width_px=render_hints.available_width_px,
            available_height_px=render_hints.available_height_px,
            font_category=None,
        ),
        estimated_lines=render_hints.estimated_lines,
        target_line_count=render_hints.target_line_count,
        preserve_layout=render_hints.preserve_layout,
    )


def build_translation_context_from_text_block(block: TextBlock) -> TranslationContext:
    font_category = block.visual.font_category if block.visual else None
    semantic_role = block.vision.semantic_role if block.vision else "other"
    return TranslationContext(
        input=TranslationInput(
            id=block.id,
            text=block.original_text,
            semantic_role=semantic_role,
            available_width_px=block.render_hints.available_width_px,
            available_height_px=block.render_hints.available_height_px,
            font_category=font_category,
        ),
        estimated_lines=block.render_hints.estimated_lines,
        target_line_count=block.render_hints.target_line_count,
        preserve_layout=block.render_hints.preserve_layout,
    )


def build_translation_payload(contexts: list[TranslationContext]) -> str:
    payload = [
        {
            "id": ctx.input.id,
            "text": ctx.input.text,
            "semantic_role": ctx.input.semantic_role,
            "available_width_px": ctx.input.available_width_px,
            "available_height_px": ctx.input.available_height_px,
            "estimated_lines": ctx.estimated_lines,
            "target_line_count": ctx.target_line_count,
            "preserve_layout": ctx.preserve_layout,
            "font_category": ctx.input.font_category,
        }
        for ctx in contexts
    ]
    return json.dumps(payload, ensure_ascii=False)


def build_messages(
    contexts: list[TranslationContext],
    target_lang: str,
    source_lang: str | None,
) -> list[dict[str, str]]:
    source = source_lang or "auto"
    blocks_json = build_translation_payload(contexts)
    user_prompt = (
        f"Source language: {source}\n"
        f"Target language: {target_lang}\n"
        f"Blocks to translate:\n{blocks_json}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_translation_response(content: str) -> list[TranslationResultDetail]:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"OpenAI Translation returned invalid JSON: {sanitize_message(str(exc))}"
        ) from exc

    entries = data.get("translations", []) if isinstance(data, dict) else []
    results: list[TranslationResultDetail] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            results.append(
                TranslationResultDetail(
                    id=entry["id"],
                    original_text=entry["original_text"],
                    translated_text=entry["translated_text"],
                    detected_language=entry.get("detected_language"),
                    translation_confidence=_clamp_confidence(
                        entry.get("translation_confidence", 0.5)
                    ),
                    render_notes=entry.get("render_notes"),
                )
            )
        except KeyError:
            continue

    return results


class OpenAITranslationProvider:
    """Translation provider backed by the OpenAI API.

    Implements TranslationProvider.translate(). Use translate_details() when
    the caller needs detected_language, confidence and render_notes.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", client: Any | None = None) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def translate_details(
        self,
        contexts: list[TranslationContext],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslationResultDetail]:
        if not contexts:
            return []

        client = self._get_client()
        messages = build_messages(contexts, target_lang, source_lang)

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": TRANSLATION_JSON_SCHEMA},
            )
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI Translation request failed: {sanitize_message(str(exc))}"
            ) from exc

        content = response.choices[0].message.content
        if not content:
            return []

        return parse_translation_response(content)

    def translate(
        self,
        blocks: list[TranslationInput],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslatedBlock]:
        """Protocol method — returns id + translated_text for each block."""
        contexts = [
            TranslationContext(input=block, estimated_lines=1, target_line_count=1)
            for block in blocks
        ]
        details = self.translate_details(contexts, target_lang, source_lang)
        return [
            TranslatedBlock(id=item.id, translated_text=item.translated_text) for item in details
        ]

    def translate_fused_blocks(
        self,
        items: list[tuple[FusedBlock, RenderHints]],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslationResultDetail]:
        contexts = [build_translation_context_from_fused(block, hints) for block, hints in items]
        return self.translate_details(contexts, target_lang, source_lang)

    def translate_text_blocks(
        self,
        blocks: list[TextBlock],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslationResultDetail]:
        contexts = [build_translation_context_from_text_block(block) for block in blocks]
        return self.translate_details(contexts, target_lang, source_lang)
