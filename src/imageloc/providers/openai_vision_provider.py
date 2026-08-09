"""OpenAI Vision provider — coррекция OCR, semantic role, typography, language.

Sends the image plus a compact summary of EasyOCR anchors (id, raw_text,
bbox) to OpenAI, asking it to correct/verify text, classify each block's
role and describe its approximate visual style. Never treated as an
authoritative source of geometry — see OCRVisionFusion.
"""

from __future__ import annotations

import base64
import io
import json
import re
from difflib import SequenceMatcher
from typing import Any

from PIL import Image

from imageloc.models.raw_blocks import BoundingBox, RawOCRBlock, RawVisionBlock
from imageloc.utils.security import sanitize_message

# Marker confidence applied when anchor scope guard rejects a Vision correction.
VISION_SCOPE_FALLBACK_CONFIDENCE = 0.35
_MIN_CORRECTION_SIMILARITY = 0.20

VISION_JSON_SCHEMA: dict[str, Any] = {
    "name": "vision_blocks",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "blocks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ocr_block_id": {"type": ["string", "null"]},
                        "text": {"type": "string"},
                        "corrected_text": {"type": "string"},
                        "semantic_role": {"type": "string"},
                        "language": {"type": ["string", "null"]},
                        "font_category": {"type": ["string", "null"]},
                        "font_style": {"type": ["string", "null"]},
                        "font_weight": {"type": ["string", "null"]},
                        "alignment": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                        "bbox": {
                            "type": ["object", "null"],
                            "properties": {
                                "x": {"type": "integer"},
                                "y": {"type": "integer"},
                                "width": {"type": "integer"},
                                "height": {"type": "integer"},
                            },
                            "required": ["x", "y", "width", "height"],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "ocr_block_id",
                        "text",
                        "corrected_text",
                        "semantic_role",
                        "language",
                        "font_category",
                        "font_style",
                        "font_weight",
                        "alignment",
                        "confidence",
                        "bbox",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["blocks"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You analyze an image containing text for a localization pipeline. "
    "You receive a fixed list of OCR anchors (id, raw_text, bbox). "
    "Your primary task is to analyze EACH OCR anchor individually.\n\n"
    "Rules for OCR anchors:\n"
    "1. Return exactly one Vision block per OCR anchor whenever that anchor "
    "contains visible text, a symbol, or a graphic element.\n"
    "2. For every block that corresponds to an OCR anchor, set ocr_block_id to "
    "that anchor's id. This is mandatory — never leave ocr_block_id null for "
    "an anchor you describe.\n"
    "3. Do NOT merge multiple OCR anchors into one Vision block. Do NOT combine "
    "paragraphs, lines, or adjacent anchors. Each anchor keeps its own block.\n"
    "4. Use the OCR anchor's bbox as the spatial reference; do not expand a block "
    "to cover neighboring anchors.\n"
    "5. corrected_text MUST contain ONLY the text visible inside that anchor's bbox. "
    "Never copy text from a neighboring line, neighboring anchor, or larger paragraph.\n"
    "6. Correct OCR misreads in corrected_text while preserving the anchor scope and "
    "rough length — corrected_text should be a local correction of that anchor's raw_text, "
    "not replacement text from elsewhere.\n"
    "7. If you cannot confidently read the text inside a specific bbox, set both text and "
    "corrected_text to that anchor's raw_text. Do not substitute text from another bbox.\n\n"
    "Additional text:\n"
    "- Only when you see readable text that was NOT covered by any OCR anchor, "
    "return extra block(s) with ocr_block_id set to null.\n"
    "- Never invent text that is not visible in the image.\n\n"
    "Classification:\n"
    "- semantic_role must be one of: paragraph, heading, label, caption, title, "
    "button, other.\n"
    "- Use semantic_role=other and language=null for icons, decorative shapes, "
    "or non-linguistic glyphs that are not normal translatable text.\n"
    "- Use a real language code for actual readable words."
)


USER_PROMPT_TEMPLATE = (
    "Analyze every OCR anchor below individually. "
    "Return one Vision block per anchor with matching ocr_block_id. "
    "Do not merge anchors. "
    "For each anchor, corrected_text must come only from that anchor's bbox — "
    "never from a neighbor. "
    "If the bbox is unreadable, return corrected_text equal to raw_text. "
    "Use ocr_block_id=null only for genuinely additional text absent from the list.\n\n"
    "OCR anchors:\n{anchors_json}"
)


def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _format_ocr_anchors(ocr_blocks: list[RawOCRBlock]) -> str:
    anchors = [
        {
            "id": block.id,
            "raw_text": block.raw_text,
            "bbox": block.bbox.model_dump(),
        }
        for block in ocr_blocks
    ]
    return json.dumps(anchors, ensure_ascii=False)


def build_messages(image: Image.Image, ocr_blocks: list[RawOCRBlock]) -> list[dict[str, Any]]:
    """Build the chat messages payload sent to OpenAI (image + OCR anchors)."""
    image_b64 = _encode_image(image)
    anchors_json = _format_ocr_anchors(ocr_blocks)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT_TEMPLATE.format(anchors_json=anchors_json),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        },
    ]


def parse_vision_response(content: str) -> list[RawVisionBlock]:
    """Parse the model's JSON response into a list of RawVisionBlock.

    Individual malformed entries are skipped (not fatal) so one bad block
    doesn't discard an otherwise useful analysis; OCRVisionFusion already
    treats OCR-unmatched blocks as review-worthy.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"OpenAI Vision returned invalid JSON: {sanitize_message(str(exc))}") from exc

    raw_blocks = data.get("blocks", []) if isinstance(data, dict) else []
    blocks: list[RawVisionBlock] = []

    for entry in raw_blocks:
        if not isinstance(entry, dict):
            continue
        try:
            # Required keys accessed via [] on purpose: a missing "text" or
            # "corrected_text" means the entry doesn't describe real text
            # and should be skipped, not silently defaulted to "".
            text = entry["text"]
            corrected_text = entry["corrected_text"]

            bbox_data = entry.get("bbox")
            bbox = BoundingBox(**bbox_data) if bbox_data else None
            blocks.append(
                RawVisionBlock(
                    ocr_block_id=entry.get("ocr_block_id"),
                    text=text,
                    corrected_text=corrected_text,
                    semantic_role=entry.get("semantic_role") or "other",
                    language=entry.get("language"),
                    font_category=entry.get("font_category"),
                    font_style=entry.get("font_style"),
                    font_weight=entry.get("font_weight"),
                    alignment=entry.get("alignment"),
                    confidence=entry.get("confidence", 0.5),
                    bbox=bbox,
                )
            )
        except Exception:
            # Skip malformed entries rather than failing the whole analysis;
            # the affected OCR block simply ends up with vision=None and is
            # flagged `vision_no_match` by OCRVisionFusion.
            continue

    return blocks


def _word_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text, flags=re.UNICODE) if len(token) >= 2}


def correction_similarity(ocr_text: str, corrected_text: str) -> float:
    """Similarity between OCR raw text and Vision corrected_text."""
    return SequenceMatcher(None, ocr_text.strip(), corrected_text.strip()).ratio()


def is_plausible_anchor_correction(ocr_text: str, corrected_text: str) -> bool:
    """Return True when corrected_text plausibly corrects OCR inside the same anchor."""
    ocr = ocr_text.strip()
    corrected = corrected_text.strip()
    if not ocr and not corrected:
        return True
    if not corrected:
        return False

    similarity = correction_similarity(ocr, corrected)
    shared_tokens = _word_tokens(ocr) & _word_tokens(corrected)

    if similarity >= _MIN_CORRECTION_SIMILARITY:
        if shared_tokens or len(corrected) <= len(ocr) + 6:
            return True
        return False

    if shared_tokens:
        return True

    ocr_lower = ocr.lower()
    corrected_lower = corrected.lower()
    if ocr_lower and corrected_lower and (ocr_lower in corrected_lower or corrected_lower in ocr_lower):
        return len(corrected) <= max(len(ocr) * 2, len(ocr) + 8)

    if len(corrected) > len(ocr) and similarity < 0.35:
        return False

    if len(corrected) > max(len(ocr) * 2, 20) and similarity < 0.15:
        return False

    if len(ocr) <= 3 and similarity < 0.4:
        return False

    return similarity >= 0.15


def apply_anchor_scope_guard(vision_block: RawVisionBlock, ocr_block: RawOCRBlock) -> RawVisionBlock:
    """Reject neighbor-scope bleed; fall back to OCR raw text for fusion review."""
    if vision_block.ocr_block_id != ocr_block.id:
        return vision_block
    if is_plausible_anchor_correction(ocr_block.raw_text, vision_block.corrected_text):
        return vision_block

    return vision_block.model_copy(
        update={
            "text": ocr_block.raw_text,
            "corrected_text": ocr_block.raw_text,
            "confidence": VISION_SCOPE_FALLBACK_CONFIDENCE,
        }
    )


def normalize_vision_blocks(
    blocks: list[RawVisionBlock],
    ocr_blocks: list[RawOCRBlock],
) -> list[RawVisionBlock]:
    """Keep one Vision block per OCR anchor; demote invalid ids to unmatched."""
    valid_ids = {block.id for block in ocr_blocks}
    ocr_by_id = {block.id: block for block in ocr_blocks}
    best_for_anchor: dict[str, RawVisionBlock] = {}
    unmatched: list[RawVisionBlock] = []

    for block in blocks:
        anchor_id = block.ocr_block_id
        if anchor_id and anchor_id in valid_ids:
            existing = best_for_anchor.get(anchor_id)
            if existing is None or block.confidence > existing.confidence:
                best_for_anchor[anchor_id] = block
            continue

        if anchor_id and anchor_id not in valid_ids:
            block = block.model_copy(update={"ocr_block_id": None})
        unmatched.append(block)

    guarded = [
        apply_anchor_scope_guard(best_for_anchor[anchor_id], ocr_by_id[anchor_id])
        for anchor_id in sorted(best_for_anchor)
    ]
    return guarded + unmatched


class OpenAIVisionProvider:
    """Vision provider backed by the OpenAI API.

    Implements VisionProvider.analyze(image, ocr_blocks). The API key is
    passed in explicitly by the caller (typically loaded from .env via
    imageloc.config) — this class never reads .env or hardcodes a key.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", client: Any | None = None) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def analyze(self, image: Image.Image, ocr_blocks: list[RawOCRBlock]) -> list[RawVisionBlock]:
        """Send the image + OCR anchors to OpenAI and return RawVisionBlocks."""
        client = self._get_client()
        messages = build_messages(image, ocr_blocks)

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": VISION_JSON_SCHEMA},
            )
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI Vision request failed: {sanitize_message(str(exc))}"
            ) from exc

        content = response.choices[0].message.content
        if not content:
            return []

        parsed = parse_vision_response(content)
        return normalize_vision_blocks(parsed, ocr_blocks)
