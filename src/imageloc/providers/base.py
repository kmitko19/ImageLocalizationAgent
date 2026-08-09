"""Provider interfaces (Protocols) for OCR, Vision analysis and Translation.

These are intentionally plain `typing.Protocol` definitions rather than a
plugin framework: the orchestrator takes concrete provider instances through
its constructor (simple dependency injection), so swapping an implementation
later (e.g. adding a DeepL translator) means passing a different object, not
registering it anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from imageloc.models.raw_blocks import RawOCRBlock, RawVisionBlock


class OCRProvider(Protocol):
    """Detects text regions and reads their text. Owns all geometry
    (bbox/polygon) — this is the single source of truth for coordinates
    used throughout the pipeline."""

    def detect(self, image: Image.Image) -> list[RawOCRBlock]:
        ...


class VisionProvider(Protocol):
    """Understands the image: corrects OCR text, infers semantic role and
    approximate typography. Never treated as authoritative for geometry."""

    def analyze(
        self, image: Image.Image, ocr_blocks: list[RawOCRBlock]
    ) -> list[RawVisionBlock]:
        ...


@dataclass
class TranslationInput:
    """Everything a translation provider needs to produce a context-aware
    translation for one block, without re-touching the image."""

    id: str
    text: str
    semantic_role: str
    available_width_px: int
    available_height_px: int
    font_category: str | None = None


@dataclass
class TranslatedBlock:
    id: str
    translated_text: str


class TranslationProvider(Protocol):
    """Translates a batch of blocks in one call, using layout/context hints
    so translations have a realistic chance of fitting their original
    footprint."""

    def translate(
        self,
        blocks: list[TranslationInput],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslatedBlock]:
        ...
