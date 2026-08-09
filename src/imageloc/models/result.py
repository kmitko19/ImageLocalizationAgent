"""Top-level pipeline output — the JSON file written for each analyzed image."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from imageloc.models.raw_blocks import RawVisionBlock
from imageloc.models.text_block import TextBlock

CONTRACT_VERSION = "1.1"


class SourceImage(BaseModel):
    filename: str
    path: str
    width: int
    height: int
    format: str


class Stats(BaseModel):
    total_blocks: int
    blocks_requiring_review: int
    avg_confidence: float
    ocr_blocks: int
    vision_blocks: int


class PipelineResult(BaseModel):
    version: str = CONTRACT_VERSION
    source_image_id: str
    source_image: SourceImage

    source_language: str | None = None
    source_language_mode: str = "auto"
    target_language: str

    processed_at: datetime

    text_blocks: list[TextBlock] = Field(default_factory=list)
    unmatched_vision_blocks: list[RawVisionBlock] = Field(default_factory=list)
    stats: Stats
