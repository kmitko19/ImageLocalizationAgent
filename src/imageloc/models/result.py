"""Top-level pipeline output — the JSON file written for each analyzed image."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from imageloc.models.raw_blocks import RawVisionBlock
from imageloc.models.text_block import TextBlock

CONTRACT_VERSION = "1.2"


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


class RendererBlockDiagnostics(BaseModel):
    block_id: str
    status: str
    restore_success: bool
    restore_method: str | None = None
    render_success: bool
    failure_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class RendererDiagnostics(BaseModel):
    success: bool = False
    completion_status: str = "complete"
    rendered_block_count: int = 0
    skipped_block_count: int = 0
    failed_block_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    fatal_error: str | None = None
    output_paths: dict[str, str] = Field(default_factory=dict)
    block_results: list[RendererBlockDiagnostics] = Field(default_factory=list)


class PipelineResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    pipeline_success: bool = True
    pipeline_completion_status: str = "complete"
    renderer: RendererDiagnostics | None = None
    localized_image_path: str | None = None
    clean_background_path: str | None = None
    localized_image: object | None = Field(default=None, exclude=True)
    clean_background: object | None = Field(default=None, exclude=True)
