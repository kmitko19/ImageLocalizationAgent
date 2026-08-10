"""Renderer orchestration layer for R0.7."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from imageloc.config import PROJECT_ROOT
from imageloc.models.text_block import TextBlock
from imageloc.renderer.background_restorer import (
    BackgroundRestoreResult,
    restore_backgrounds,
    save_clean_background,
)
from imageloc.renderer.text_renderer import (
    TextRenderResult,
    render_text_blocks,
    save_localized_image,
)

RENDERER_RESULT_FILENAME = "renderer_result.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
SOURCE_IMAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

BLOCK_STATUS_RENDERED = "rendered"
BLOCK_STATUS_SKIPPED = "skipped"
BLOCK_STATUS_FAILED = "failed"

# Block processing order: sequential compositing in input list order.
# Overlapping blocks later in the list paint over earlier ones without semantic merge.


@dataclass
class BlockRendererResult:
    """Per-block Renderer orchestration outcome."""

    block_id: str
    status: str
    restore_success: bool
    restore_method: str | None
    render_success: bool
    failure_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "status": self.status,
            "restore_success": self.restore_success,
            "restore_method": self.restore_method,
            "render_success": self.render_success,
            "failure_reason": self.failure_reason,
            "warnings": list(self.warnings),
        }


@dataclass
class RendererResult:
    """Structured result of render_image_localization."""

    image: Image.Image
    clean_background: Image.Image
    success: bool
    block_results: list[BlockRendererResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failed_block_ids: list[str] = field(default_factory=list)
    rendered_block_count: int = 0
    skipped_block_count: int = 0
    source_image_id: str | None = None
    output_paths: dict[str, str] = field(default_factory=dict)
    fatal_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "source_image_id": self.source_image_id,
            "rendered_block_count": self.rendered_block_count,
            "skipped_block_count": self.skipped_block_count,
            "failed_block_ids": list(self.failed_block_ids),
            "warnings": list(self.warnings),
            "fatal_error": self.fatal_error,
            "output_paths": dict(self.output_paths),
            "block_results": [item.to_dict() for item in self.block_results],
        }


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _validate_image(image: Image.Image) -> None:
    if not isinstance(image, Image.Image):
        raise TypeError("original_image must be a PIL Image")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("original_image must have positive width and height")


def _validate_source_image_id(source_image_id: str) -> None:
    normalized = source_image_id.strip()
    if not normalized:
        raise ValueError("source_image_id must be a non-empty string")
    if not SOURCE_IMAGE_ID_PATTERN.fullmatch(normalized):
        raise ValueError("source_image_id contains invalid characters")


def _resolve_output_dir(output_dir: Path | None) -> Path:
    resolved = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _is_empty_translated_text(block: TextBlock) -> bool:
    translated = block.translated_text
    return translated is None or not str(translated).strip()


def _classify_block_status(block: TextBlock, render_result: TextRenderResult) -> str:
    if _is_empty_translated_text(block):
        return BLOCK_STATUS_SKIPPED
    if not render_result.success:
        return BLOCK_STATUS_FAILED
    if render_result.rendered_lines:
        return BLOCK_STATUS_RENDERED
    return BLOCK_STATUS_SKIPPED


def _build_block_result(
    block: TextBlock,
    restore_result: BackgroundRestoreResult,
    render_result: TextRenderResult,
) -> BlockRendererResult:
    warnings: list[str] = []
    for warning in restore_result.warnings:
        warnings.append(f"restore[{block.id}]: {warning}")
    for warning in render_result.warnings:
        warnings.append(f"render[{block.id}]: {warning}")

    status = _classify_block_status(block, render_result)
    return BlockRendererResult(
        block_id=block.id,
        status=status,
        restore_success=restore_result.success,
        restore_method=restore_result.method,
        render_success=render_result.success,
        failure_reason=render_result.failure_reason,
        warnings=warnings,
    )


def _aggregate_counts(block_results: list[BlockRendererResult]) -> tuple[int, int, list[str]]:
    rendered_count = sum(1 for item in block_results if item.status == BLOCK_STATUS_RENDERED)
    skipped_count = sum(1 for item in block_results if item.status == BLOCK_STATUS_SKIPPED)
    failed_block_ids = [item.block_id for item in block_results if item.status == BLOCK_STATUS_FAILED]
    return rendered_count, skipped_count, failed_block_ids


def save_renderer_result_json(
    result: RendererResult,
    *,
    output_dir: Path,
    source_image_id: str,
) -> str:
    """Persist renderer diagnostics JSON under render_assets atomically."""
    _validate_source_image_id(source_image_id)
    assets_dir = output_dir / source_image_id / "render_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    target_path = assets_dir / RENDERER_RESULT_FILENAME
    temp_path = target_path.with_suffix(".tmp.json")
    temp_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(target_path)
    return _relative_to_project(target_path)


def render_image_localization(
    original_image: Image.Image,
    blocks: list[TextBlock],
    *,
    masks: dict[str, np.ndarray] | None = None,
    source_image_id: str | None = None,
    save_assets: bool = False,
    output_dir: Path | None = None,
) -> RendererResult:
    """Orchestrate background restoration, text rendering, and optional asset output."""
    warnings: list[str] = []
    normalized_source_id = source_image_id.strip() if source_image_id else None

    try:
        _validate_image(original_image)
        if save_assets:
            if normalized_source_id is None:
                raise ValueError("source_image_id is required when save_assets=True")
            _validate_source_image_id(normalized_source_id)
            resolved_output_dir = _resolve_output_dir(output_dir)
        else:
            resolved_output_dir = Path(output_dir) if output_dir is not None else None

        clean_background, restore_results = restore_backgrounds(
            original_image,
            blocks,
            masks=masks,
        )
        localized_image, render_results = render_text_blocks(clean_background, blocks)

        if len(restore_results) != len(blocks) or len(render_results) != len(blocks):
            raise RuntimeError("renderer block result count mismatch")

        block_results = [
            _build_block_result(block, restore_result, render_result)
            for block, restore_result, render_result in zip(blocks, restore_results, render_results)
        ]
        rendered_count, skipped_count, failed_block_ids = _aggregate_counts(block_results)
        for item in block_results:
            warnings.extend(item.warnings)

        output_paths: dict[str, str] = {}
        if save_assets:
            assert normalized_source_id is not None
            assert resolved_output_dir is not None
            output_paths["clean_background"] = save_clean_background(
                clean_background,
                output_dir=resolved_output_dir,
                source_image_id=normalized_source_id,
            )
            output_paths["localized_image"] = save_localized_image(
                localized_image,
                output_dir=resolved_output_dir,
                source_image_id=normalized_source_id,
            )
            result_for_json = RendererResult(
                image=localized_image,
                clean_background=clean_background,
                success=True,
                block_results=block_results,
                warnings=warnings,
                failed_block_ids=failed_block_ids,
                rendered_block_count=rendered_count,
                skipped_block_count=skipped_count,
                source_image_id=normalized_source_id,
                output_paths=dict(output_paths),
            )
            output_paths["renderer_result"] = save_renderer_result_json(
                result_for_json,
                output_dir=resolved_output_dir,
                source_image_id=normalized_source_id,
            )

        return RendererResult(
            image=localized_image,
            clean_background=clean_background,
            success=True,
            block_results=block_results,
            warnings=warnings,
            failed_block_ids=failed_block_ids,
            rendered_block_count=rendered_count,
            skipped_block_count=skipped_count,
            source_image_id=normalized_source_id,
            output_paths=output_paths,
        )
    except Exception as exc:
        fallback_image = original_image.copy() if isinstance(original_image, Image.Image) else Image.new("RGB", (1, 1))
        return RendererResult(
            image=fallback_image,
            clean_background=fallback_image.copy(),
            success=False,
            warnings=warnings,
            source_image_id=normalized_source_id,
            fatal_error=str(exc),
        )


def resolve_renderer_result_path(renderer_result_path: str | Path) -> Path:
    """Resolve a stored renderer_result.json path relative to PROJECT_ROOT."""
    path = Path(renderer_result_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
