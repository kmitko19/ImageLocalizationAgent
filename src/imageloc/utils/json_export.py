"""Serialize PipelineResult to JSON and apply manual review updates."""

from __future__ import annotations

import json
from pathlib import Path

from imageloc.models.result import PipelineResult


def save_result(result: PipelineResult, output_dir: Path) -> Path:
    """Write PipelineResult to data/output/{source_image_id}/result.json."""
    destination_dir = output_dir / result.source_image_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / "result.json"
    output_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output_path


def load_result(path: str | Path) -> PipelineResult:
    """Load PipelineResult from a JSON file."""
    json_path = Path(path)
    return PipelineResult.model_validate_json(json_path.read_text(encoding="utf-8"))


def apply_user_review(
    result: PipelineResult,
    block_id: str,
    *,
    approved: bool,
) -> PipelineResult:
    """Update review.user_confirmed / user_action for one block (GUI use)."""
    return apply_block_review(result, block_id, approved=approved)


def apply_block_review(
    result: PipelineResult,
    block_id: str,
    *,
    approved: bool,
    user_action: str | None = None,
    original_text: str | None = None,
    translated_text: str | None = None,
) -> PipelineResult:
    """Apply a manual review decision and optional text edits for one block.

    OCR raw_text and vision audit fields are never modified.
    """
    if user_action is None:
        user_action = "approved" if approved else "rejected"

    updated_blocks = []
    for block in result.text_blocks:
        if block.id != block_id:
            updated_blocks.append(block)
            continue

        updates: dict = {
            "review": block.review.model_copy(
                update={
                    "user_confirmed": approved,
                    "user_action": user_action,
                }
            )
        }
        if original_text is not None:
            updates["original_text"] = original_text
        if translated_text is not None:
            updates["translated_text"] = translated_text

        updated_blocks.append(block.model_copy(update=updates))

    return result.model_copy(update={"text_blocks": updated_blocks})


def result_to_json_string(result: PipelineResult, *, indent: int = 2) -> str:
    """Return formatted JSON string for GUI preview/download."""
    return result.model_dump_json(indent=indent)


def validate_json_export(result: PipelineResult) -> dict:
    """Round-trip validate JSON export (useful in tests)."""
    restored = PipelineResult.model_validate_json(result.model_dump_json())
    return json.loads(restored.model_dump_json())
