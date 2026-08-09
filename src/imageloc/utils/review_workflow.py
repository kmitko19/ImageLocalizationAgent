"""Manual review workflow helpers for the Gradio GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.result import PipelineResult
from imageloc.models.text_block import TextBlock
from imageloc.utils.json_export import apply_block_review, load_result, save_result


@dataclass(frozen=True)
class ReviewProgress:
    reviewed: int
    total: int
    remaining: int
    is_complete: bool

    @property
    def label(self) -> str:
        if self.is_complete:
            return f"Review complete\nReviewed {self.reviewed} / {self.total}\nRemaining 0"
        return f"Reviewed {self.reviewed} / {self.total}\nRemaining {self.remaining}"


@dataclass(frozen=True)
class BlockReviewDisplay:
    block_id: str
    ocr_raw_text: str
    vision_corrected_text: str | None
    original_text: str
    translated_text: str | None
    review_reasons: list[str]
    ocr_confidence: float
    vision_confidence: float | None
    bbox: BoundingBox


def review_queue(result: PipelineResult) -> list[TextBlock]:
    """Blocks that still require a manual decision."""
    return [
        block
        for block in result.text_blocks
        if block.review.requires_review and block.review.user_confirmed is None
    ]


def review_progress(result: PipelineResult) -> ReviewProgress:
    """Dynamic review progress based on user_confirmed, not stats snapshot."""
    pending = review_queue(result)
    total = sum(1 for block in result.text_blocks if block.review.requires_review)
    remaining = len(pending)
    reviewed = total - remaining
    return ReviewProgress(
        reviewed=reviewed,
        total=total,
        remaining=remaining,
        is_complete=total > 0 and remaining == 0,
    )


def get_block_by_id(result: PipelineResult, block_id: str) -> TextBlock | None:
    return next((block for block in result.text_blocks if block.id == block_id), None)


def build_block_display(block: TextBlock) -> BlockReviewDisplay:
    return BlockReviewDisplay(
        block_id=block.id,
        ocr_raw_text=block.ocr.raw_text,
        vision_corrected_text=block.vision.corrected_text if block.vision else None,
        original_text=block.original_text,
        translated_text=block.translated_text,
        review_reasons=list(block.review.reasons),
        ocr_confidence=block.ocr.confidence,
        vision_confidence=block.vision.confidence if block.vision else None,
        bbox=block.bbox,
    )


def crop_block_image(image: Image.Image, bbox: BoundingBox, *, padding: int = 2) -> Image.Image:
    """Crop the source image to a block bbox for review preview."""
    width, height = image.size
    left = max(0, bbox.x - padding)
    top = max(0, bbox.y - padding)
    right = min(width, bbox.x + bbox.width + padding)
    bottom = min(height, bbox.y + bbox.height + padding)
    return image.crop((left, top, right, bottom))


def next_pending_block_id(result: PipelineResult, current_block_id: str | None = None) -> str | None:
    """Return the next pending review block after current_block_id."""
    pending = review_queue(result)
    if not pending:
        return None

    if current_block_id is None:
        return pending[0].id

    pending_ids = [block.id for block in pending]
    if current_block_id not in pending_ids:
        return pending[0].id

    current_index = pending_ids.index(current_block_id)
    next_index = current_index + 1
    if next_index >= len(pending_ids):
        return None
    return pending_ids[next_index]


def approve_block(result: PipelineResult, block_id: str) -> PipelineResult:
    return apply_block_review(result, block_id, approved=True, user_action="approved")


def edit_and_approve_block(
    result: PipelineResult,
    block_id: str,
    *,
    original_text: str,
    translated_text: str | None,
) -> PipelineResult:
    return apply_block_review(
        result,
        block_id,
        approved=True,
        user_action="edited",
        original_text=original_text,
        translated_text=translated_text,
    )


def reject_block(result: PipelineResult, block_id: str) -> PipelineResult:
    return apply_block_review(result, block_id, approved=False, user_action="rejected")


def save_reviewed_result(result: PipelineResult, output_dir: Path) -> Path:
    return save_result(result, output_dir)


def load_reviewed_result(path: str | Path) -> PipelineResult:
    return load_result(path)
