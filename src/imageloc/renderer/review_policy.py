"""Final-render eligibility rules for review-gated TextBlocks."""

from __future__ import annotations

from imageloc.models.text_block import TextBlock

BLOCK_STATUS_PENDING_REVIEW = "pending_review"
BLOCK_STATUS_REJECTED = "rejected"


def is_final_render_eligible(block: TextBlock) -> bool:
    """Return True when a block may participate in final localized rendering."""
    review = block.review
    if not review.requires_review:
        return True
    return review.user_confirmed is True


def final_render_skip_reason(block: TextBlock) -> str | None:
    """Return a stable skip reason when final render must not run for this block."""
    review = block.review
    if not review.requires_review:
        return None
    if review.user_confirmed is True:
        return None
    if review.user_confirmed is False or review.user_action == "rejected":
        return BLOCK_STATUS_REJECTED
    return BLOCK_STATUS_PENDING_REVIEW
