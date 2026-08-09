"""Round-trip serialization tests for the PipelineResult JSON contract."""

from datetime import datetime, timezone

from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.result import PipelineResult, SourceImage, Stats
from imageloc.models.text_block import (
    OCRData,
    RenderHints,
    ReviewInfo,
    TextBlock,
    VisionData,
    VisualStyle,
)
from imageloc.utils.json_export import (
    apply_block_review,
    apply_user_review,
    load_result,
    save_result,
)


def _sample_result() -> PipelineResult:
    block = TextBlock(
        id="block_001",
        original_text="Новая коллекция",
        translated_text="New Collection",
        source_language="RU",
        confidence=0.88,
        bbox=BoundingBox(x=120, y=200, width=680, height=95),
        polygon=[(120, 200), (800, 205), (798, 295), (118, 290)],
        ocr=OCRData(raw_text="Новая коллекцiя", confidence=0.72),
        vision=VisionData(
            corrected_text="Новая коллекция",
            semantic_role="heading",
            language="RU",
            confidence=0.94,
        ),
        visual=VisualStyle(
            text_color="#2B1A0E",
            background_color="#F5E6D0",
            estimated_text_height_px=78,
            font_size_estimate=64,
            font_size_confidence=0.55,
            font_category="decorative_script",
            font_style="regular",
            font_weight="regular",
            alignment="center",
            rotation_deg=1.2,
        ),
        render_hints=RenderHints(available_width_px=680, available_height_px=95),
        review=ReviewInfo(
            requires_review=True,
            reasons=["decorative_text", "text_mismatch"],
            ocr_vision_similarity=0.91,
        ),
    )

    return PipelineResult(
        source_image_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        source_image=SourceImage(
            filename="catalog_page.jpg",
            path="data/input/test_cases/01_catalog_page_ru.jpg",
            width=2400,
            height=3200,
            format="JPEG",
        ),
        source_language="RU",
        source_language_mode="auto",
        target_language="EN",
        processed_at=datetime.now(timezone.utc),
        text_blocks=[block],
        stats=Stats(
            total_blocks=1,
            blocks_requiring_review=1,
            avg_confidence=0.88,
            ocr_blocks=1,
            vision_blocks=1,
        ),
    )


def test_pipeline_result_round_trips_through_json():
    result = _sample_result()

    json_str = result.model_dump_json(indent=2)
    restored = PipelineResult.model_validate_json(json_str)

    assert restored == result


def test_never_required_review_defaults_to_null_user_decision():
    result = _sample_result()
    review = result.text_blocks[0].review

    assert review.user_confirmed is None
    assert review.user_action is None


def test_api_key_field_does_not_exist_anywhere_in_contract():
    result = _sample_result()
    dumped = result.model_dump_json()

    assert "OPENAI_API_KEY" not in dumped
    assert "api_key" not in dumped.lower()


def test_apply_user_review_sets_approved_action():
    result = _sample_result()
    updated = apply_user_review(result, "block_001", approved=True)
    review = updated.text_blocks[0].review

    assert review.user_confirmed is True
    assert review.user_action == "approved"


def test_apply_block_review_edit_keeps_ocr_and_vision_audit_fields():
    result = _sample_result()
    original_ocr = result.text_blocks[0].ocr.raw_text
    original_vision = result.text_blocks[0].vision.corrected_text

    updated = apply_block_review(
        result,
        "block_001",
        approved=True,
        user_action="edited",
        original_text="Edited original",
        translated_text="Edited translation",
    )
    block = updated.text_blocks[0]

    assert block.original_text == "Edited original"
    assert block.translated_text == "Edited translation"
    assert block.ocr.raw_text == original_ocr
    assert block.vision.corrected_text == original_vision
    assert block.review.user_action == "edited"


def test_apply_block_review_reject_sets_rejected_action():
    result = _sample_result()
    updated = apply_block_review(
        result,
        "block_001",
        approved=False,
        user_action="rejected",
    )

    assert updated.text_blocks[0].review.user_confirmed is False
    assert updated.text_blocks[0].review.user_action == "rejected"


def test_save_and_load_round_trip_after_review(tmp_path):
    result = _sample_result()
    output_dir = tmp_path / "output"

    reviewed = apply_block_review(
        result,
        "block_001",
        approved=True,
        user_action="edited",
        original_text="Edited",
        translated_text="Edited EN",
    )
    saved_path = save_result(reviewed, output_dir)
    restored = load_result(saved_path)

    assert restored.text_blocks[0].review.user_action == "edited"
    assert restored.text_blocks[0].original_text == "Edited"
    assert restored.stats.blocks_requiring_review == 1
