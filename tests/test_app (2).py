"""Tests for the Gradio GUI entry point — no real API or server startup."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest
from PIL import Image

from imageloc.app import (
    RESULT_JSON_CSS,
    build_orchestrator,
    build_review_panel,
    create_app,
    main,
    process_review_action,
    run_analysis,
    _empty_review_outputs,
)
from imageloc.config import FusionSettings, OpenAISettings, PathsSettings, Settings
from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.result import PipelineResult, SourceImage, Stats
from imageloc.models.text_block import OCRData, RenderHints, ReviewInfo, TextBlock, VisionData, VisualStyle

SAMPLE_SETTINGS = Settings(
    openai=OpenAISettings(
        vision_model="gpt-4o",
        translation_model="gpt-4o-mini",
        api_key="test-key",
    ),
    ocr_languages=["ru", "en"],
    fusion=FusionSettings(
        iou_threshold=0.25,
        text_similarity_threshold=0.55,
        text_mismatch_threshold=0.5,
        low_ocr_confidence_threshold=0.5,
        low_font_size_confidence_threshold=0.4,
    ),
    paths=PathsSettings(
        test_cases_dir=Path("data/input/test_cases"),
        output_dir=Path("data/output"),
    ),
    gui_port=7860,
    source_languages=["auto", "ru", "en"],
    target_languages=["en", "ru"],
)


def _sample_result(source_image_id: str = "test-id", *, requires_review: bool = False) -> PipelineResult:
    return PipelineResult(
        source_image_id=source_image_id,
        source_image=SourceImage(
            filename="sample.png",
            path="data/output/sample/source.png",
            width=200,
            height=120,
            format="PNG",
        ),
        source_language="RU",
        source_language_mode="auto",
        target_language="EN",
        processed_at=datetime.now(timezone.utc),
        text_blocks=[
            TextBlock(
                id="block_001",
                original_text="Sample",
                translated_text="Пример",
                source_language="EN",
                confidence=0.9,
                bbox=BoundingBox(x=10, y=20, width=100, height=40),
                polygon=[(10, 20), (110, 20), (110, 60), (10, 60)],
                ocr=OCRData(raw_text="Sample", confidence=0.9),
                vision=VisionData(
                    corrected_text="Sample",
                    semantic_role="label",
                    language="EN",
                    confidence=0.95,
                ),
                visual=VisualStyle(estimated_text_height_px=40),
                render_hints=RenderHints(available_width_px=100, available_height_px=40),
                review=ReviewInfo(requires_review=requires_review, reasons=["text_mismatch"]),
            )
        ],
        stats=Stats(
            total_blocks=1,
            blocks_requiring_review=1 if requires_review else 0,
            avg_confidence=0.9,
            ocr_blocks=1,
            vision_blocks=1,
        ),
    )


def test_build_orchestrator_wires_settings():
    orchestrator = build_orchestrator(SAMPLE_SETTINGS)

    assert orchestrator.fusion_settings.iou_threshold == pytest.approx(0.25)
    assert orchestrator.output_dir == SAMPLE_SETTINGS.paths.output_dir


def test_run_analysis_requires_upload(tmp_path):
    orchestrator = MagicMock()

    result, json_text, preview, status, source_path = run_analysis(
        None,
        "en",
        "auto",
        orchestrator=orchestrator,
        output_dir=tmp_path,
    )

    assert result is None
    assert json_text == ""
    assert preview is None
    assert source_path is None
    assert "Upload an image first" in status
    orchestrator.run.assert_not_called()


def test_run_analysis_returns_json_and_preview(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 120), color="white").save(image_path)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = _sample_result(requires_review=True)
    orchestrator = MagicMock()
    orchestrator.run.return_value = result

    monkeypatch.setattr(
        "imageloc.app._resolve_image_path_for_preview",
        lambda _result, upload_path: upload_path,
    )

    analysis_result, json_text, preview, status, source_path = run_analysis(
        str(image_path),
        "ru",
        "auto",
        orchestrator=orchestrator,
        output_dir=output_dir,
    )

    orchestrator.run.assert_called_once_with(
        image_path,
        target_language="RU",
        source_language=None,
        source_language_mode="auto",
        is_test_case=False,
    )
    assert analysis_result is result
    assert '"original_text": "Sample"' in json_text
    assert preview is not None
    assert preview.size == (200, 120)
    assert source_path == str(image_path)
    assert "Analysis complete: 1 blocks" in status
    assert (output_dir / result.source_image_id / "result.json").exists()


def test_create_app_returns_gradio_blocks():
    app = create_app(SAMPLE_SETTINGS)

    assert isinstance(app, gr.Blocks)


def test_main_launches_gui_on_configured_port():
    mock_app = MagicMock()

    with patch("imageloc.app.load_settings", return_value=SAMPLE_SETTINGS), patch(
        "imageloc.app.create_app",
        return_value=mock_app,
    ) as mock_create_app:
        main()

    mock_create_app.assert_called_once_with(SAMPLE_SETTINGS)
    mock_app.launch.assert_called_once_with(
        server_port=7860,
        share=False,
        css=RESULT_JSON_CSS,
    )


def test_build_review_panel_shows_complete_state():
    result = _sample_result(requires_review=True)
    result = result.model_copy(
        update={
            "text_blocks": [
                result.text_blocks[0].model_copy(
                    update={
                        "review": result.text_blocks[0].review.model_copy(
                            update={
                                "user_confirmed": True,
                                "user_action": "approved",
                            }
                        )
                    }
                )
            ]
        }
    )

    panel = build_review_panel(result, None, None)

    assert len(panel) == 12
    assert panel[-1] == "Review complete\nReviewed 1 / 1\nRemaining 0"


def test_empty_review_outputs_matches_review_panel_output_count():
    assert len(_empty_review_outputs()) == 12


def test_process_review_action_last_block_returns_complete_panel_without_output_mismatch(
    tmp_path,
):
    """Regression: completing the last pending block must return 12 review outputs."""
    block_one = TextBlock(
        id="block_001",
        original_text="First",
        translated_text="First EN",
        source_language="EN",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=20, width=100, height=40),
        polygon=[(10, 20), (110, 20), (110, 60), (10, 60)],
        ocr=OCRData(raw_text="First", confidence=0.9),
        vision=VisionData(
            corrected_text="First",
            semantic_role="label",
            language="EN",
            confidence=0.95,
        ),
        visual=VisualStyle(estimated_text_height_px=40),
        render_hints=RenderHints(available_width_px=100, available_height_px=40),
        review=ReviewInfo(requires_review=True, reasons=["text_mismatch"]),
    )
    block_two = block_one.model_copy(
        update={
            "id": "block_002",
            "original_text": "Second",
            "translated_text": "Second EN",
            "ocr": OCRData(raw_text="Second", confidence=0.9),
            "vision": VisionData(
                corrected_text="Second",
                semantic_role="label",
                language="EN",
                confidence=0.95,
            ),
        }
    )
    result = PipelineResult(
        source_image_id="review-complete-test",
        source_image=SourceImage(
            filename="sample.png",
            path="data/output/sample/source.png",
            width=200,
            height=120,
            format="PNG",
        ),
        source_language="RU",
        source_language_mode="auto",
        target_language="EN",
        processed_at=datetime.now(timezone.utc),
        text_blocks=[block_one, block_two],
        stats=Stats(
            total_blocks=2,
            blocks_requiring_review=2,
            avg_confidence=0.9,
            ocr_blocks=2,
            vision_blocks=2,
        ),
    )
    output_dir = tmp_path / "output"

    after_first, _, _, first_panel = process_review_action(
        result,
        None,
        "block_001",
        action="approve",
        output_dir=output_dir,
    )
    assert len(first_panel) == 12
    assert first_panel[-1] == "Reviewed 1 / 2\nRemaining 1"

    _, json_text, _, final_panel = process_review_action(
        after_first,
        None,
        "block_002",
        action="edit",
        edited_original="Second edited",
        edited_translation="Second edited EN",
        output_dir=output_dir,
    )

    assert len(final_panel) == 12
    assert final_panel[0] == gr.update(choices=[], value=None)
    assert final_panel[-1] == "Review complete\nReviewed 2 / 2\nRemaining 0"
    assert '"user_action": "edited"' in json_text


def test_process_review_action_approve_updates_json_and_progress(tmp_path):
    result = _sample_result(requires_review=True)
    output_dir = tmp_path / "output"

    updated, json_text, _, panel = process_review_action(
        result,
        None,
        "block_001",
        action="approve",
        output_dir=output_dir,
    )

    assert updated is not None
    assert updated.text_blocks[0].review.user_action == "approved"
    assert '"user_action": "approved"' in json_text
    assert panel[-1] == "Review complete\nReviewed 1 / 1\nRemaining 0"
    assert (output_dir / result.source_image_id / "result.json").exists()
