"""R0.8 regression tests for demo contract, review policy, and block associations."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from imageloc.config import FusionSettings
from imageloc.models.raw_blocks import BoundingBox
from imageloc.models.text_block import (
    BackgroundInfo,
    OCRData,
    RenderGeometry,
    RenderHints,
    ReviewInfo,
    TextBlock,
    VisualStyle,
)
from imageloc.pipeline.orchestrator import PipelineOrchestrator
from imageloc.providers.easyocr_provider import EasyOCRProvider, parse_easyocr_results
from imageloc.providers.openai_translation_provider import OpenAITranslationProvider
from imageloc.renderer.background_restorer import RESTORE_MIN_MASK_CONFIDENCE
from imageloc.renderer.orchestrator import (
    BLOCK_STATUS_FAILED,
    BLOCK_STATUS_PENDING_REVIEW,
    BLOCK_STATUS_REJECTED,
    BLOCK_STATUS_RENDERED,
    BLOCK_STATUS_SKIPPED,
    COMPLETION_STATUS_PARTIAL,
    render_image_localization,
)
from imageloc.renderer.review_policy import final_render_skip_reason
from imageloc.testing.pipeline_demo_fixtures import (
    DEMO_OCR_RESULTS,
    DEMO_SOURCE_IMAGE_ID,
    build_demo_source_image,
    build_demo_translation_response_json,
    build_demo_vision_blocks,
    make_demo_load_patch,
    sorted_demo_ocr_blocks,
)
from imageloc.utils import image_loader
from imageloc.utils.json_export import save_result
from imageloc.utils.text_mask_analyzer import (
    MASK_FOREGROUND_VALUE,
    MASK_SAVE_MIN_CONFIDENCE,
    analyze_text_mask_for_block,
)
from imageloc.fusion.ocr_vision_fusion import FusedBlock

FUSION_SETTINGS = FusionSettings(
    iou_threshold=0.25,
    text_similarity_threshold=0.55,
    text_mismatch_threshold=0.5,
    low_ocr_confidence_threshold=0.5,
    low_font_size_confidence_threshold=0.4,
)

pytestmark = pytest.mark.skipif(
    not (Path(__import__("os").environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf").is_file(),
    reason="Windows Unicode fonts required for R0.8 regression tests",
)


class MockVisionProvider:
    def analyze(self, image, ocr_blocks):
        return build_demo_vision_blocks(ocr_blocks)


def _mock_openai_client(response_content: str) -> MagicMock:
    client = MagicMock()
    message = MagicMock()
    message.content = response_content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


def _make_demo_image(path: Path) -> None:
    build_demo_source_image().save(path, format="PNG")


def _run_demo_pipeline(tmp_path: Path):
    image_path = tmp_path / "pipeline_input.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    ocr_blocks = sorted_demo_ocr_blocks()
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = DEMO_OCR_RESULTS

    orchestrator = PipelineOrchestrator(
        ocr_provider=EasyOCRProvider(languages=["en"]),
        vision_provider=MockVisionProvider(),
        translation_provider=OpenAITranslationProvider(
            api_key="test-key",
            client=_mock_openai_client(build_demo_translation_response_json(ocr_blocks)),
        ),
        fusion_settings=FUSION_SETTINGS,
        output_dir=output_dir,
    )

    patched_load = make_demo_load_patch(
        image_loader.load_image,
        output_dir=output_dir,
        source_image_id=DEMO_SOURCE_IMAGE_ID,
    )

    with patch("easyocr.Reader", return_value=mock_reader), patch(
        "imageloc.pipeline.orchestrator.load_image",
        patched_load,
    ):
        result = orchestrator.run(
            image_path,
            target_language="EN",
            source_language="EN",
            is_test_case=True,
            save_render_assets=True,
        )

    save_result(result, output_dir)
    return result


def _block_by_text(result, text: str):
    return next(block for block in result.text_blocks if block.original_text == text)


def test_reading_order_assigns_expected_block_ids():
    blocks = sorted_demo_ocr_blocks()
    assert [block.id for block in blocks] == [
        "block_001",
        "block_002",
        "block_003",
        "block_004",
        "block_005",
        "block_006",
    ]
    assert [block.raw_text for block in blocks] == [
        "SALE",
        "DETAILS",
        "HOT",
        "FX",
        "EMPTY",
        "FAIL",
    ]


def test_translation_payload_matches_sorted_block_ids():
    blocks = sorted_demo_ocr_blocks()
    payload = json.loads(build_demo_translation_response_json(blocks))
    assert {item["id"] for item in payload["translations"]} == {block.id for block in blocks}
    by_id = {item["id"]: item for item in payload["translations"]}
    for block in blocks:
        assert by_id[block.id]["original_text"] == block.raw_text


@patch("easyocr.Reader")
def test_demo_output_assets_contract(mock_reader_cls: MagicMock, tmp_path: Path):
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = DEMO_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _run_demo_pipeline(tmp_path)
    demo_root = tmp_path / "output" / DEMO_SOURCE_IMAGE_ID
    required_files = {
        "source": demo_root / "source.png",
        "result": demo_root / "result.json",
        "clean_background": demo_root / "render_assets" / "clean_background.png",
        "localized_image": demo_root / "render_assets" / "localized_image.png",
        "renderer_result": demo_root / "render_assets" / "renderer_result.json",
    }

    for asset_name, asset_path in required_files.items():
        assert asset_path.is_file(), f"missing required demo asset: {asset_name} -> {asset_path}"

    source = Image.open(demo_root / "source.png")
    localized = Image.open(demo_root / "render_assets" / "localized_image.png")
    try:
        assert source.mode == "RGB"
        assert localized.mode == "RGB"
        assert source.size == localized.size
        assert source.size == (result.source_image.width, result.source_image.height)
        assert result.source_image.path.endswith("source.png")
        assert result.source_image.filename == "source.png"
    finally:
        source.close()
        localized.close()


@patch("easyocr.Reader")
def test_demo_contract_rendered_skipped_failed(mock_reader_cls: MagicMock, tmp_path: Path):
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = DEMO_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _run_demo_pipeline(tmp_path)

    assert result.renderer is not None
    assert result.renderer.rendered_block_count == 4
    assert result.renderer.skipped_block_count == 1
    assert result.renderer.failed_block_ids == ["block_006"]
    assert result.pipeline_completion_status == COMPLETION_STATUS_PARTIAL

    fail_block = next(item for item in result.renderer.block_results if item.block_id == "block_006")
    assert fail_block.status == BLOCK_STATUS_FAILED
    assert fail_block.restore_success is True
    assert fail_block.render_success is False
    assert fail_block.failure_reason == "text_does_not_fit_at_minimum_font_size"

    empty_block = next(item for item in result.renderer.block_results if item.block_id == "block_005")
    assert empty_block.status == BLOCK_STATUS_SKIPPED

    assert not any("restore[" in warning for warning in result.renderer.warnings)


@patch("easyocr.Reader")
def test_block_translation_matches_spatial_region(mock_reader_cls: MagicMock, tmp_path: Path):
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = DEMO_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _run_demo_pipeline(tmp_path)

    sale = _block_by_text(result, "SALE")
    hot = _block_by_text(result, "HOT")
    fail = _block_by_text(result, "FAIL")

    assert sale.translated_text == "SALE"
    assert sale.bbox.x < 100
    assert hot.translated_text == "HOT"
    assert hot.bbox.x > 400
    assert fail.translated_text == "Too long for tiny block"
    assert fail.bbox.width < 120


@patch("easyocr.Reader")
def test_mask_path_matches_block_id(mock_reader_cls: MagicMock, tmp_path: Path):
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = DEMO_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _run_demo_pipeline(tmp_path)

    for block in result.text_blocks:
        if block.render_geometry is None or block.render_geometry.mask_path is None:
            continue
        assert block.id in block.render_geometry.mask_path


@patch("easyocr.Reader")
def test_multiline_details_block_renders(mock_reader_cls: MagicMock, tmp_path: Path):
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = DEMO_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _run_demo_pipeline(tmp_path)
    details = _block_by_text(result, "DETAILS")
    diag = next(item for item in result.renderer.block_results if item.block_id == details.id)

    assert details.render_hints.max_line_count >= 1
    assert diag.status == BLOCK_STATUS_RENDERED


def test_light_text_on_red_solid_background_mask_confidence(tmp_path: Path):
    image = Image.new("RGB", (200, 80), color=(180, 60, 60))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "HOT", fill=(240, 240, 240))
    bbox = BoundingBox(x=10, y=10, width=180, height=60)
    block = FusedBlock(
        id="block_hot",
        text="HOT",
        source_language="EN",
        confidence=0.9,
        bbox=bbox,
        polygon=[(10, 10), (190, 10), (190, 70), (10, 70)],
        ocr=OCRData(raw_text="HOT", confidence=0.9),
        vision=None,
        review=ReviewInfo(),
    )
    visual = VisualStyle(
        text_color="#F0F0F0",
        background_color="#B43C3C",
        estimated_text_height_px=40,
    )
    result = analyze_text_mask_for_block(
        image,
        block,
        visual=visual,
        render_hints=RenderHints(available_width_px=180, available_height_px=60),
        background=BackgroundInfo(
            background_type="solid",
            dominant_color="#B43C3C",
            classification_confidence=0.9,
            preferred_restore_method="fill",
        ),
        source_image_id="red-hot-case",
        output_dir=tmp_path,
    )
    assert result.mask_analysis is not None
    assert result.mask_analysis.confidence >= MASK_SAVE_MIN_CONFIDENCE
    assert result.render_geometry is not None
    assert (result.render_geometry.mask_confidence or 0.0) >= RESTORE_MIN_MASK_CONFIDENCE


def test_renderer_summary_counts_match_block_results():
    image = Image.new("RGB", (320, 160), color=(245, 240, 232))
    blocks = [
        TextBlock(
            id="block_001",
            original_text="A",
            translated_text="A",
            source_language="EN",
            confidence=0.9,
            bbox=BoundingBox(x=10, y=10, width=80, height=30),
            polygon=[(10, 10), (90, 10), (90, 40), (10, 40)],
            ocr=OCRData(raw_text="A", confidence=0.9),
            visual=VisualStyle(
                text_color="#111111",
                background_color="#FFFFFF",
                estimated_text_height_px=24,
                font_size_estimate=18,
                alignment="center",
            ),
            render_hints=RenderHints(available_width_px=80, available_height_px=30, max_line_count=1),
            review=ReviewInfo(),
            background=BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill"),
            render_geometry=RenderGeometry(
                mask_bbox=BoundingBox(x=10, y=10, width=80, height=30),
                mask_confidence=0.9,
            ),
        )
    ]
    mask = np.zeros((30, 80), dtype=np.uint8)
    mask[8:22, 20:60] = MASK_FOREGROUND_VALUE
    result = render_image_localization(image, blocks, masks={"block_001": mask})
    rendered = sum(1 for item in result.block_results if item.status == BLOCK_STATUS_RENDERED)
    skipped = sum(1 for item in result.block_results if item.status == BLOCK_STATUS_SKIPPED)
    failed = [item.block_id for item in result.block_results if item.status == BLOCK_STATUS_FAILED]
    assert result.rendered_block_count == rendered
    assert result.skipped_block_count == skipped
    assert result.failed_block_ids == failed


def test_pending_review_block_not_final_rendered():
    image = Image.new("RGB", (200, 80), color=(255, 255, 255))
    block = TextBlock(
        id="block_review",
        original_text="SALE",
        translated_text="SALE",
        source_language="EN",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=10, width=120, height=40),
        polygon=[(10, 10), (130, 10), (130, 50), (10, 50)],
        ocr=OCRData(raw_text="SALE", confidence=0.9),
        visual=VisualStyle(text_color="#111111", background_color="#FFFFFF", estimated_text_height_px=30, font_size_estimate=20),
        render_hints=RenderHints(available_width_px=120, available_height_px=40, max_line_count=1),
        review=ReviewInfo(requires_review=True, reasons=["text_mismatch"], user_confirmed=None),
        background=BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill"),
        render_geometry=RenderGeometry(mask_bbox=BoundingBox(x=10, y=10, width=120, height=40), mask_confidence=0.9),
    )
    assert final_render_skip_reason(block) == BLOCK_STATUS_PENDING_REVIEW
    result = render_image_localization(image, [block])
    assert result.block_results[0].status == BLOCK_STATUS_PENDING_REVIEW
    assert result.block_results[0].render_success is False


def test_confirmed_review_block_renders():
    image = Image.new("RGB", (200, 80), color=(255, 255, 255))
    ImageDraw.Draw(image).text((20, 15), "OLD", fill=(0, 0, 0))
    block = TextBlock(
        id="block_ok",
        original_text="OLD",
        translated_text="NEW",
        source_language="EN",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=10, width=120, height=40),
        polygon=[(10, 10), (130, 10), (130, 50), (10, 50)],
        ocr=OCRData(raw_text="OLD", confidence=0.9),
        visual=VisualStyle(
            text_color="#111111",
            background_color="#FFFFFF",
            estimated_text_height_px=30,
            font_size_estimate=20,
            alignment="center",
        ),
        render_hints=RenderHints(available_width_px=120, available_height_px=40, max_line_count=1),
        review=ReviewInfo(requires_review=True, reasons=["text_mismatch"], user_confirmed=True, user_action="approved"),
        background=BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill"),
        render_geometry=RenderGeometry(mask_bbox=BoundingBox(x=10, y=10, width=120, height=40), mask_confidence=0.9),
    )
    mask = np.zeros((40, 120), dtype=np.uint8)
    mask[10:30, 15:70] = MASK_FOREGROUND_VALUE
    result = render_image_localization(image, [block], masks={"block_ok": mask})
    assert result.block_results[0].status == BLOCK_STATUS_RENDERED


def test_rejected_review_block_not_rendered():
    image = Image.new("RGB", (200, 80), color=(255, 255, 255))
    block = TextBlock(
        id="block_no",
        original_text="SALE",
        translated_text="SALE",
        source_language="EN",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=10, width=120, height=40),
        polygon=[(10, 10), (130, 10), (130, 50), (10, 50)],
        ocr=OCRData(raw_text="SALE", confidence=0.9),
        visual=VisualStyle(text_color="#111111", background_color="#FFFFFF", estimated_text_height_px=30, font_size_estimate=20),
        render_hints=RenderHints(available_width_px=120, available_height_px=40, max_line_count=1),
        review=ReviewInfo(requires_review=True, reasons=["text_mismatch"], user_confirmed=False, user_action="rejected"),
        background=BackgroundInfo(background_type="solid", dominant_color="#FFFFFF", preferred_restore_method="fill"),
        render_geometry=RenderGeometry(mask_bbox=BoundingBox(x=10, y=10, width=120, height=40), mask_confidence=0.9),
    )
    result = render_image_localization(image, [block])
    assert result.block_results[0].status == BLOCK_STATUS_REJECTED


def test_restore_failure_prevents_text_overlay():
    image = Image.new("RGB", (200, 80), color=(180, 60, 60))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "HOT", fill=(240, 240, 240))
    block = TextBlock(
        id="block_hot",
        original_text="HOT",
        translated_text="SALE",
        source_language="EN",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=10, width=160, height=60),
        polygon=[(10, 10), (170, 10), (170, 70), (10, 70)],
        ocr=OCRData(raw_text="HOT", confidence=0.9),
        visual=VisualStyle(
            text_color="#F0F0F0",
            background_color="#B43C3C",
            estimated_text_height_px=40,
            font_size_estimate=24,
            alignment="center",
        ),
        render_hints=RenderHints(available_width_px=160, available_height_px=60, max_line_count=1),
        review=ReviewInfo(),
        background=BackgroundInfo(background_type="solid", dominant_color="#B43C3C", preferred_restore_method="fill"),
        render_geometry=RenderGeometry(mask_bbox=BoundingBox(x=10, y=10, width=160, height=60), mask_confidence=0.2),
    )
    result = render_image_localization(image, [block])
    assert result.block_results[0].status == BLOCK_STATUS_FAILED
    assert result.block_results[0].failure_reason == "restore_failed"
    assert np.array_equal(np.array(result.image), np.array(image))


@patch("easyocr.Reader")
def test_deterministic_repeated_demo_run(mock_reader_cls: MagicMock, tmp_path: Path):
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = DEMO_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    first = np.array(_run_demo_pipeline(tmp_path).localized_image)
    second = np.array(_run_demo_pipeline(tmp_path).localized_image)
    assert np.array_equal(first, second)


def test_reordered_ocr_input_still_maps_translation_by_block_id():
    shuffled = list(reversed(DEMO_OCR_RESULTS))
    blocks = parse_easyocr_results(shuffled)
    payload = json.loads(build_demo_translation_response_json(blocks))
    by_id = {item["id"]: item["original_text"] for item in payload["translations"]}
    for block in blocks:
        assert by_id[block.id] == block.raw_text
