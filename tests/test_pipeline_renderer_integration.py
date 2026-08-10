"""Pipeline → Renderer R0.8 integration tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from imageloc.config import FusionSettings
from imageloc.models.raw_blocks import RawVisionBlock
from imageloc.pipeline.orchestrator import PipelineOrchestrator, run_pipeline
from imageloc.providers.easyocr_provider import EasyOCRProvider, parse_easyocr_results
from imageloc.providers.openai_translation_provider import OpenAITranslationProvider
from imageloc.renderer.orchestrator import BLOCK_STATUS_FAILED, BLOCK_STATUS_RENDERED, BLOCK_STATUS_SKIPPED
from imageloc.renderer.text_renderer import resolve_localized_image_path
from imageloc.utils.json_export import save_result
from imageloc.utils import image_loader
from imageloc.utils.review_workflow import approve_block, review_queue
from imageloc.renderer.background_restorer import resolve_clean_background_path

FUSION_SETTINGS = FusionSettings(
    iou_threshold=0.25,
    text_similarity_threshold=0.55,
    text_mismatch_threshold=0.5,
    low_ocr_confidence_threshold=0.5,
    low_font_size_confidence_threshold=0.4,
)

pytestmark = pytest.mark.skipif(
    not (Path(__import__("os").environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeui.ttf").is_file(),
    reason="Windows Unicode fonts required for pipeline renderer integration tests",
)


MOCK_OCR_RESULTS = [
    (
        [[20.0, 20.0], [220.0, 20.0], [220.0, 80.0], [20.0, 80.0]],
        "SALE",
        0.92,
    ),
    (
        [[240.0, 20.0], [470.0, 20.0], [470.0, 80.0], [240.0, 80.0]],
        "Long promo text",
        0.88,
    ),
    (
        [[490.0, 20.0], [680.0, 20.0], [680.0, 80.0], [490.0, 80.0]],
        "HOT",
        0.9,
    ),
    (
        [[450.0, 130.0], [540.0, 130.0], [540.0, 200.0], [450.0, 200.0]],
        "FAILME",
        0.85,
    ),
]

INTEGRATION_TRANSLATIONS_BY_TEXT = {
    "SALE": "SALE",
    "Long promo text": "Very long translated promo copy",
    "HOT": "HOT",
    "FAILME": "Too long for tiny block",
}


def _build_integration_translation_json() -> str:
    blocks = parse_easyocr_results(MOCK_OCR_RESULTS)
    return json.dumps(
        {
            "translations": [
                {
                    "id": block.id,
                    "original_text": block.raw_text,
                    "translated_text": INTEGRATION_TRANSLATIONS_BY_TEXT[block.raw_text],
                    "detected_language": "EN",
                    "translation_confidence": 0.9,
                }
                for block in blocks
            ]
        }
    )


def _build_integration_vision_blocks(ocr_blocks) -> list[RawVisionBlock]:
    return [
        RawVisionBlock(
            ocr_block_id=block.id,
            text=block.raw_text,
            corrected_text=block.raw_text,
            semantic_role="heading" if block.raw_text == "SALE" else "label",
            language="EN",
            confidence=0.9,
            alignment="left" if block.raw_text == "Long promo text" else "center",
        )
        for block in ocr_blocks
    ]


class MockVisionProvider:
    def analyze(self, image, ocr_blocks):
        return _build_integration_vision_blocks(ocr_blocks)


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
    image = Image.new("RGB", (920, 420), color=(245, 240, 232))
    draw = ImageDraw.Draw(image)
    specs = [
        ((20, 20, 220, 80), (255, 255, 255), "SALE"),
        ((240, 20, 470, 80), (255, 255, 255), "LONG"),
        ((490, 20, 680, 80), (180, 60, 60), "HOT"),
        ((450, 130, 540, 200), (255, 255, 255), "FAIL"),
    ]
    for rect, fill, label in specs:
        draw.rectangle(rect, fill=fill)
        draw.text((rect[0] + 10, rect[1] + 8), label, fill=(240, 240, 240) if fill != (255, 255, 255) else (20, 20, 20))
    image.save(path)


def _build_orchestrator(output_dir: Path) -> PipelineOrchestrator:
    translation_client = _mock_openai_client(_build_integration_translation_json())
    return PipelineOrchestrator(
        ocr_provider=EasyOCRProvider(languages=["en"]),
        vision_provider=MockVisionProvider(),
        translation_provider=OpenAITranslationProvider(api_key="test-key", client=translation_client),
        fusion_settings=FUSION_SETTINGS,
        output_dir=output_dir,
    )


@patch("easyocr.Reader")
def test_complete_pipeline_reaches_renderer(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    assert result.pipeline_success is True
    assert result.renderer is not None
    assert result.renderer.success is True
    assert result.localized_image is not None


@patch("easyocr.Reader")
def test_renderer_receives_translated_blocks(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    sale = next(block for block in result.text_blocks if block.original_text == "SALE")
    assert sale.translated_text == "SALE"
    assert result.renderer is not None
    assert result.renderer.rendered_block_count >= 1


@patch("easyocr.Reader")
def test_localized_image_in_pipeline_result(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    assert isinstance(result.localized_image, Image.Image)
    assert result.localized_image.size == (920, 420)


@patch("easyocr.Reader")
def test_renderer_diagnostics_preserved(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    assert result.renderer is not None
    assert len(result.renderer.block_results) == len(result.text_blocks)
    assert result.renderer.rendered_block_count >= 1


@patch("easyocr.Reader")
def test_multiple_blocks_rendered(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    assert len(result.text_blocks) == 4
    assert result.renderer is not None
    assert result.renderer.rendered_block_count >= 2


@patch("easyocr.Reader")
def test_skipped_non_text_block(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "heart.png"
    Image.new("RGB", (200, 200), color="white").save(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[80.0, 80.0], [120.0, 80.0], [120.0, 120.0], [80.0, 120.0]], "~", 0.2),
    ]
    mock_reader_cls.return_value = mock_reader

    translation_client = _mock_openai_client(json.dumps({"translations": []}))
    orchestrator = PipelineOrchestrator(
        ocr_provider=EasyOCRProvider(languages=["en"]),
        vision_provider=MockVisionProvider(),
        translation_provider=OpenAITranslationProvider(api_key="test-key", client=translation_client),
        fusion_settings=FUSION_SETTINGS,
        output_dir=output_dir,
    )
    orchestrator.vision_provider = type(
        "HeartVision",
        (),
        {
            "analyze": lambda self, image, ocr_blocks: [
                RawVisionBlock(
                    ocr_block_id="block_001",
                    text="~",
                    corrected_text="~",
                    semantic_role="other",
                    language=None,
                    confidence=0.99,
                )
            ]
        },
    )()

    result = orchestrator.run(image_path, target_language="EN", is_test_case=True)

    assert result.text_blocks[0].translated_text is None
    assert result.renderer is not None
    assert result.renderer.skipped_block_count >= 1
    translation_client.chat.completions.create.assert_not_called()


@patch("easyocr.Reader")
def test_controlled_renderer_block_failure(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    assert result.pipeline_success is True
    assert result.renderer is not None
    assert "block_004" in result.renderer.failed_block_ids
    failed = next(item for item in result.renderer.block_results if item.block_id == "block_004")
    assert failed.status == BLOCK_STATUS_FAILED


@patch("easyocr.Reader")
def test_renderer_fatal_failure_is_controlled(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    orchestrator = _build_orchestrator(output_dir)

    with patch(
        "imageloc.pipeline.orchestrator.render_image_localization",
        return_value=type(
            "FatalRendererResult",
            (),
                {
                    "success": False,
                    "completion_status": "failed",
                    "fatal_error": "renderer fatal",
                "image": Image.new("RGB", (920, 420)),
                "clean_background": Image.new("RGB", (920, 420)),
                "block_results": [],
                "warnings": [],
                "failed_block_ids": [],
                "rendered_block_count": 0,
                "skipped_block_count": 0,
                "source_image_id": "x",
                "output_paths": {},
            },
        )(),
    ):
        result = orchestrator.run(image_path, target_language="EN", is_test_case=True)

    assert result.pipeline_success is False
    assert result.renderer is not None
    assert result.renderer.fatal_error == "renderer fatal"


@patch("easyocr.Reader")
def test_translation_failure_does_not_invoke_renderer(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    orchestrator = _build_orchestrator(output_dir)

    with patch.object(orchestrator.translation_provider, "translate_fused_blocks", side_effect=RuntimeError("translation failed")):
        result = orchestrator.run(image_path, target_language="EN", is_test_case=True)

    assert result.pipeline_success is False
    assert result.text_blocks == []
    assert result.renderer is not None
    assert result.renderer.fatal_error == "translation failed"


@patch("easyocr.Reader")
def test_source_image_id_preserved(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    assert result.renderer is not None
    assert result.source_image_id
    if result.localized_image_path:
        assert result.source_image_id in result.localized_image_path.replace("\\", "/")


@patch("easyocr.Reader")
def test_masks_available_via_render_geometry(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    assert any(
        block.render_geometry is not None and block.render_geometry.mask_path
        for block in result.text_blocks
    )


@patch("easyocr.Reader")
def test_block_order_preserved(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    assert [block.id for block in result.text_blocks] == [block.id for block in parse_easyocr_results(MOCK_OCR_RESULTS)]


@patch("easyocr.Reader")
def test_rgba_alpha_preserved(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_rgba.png"
    Image.new("RGBA", (920, 420), (245, 240, 232, 128)).save(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS[:1]
    mock_reader_cls.return_value = mock_reader

    original_load = image_loader.load_image

    def rgba_load(input_path, output_dir, is_test_case):
        loaded = original_load(input_path, output_dir, is_test_case)
        rgba = Image.open(input_path).convert("RGBA")
        return image_loader.LoadedImage(
            image=rgba,
            source_image_id=loaded.source_image_id,
            source_image=loaded.source_image.model_copy(update={"format": "PNG"}),
        )

    orchestrator = _build_orchestrator(output_dir)
    with patch("imageloc.pipeline.orchestrator.load_image", rgba_load):
        result = orchestrator.run(image_path, target_language="EN", is_test_case=True)

    assert result.localized_image is not None
    assert result.localized_image.mode == "RGBA"


@patch("easyocr.Reader")
def test_canvas_size_unchanged(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)

    assert result.localized_image is not None
    assert result.localized_image.size == (920, 420)


@patch("easyocr.Reader")
def test_deterministic_repeated_run(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    orchestrator = _build_orchestrator(output_dir)
    first = np.array(orchestrator.run(image_path, target_language="EN", is_test_case=True).localized_image)
    second = np.array(orchestrator.run(image_path, target_language="EN", is_test_case=True).localized_image)
    assert np.array_equal(first, second)


@patch("easyocr.Reader")
def test_save_assets_false(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(
        image_path,
        target_language="EN",
        is_test_case=True,
        save_render_assets=False,
    )

    assert result.localized_image_path is None
    assert not list(output_dir.glob("*/render_assets/localized_image.png"))


@patch("easyocr.Reader")
def test_save_assets_true(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(
        image_path,
        target_language="EN",
        is_test_case=True,
        save_render_assets=True,
    )

    assert result.localized_image_path is not None
    assert result.clean_background_path is not None
    assert result.renderer is not None
    assert "localized_image" in result.renderer.output_paths
    localized = resolve_localized_image_path(result.localized_image_path)
    clean = resolve_clean_background_path(result.clean_background_path)
    assert localized.is_file()
    assert clean.is_file()
    assert localized.parent == clean.parent


@patch("easyocr.Reader")
def test_review_workflow_still_works(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS[:1]
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(image_path, target_language="EN", is_test_case=True)
    pending = review_queue(result)
    updated = approve_block(result, pending[0].id if pending else result.text_blocks[0].id)
    assert updated.text_blocks[0].review.user_action == "approved"


@patch("easyocr.Reader")
def test_render_localized_false_skips_renderer(mock_reader_cls: MagicMock, tmp_path: Path):
    image_path = tmp_path / "pipeline_render.png"
    _make_demo_image(image_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = MOCK_OCR_RESULTS[:1]
    mock_reader_cls.return_value = mock_reader

    result = _build_orchestrator(output_dir).run(
        image_path,
        target_language="EN",
        is_test_case=True,
        render_localized=False,
    )

    assert result.renderer is None
    assert result.localized_image is None
