"""End-to-end pipeline → Renderer R0.8 demo with deterministic mock providers."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from imageloc.config import FusionSettings, PROJECT_ROOT as CONFIG_ROOT  # noqa: E402
from imageloc.pipeline.orchestrator import PipelineOrchestrator  # noqa: E402
from imageloc.providers.easyocr_provider import EasyOCRProvider  # noqa: E402
from imageloc.providers.openai_translation_provider import OpenAITranslationProvider  # noqa: E402
from imageloc.testing.pipeline_demo_fixtures import (  # noqa: E402
    DEMO_OCR_RESULTS,
    DEMO_SOURCE_IMAGE_ID,
    build_demo_source_image,
    build_demo_translation_response_json,
    demo_required_asset_paths,
    make_demo_load_patch,
    sorted_demo_ocr_blocks,
)
from imageloc.utils import image_loader  # noqa: E402
from imageloc.utils.json_export import save_result  # noqa: E402

FUSION_SETTINGS = FusionSettings(
    iou_threshold=0.25,
    text_similarity_threshold=0.55,
    text_mismatch_threshold=0.5,
    low_ocr_confidence_threshold=0.5,
    low_font_size_confidence_threshold=0.4,
)


class MockVisionProvider:
    def analyze(self, image, ocr_blocks):
        from imageloc.testing.pipeline_demo_fixtures import build_demo_vision_blocks

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


def _print_block_summary(result) -> None:
    if result.renderer is None:
        return
    by_id = {item.block_id: item for item in result.renderer.block_results}
    print("Block summary:")
    for block in result.text_blocks:
        diag = by_id.get(block.id)
        if diag is None:
            continue
        print(
            f"  {block.id}: {block.original_text!r} -> {block.translated_text!r} | "
            f"restore={diag.restore_success} render={diag.render_success} status={diag.status}"
        )


def _print_required_assets() -> None:
    print("Required demo assets:")
    for name, relative_path in demo_required_asset_paths().items():
        absolute_path = CONFIG_ROOT / relative_path
        if absolute_path.is_file():
            print(f"  [OK] {absolute_path} ({absolute_path.stat().st_size} bytes)")
        else:
            print(f"  [MISSING] {absolute_path}")


@patch("easyocr.Reader")
def _run_demo(mock_reader_cls: MagicMock) -> None:
    output_dir = CONFIG_ROOT / "data" / "output"
    demo_dir = output_dir / DEMO_SOURCE_IMAGE_ID
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    pipeline_input_path = demo_dir / ".pipeline_input.png"
    build_demo_source_image().save(pipeline_input_path, format="PNG")

    ocr_blocks = sorted_demo_ocr_blocks()
    assert [block.raw_text for block in ocr_blocks] == [
        "SALE",
        "DETAILS",
        "HOT",
        "FX",
        "EMPTY",
        "FAIL",
    ]

    mock_reader = MagicMock()
    mock_reader.readtext.return_value = DEMO_OCR_RESULTS
    mock_reader_cls.return_value = mock_reader

    orchestrator = PipelineOrchestrator(
        ocr_provider=EasyOCRProvider(languages=["en"]),
        vision_provider=MockVisionProvider(),
        translation_provider=OpenAITranslationProvider(
            api_key="demo-key",
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

    with patch("imageloc.pipeline.orchestrator.load_image", patched_load):
        result = orchestrator.run(
            pipeline_input_path,
            target_language="EN",
            source_language="EN",
            is_test_case=True,
            save_render_assets=True,
        )

    save_result(result, output_dir)
    pipeline_input_path.unlink(missing_ok=True)

    print("Pipeline -> Renderer demo complete.")
    print(f"- pipeline_success: {result.pipeline_success}")
    print(f"- pipeline_completion_status: {result.pipeline_completion_status}")
    print(f"- source_image_id: {result.source_image_id}")
    print(f"- blocks: {len(result.text_blocks)}")
    if result.renderer is not None:
        print(f"- completion_status: {result.renderer.completion_status}")
        print(f"- rendered: {result.renderer.rendered_block_count}")
        print(f"- skipped: {result.renderer.skipped_block_count}")
        print(f"- failed_block_ids: {result.renderer.failed_block_ids}")
        print(f"- warnings: {result.renderer.warnings}")
    _print_block_summary(result)
    _print_required_assets()


def main() -> None:
    _run_demo()


if __name__ == "__main__":
    main()
