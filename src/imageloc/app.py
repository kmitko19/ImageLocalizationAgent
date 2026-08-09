"""Gradio GUI — единственная точка входа MVP.

Запуск: python -m src.imageloc.app
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from imageloc.config import PROJECT_ROOT, Settings, load_settings
from imageloc.models.result import PipelineResult
from imageloc.pipeline.orchestrator import PipelineOrchestrator
from imageloc.providers.easyocr_provider import EasyOCRProvider
from imageloc.providers.openai_translation_provider import OpenAITranslationProvider
from imageloc.providers.openai_vision_provider import OpenAIVisionProvider
from imageloc.utils.json_export import result_to_json_string, save_result
from imageloc.utils.preview import draw_preview, load_image as load_preview_image
from imageloc.utils.review_workflow import (
    approve_block,
    build_block_display,
    crop_block_image,
    edit_and_approve_block,
    next_pending_block_id,
    reject_block,
    review_progress,
    review_queue,
    save_reviewed_result,
)
from imageloc.utils.security import sanitize_message


def build_orchestrator(settings: Settings) -> PipelineOrchestrator:
    """Wire concrete providers from application settings."""
    return PipelineOrchestrator(
        ocr_provider=EasyOCRProvider(languages=settings.ocr_languages),
        vision_provider=OpenAIVisionProvider(
            api_key=settings.openai.api_key,
            model=settings.openai.vision_model,
        ),
        translation_provider=OpenAITranslationProvider(
            api_key=settings.openai.api_key,
            model=settings.openai.translation_model,
        ),
        fusion_settings=settings.fusion,
        output_dir=settings.paths.output_dir,
    )


def _resolve_source_language(source_language_mode: str) -> tuple[str | None, str]:
    mode = source_language_mode.strip().lower()
    if mode == "auto":
        return None, "auto"
    return mode.upper(), mode


def _resolve_target_language(target_language: str) -> str:
    return target_language.strip().upper()


def _resolve_image_path_for_preview(result: PipelineResult, upload_path: Path) -> Path:
    stored_path = PROJECT_ROOT / result.source_image.path
    if stored_path.exists():
        return stored_path
    return upload_path


def run_analysis(
    image_path: str | None,
    target_language: str,
    source_language_mode: str,
    *,
    orchestrator: PipelineOrchestrator,
    output_dir: Path,
) -> tuple[PipelineResult | None, str, Image.Image | None, str, str | None]:
    """Run the analysis pipeline and prepare JSON + bbox preview for the GUI."""
    if not image_path:
        return None, "", None, "Upload an image first.", None

    upload_path = Path(image_path)
    if not upload_path.exists():
        return None, "", None, f"Image not found: {upload_path.name}", None

    source_language, source_language_mode = _resolve_source_language(source_language_mode)
    target_lang = _resolve_target_language(target_language)

    result = orchestrator.run(
        upload_path,
        target_language=target_lang,
        source_language=source_language,
        source_language_mode=source_language_mode,
        is_test_case=False,
    )
    save_result(result, output_dir)

    preview_source = _resolve_image_path_for_preview(result, upload_path)
    preview_image = draw_preview(
        load_preview_image(preview_source),
        result.text_blocks,
    )
    json_text = result_to_json_string(result)
    status = (
        f"Analysis complete: {result.stats.total_blocks} blocks, "
        f"{result.stats.blocks_requiring_review} require review. "
        f"Saved to {output_dir / result.source_image_id / 'result.json'}"
    )
    return result, json_text, preview_image, status, str(preview_source)


def _empty_review_outputs() -> tuple:
    return (
        gr.update(choices=[], value=None),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        None,
        "Reviewed 0 / 0\nRemaining 0",
    )


def build_review_panel(
    result: PipelineResult | None,
    source_image_path: str | None,
    current_block_id: str | None,
) -> tuple:
    if result is None:
        return _empty_review_outputs()

    progress = review_progress(result)
    pending = review_queue(result)
    choices = [block.id for block in pending]

    if progress.is_complete:
        return (
            gr.update(choices=[], value=None),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            None,
            progress.label,
        )

    if not choices:
        return _empty_review_outputs()

    selected_id = current_block_id if current_block_id in choices else choices[0]
    block = next(block for block in pending if block.id == selected_id)
    display = build_block_display(block)

    crop_image = None
    if source_image_path and Path(source_image_path).exists():
        source = load_preview_image(source_image_path)
        crop_image = crop_block_image(source, block.bbox)

    bbox_text = f"x={display.bbox.x}, y={display.bbox.y}, w={display.bbox.width}, h={display.bbox.height}"

    return (
        gr.update(choices=choices, value=selected_id),
        display.block_id,
        display.ocr_raw_text,
        display.vision_corrected_text or "",
        display.original_text,
        display.translated_text or "",
        ", ".join(display.review_reasons),
        f"{display.ocr_confidence:.3f}",
        f"{display.vision_confidence:.3f}" if display.vision_confidence is not None else "",
        bbox_text,
        crop_image,
        progress.label,
    )


def process_review_action(
    result: PipelineResult | None,
    source_image_path: str | None,
    current_block_id: str | None,
    *,
    action: str,
    edited_original: str | None = None,
    edited_translation: str | None = None,
    output_dir: Path,
) -> tuple[PipelineResult | None, str, str | None, tuple]:
    if result is None or not current_block_id:
        review_outputs = build_review_panel(result, source_image_path, None)
        return result, result_to_json_string(result) if result else "", source_image_path, review_outputs

    if action == "approve":
        updated = approve_block(result, current_block_id)
    elif action == "edit":
        updated = edit_and_approve_block(
            result,
            current_block_id,
            original_text=edited_original or "",
            translated_text=edited_translation,
        )
    elif action == "reject":
        updated = reject_block(result, current_block_id)
    else:
        raise ValueError(f"Unknown review action: {action}")

    save_reviewed_result(updated, output_dir)
    next_id = next_pending_block_id(updated, current_block_id)
    review_outputs = build_review_panel(updated, source_image_path, next_id)
    return updated, result_to_json_string(updated), source_image_path, review_outputs


RESULT_JSON_CSS = """
.block.result-json-fixed {
    max-height: 480px !important;
    align-self: flex-start;
    overflow: hidden;
}
.block.result-json-fixed .wrap {
    max-height: 480px;
    height: auto !important;
    flex-grow: 0 !important;
}
.block.result-json-fixed .codemirror-wrapper {
    max-height: 420px;
    overflow-x: auto;
    overflow-y: auto;
    flex-grow: 0 !important;
}
.block.result-json-fixed .cm-editor {
    height: auto !important;
    max-height: 420px;
}
.block.result-json-fixed .cm-scroller {
    max-height: 420px !important;
    overflow-x: auto !important;
    overflow-y: auto !important;
}
.block.result-json-fixed .cm-content {
    min-height: unset !important;
}
"""


def create_app(settings: Settings | None = None) -> gr.Blocks:
    """Build the Gradio Blocks interface for the analysis MVP."""
    resolved_settings = settings or load_settings()
    orchestrator = build_orchestrator(resolved_settings)
    output_dir = resolved_settings.paths.output_dir

    with gr.Blocks(title="Image Localization Agent") as app:
        gr.Markdown(
            "# Image Localization Agent\n"
            "Upload an image, choose languages, run analysis, then review flagged blocks."
        )

        result_state = gr.State(None)
        source_image_state = gr.State(None)

        with gr.Row():
            image_input = gr.Image(
                label="Upload image",
                type="filepath",
                sources=["upload"],
                height=360,
            )
            with gr.Column():
                source_lang = gr.Dropdown(
                    choices=resolved_settings.source_languages,
                    value="auto",
                    label="Source language",
                )
                target_lang = gr.Dropdown(
                    choices=resolved_settings.target_languages,
                    value=resolved_settings.target_languages[0],
                    label="Target language",
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
                status_output = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("## Manual Review")

        with gr.Row():
            review_block_dropdown = gr.Dropdown(label="Blocks requiring review", choices=[], value=None)
            review_progress_output = gr.Textbox(label="Review progress", interactive=False)

        with gr.Row():
            with gr.Column():
                review_block_id = gr.Textbox(label="Block ID", interactive=False)
                review_ocr_raw = gr.Textbox(label="OCR raw text", interactive=False)
                review_vision_corrected = gr.Textbox(label="Vision corrected text", interactive=False)
                review_reasons = gr.Textbox(label="Review reasons", interactive=False)
                review_ocr_conf = gr.Textbox(label="OCR confidence", interactive=False)
                review_vision_conf = gr.Textbox(label="Vision confidence", interactive=False)
                review_bbox = gr.Textbox(label="BBox", interactive=False)
            with gr.Column():
                review_crop = gr.Image(label="Block crop", type="pil", height=220)
                edit_original = gr.Textbox(label="Original text (editable)", lines=2)
                edit_translation = gr.Textbox(label="Translated text (editable)", lines=2)
                with gr.Row():
                    approve_btn = gr.Button("Approve", variant="primary")
                    edit_approve_btn = gr.Button("Edit + Approve")
                    reject_btn = gr.Button("Reject", variant="stop")

        with gr.Row():
            preview_output = gr.Image(label="Preview (bbox + block IDs)", type="pil", height=360)
            json_output = gr.Code(
                label="Result JSON",
                language="json",
                lines=24,
                elem_classes=["result-json-fixed"],
            )

        review_outputs = [
            review_block_dropdown,
            review_block_id,
            review_ocr_raw,
            review_vision_corrected,
            edit_original,
            edit_translation,
            review_reasons,
            review_ocr_conf,
            review_vision_conf,
            review_bbox,
            review_crop,
            review_progress_output,
        ]

        def on_analyze(
            image: str | None,
            target: str,
            source: str,
        ) -> tuple:
            try:
                result, json_text, preview_image, status, source_path = run_analysis(
                    image,
                    target,
                    source,
                    orchestrator=orchestrator,
                    output_dir=output_dir,
                )
                review_panel = build_review_panel(result, source_path, next_pending_block_id(result))
                return (
                    preview_image,
                    json_text,
                    status,
                    result,
                    source_path,
                    *review_panel,
                )
            except Exception as exc:
                empty = _empty_review_outputs()
                return None, "", sanitize_message(str(exc)), None, None, *empty

        analyze_btn.click(
            fn=on_analyze,
            inputs=[image_input, target_lang, source_lang],
            outputs=[
                preview_output,
                json_output,
                status_output,
                result_state,
                source_image_state,
                *review_outputs,
            ],
        )

        def on_select_block(result, source_path, selected_id):
            panel = build_review_panel(result, source_path, selected_id)
            return panel

        review_block_dropdown.change(
            fn=on_select_block,
            inputs=[result_state, source_image_state, review_block_dropdown],
            outputs=review_outputs,
        )

        def on_approve(result, source_path, block_id):
            updated, json_text, source_path, panel = process_review_action(
                result,
                source_path,
                block_id,
                action="approve",
                output_dir=output_dir,
            )
            return updated, json_text, *panel

        def on_edit_approve(result, source_path, block_id, original, translation):
            updated, json_text, source_path, panel = process_review_action(
                result,
                source_path,
                block_id,
                action="edit",
                edited_original=original,
                edited_translation=translation,
                output_dir=output_dir,
            )
            return updated, json_text, *panel

        def on_reject(result, source_path, block_id):
            updated, json_text, source_path, panel = process_review_action(
                result,
                source_path,
                block_id,
                action="reject",
                output_dir=output_dir,
            )
            return updated, json_text, *panel

        approve_btn.click(
            fn=on_approve,
            inputs=[result_state, source_image_state, review_block_id],
            outputs=[result_state, json_output, *review_outputs],
        )
        edit_approve_btn.click(
            fn=on_edit_approve,
            inputs=[
                result_state,
                source_image_state,
                review_block_id,
                edit_original,
                edit_translation,
            ],
            outputs=[result_state, json_output, *review_outputs],
        )
        reject_btn.click(
            fn=on_reject,
            inputs=[result_state, source_image_state, review_block_id],
            outputs=[result_state, json_output, *review_outputs],
        )

    return app


def main() -> None:
    settings = load_settings()
    app = create_app(settings)
    app.launch(server_port=settings.gui_port, share=False, css=RESULT_JSON_CSS)


if __name__ == "__main__":
    main()
