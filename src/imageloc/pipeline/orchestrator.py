"""Pipeline orchestrator — wires all analysis stages into run_pipeline().

Sequence:
  load image → OCR → Vision → fusion → visual analysis → text mask / geometry
  → translation → finalize render metadata → Renderer R0.7 → PipelineResult
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from imageloc.config import FusionSettings
from imageloc.fusion.ocr_vision_fusion import FusedBlock, fuse
from imageloc.models.raw_blocks import RawVisionBlock
from imageloc.models.result import (
    PipelineResult,
    RendererBlockDiagnostics,
    RendererDiagnostics,
    Stats,
)
from imageloc.models.text_block import RenderGeometry, RenderHints, ReviewInfo, TextBlock, VisualStyle, BackgroundInfo
from imageloc.providers.base import OCRProvider, VisionProvider
from imageloc.providers.openai_translation_provider import (
    OpenAITranslationProvider,
    TranslationResultDetail,
)
from imageloc.renderer.orchestrator import RendererResult, render_image_localization, COMPLETION_STATUS_FAILED
from imageloc.utils.image_loader import load_image
from imageloc.utils.non_text_detector import mark_obvious_non_text
from imageloc.utils.render_metadata import finalize_text_block_metadata
from imageloc.utils.text_mask_analyzer import analyze_text_mask_for_block
from imageloc.utils.visual_analyzer import analyze_fused_block


@dataclass
class PipelineOrchestrator:
    """Coordinates the full MVP analysis pipeline."""

    ocr_provider: OCRProvider
    vision_provider: VisionProvider
    translation_provider: OpenAITranslationProvider
    fusion_settings: FusionSettings
    output_dir: Path

    def run(
        self,
        image_path: str | Path,
        *,
        target_language: str = "EN",
        source_language: str | None = None,
        source_language_mode: str = "auto",
        is_test_case: bool = False,
        render_localized: bool = True,
        save_render_assets: bool = False,
    ) -> PipelineResult:
        loaded = load_image(
            image_path,
            output_dir=self.output_dir,
            is_test_case=is_test_case,
        )

        try:
            ocr_blocks = self.ocr_provider.detect(loaded.image)
            vision_blocks = self.vision_provider.analyze(loaded.image, ocr_blocks)
            fusion_result = fuse(ocr_blocks, vision_blocks, self.fusion_settings)

            vision_by_block_id = _index_vision_blocks(vision_blocks)

            analyzed_blocks: list[
                tuple[
                    FusedBlock,
                    VisualStyle,
                    RenderHints,
                    BackgroundInfo | None,
                    RenderGeometry | None,
                    bool,
                ]
            ] = []
            for fused in fusion_result.fused_blocks:
                vision_raw = vision_by_block_id.get(fused.id)
                fused, skip_translation = mark_obvious_non_text(fused, vision_raw)
                visual, render_hints, background_info = analyze_fused_block(
                    loaded.image,
                    fused,
                    font_category=vision_raw.font_category if vision_raw else None,
                    font_style=vision_raw.font_style if vision_raw else None,
                    font_weight=vision_raw.font_weight if vision_raw else None,
                    alignment=vision_raw.alignment if vision_raw else None,
                )
                mask_result = analyze_text_mask_for_block(
                    loaded.image,
                    fused,
                    visual=visual,
                    render_hints=render_hints,
                    background=background_info,
                    source_image_id=loaded.source_image_id,
                    output_dir=self.output_dir,
                )
                analyzed_blocks.append(
                    (
                        fused,
                        mask_result.visual,
                        render_hints,
                        background_info,
                        mask_result.render_geometry,
                        skip_translation,
                    )
                )

            translation_details = self.translation_provider.translate_fused_blocks(
                [
                    (fused, render_hints)
                    for fused, _, render_hints, _, _, skip_translation in analyzed_blocks
                    if not skip_translation
                ],
                target_lang=target_language,
                source_lang=source_language,
            )
            translation_by_id = {item.id: item for item in translation_details}

            text_blocks = [
                finalize_text_block_metadata(
                    _build_text_block(
                        fused=fused,
                        visual=visual,
                        render_hints=render_hints,
                        translation=translation_by_id.get(fused.id),
                        skip_translation=skip_translation,
                        background=background_info,
                        render_geometry=render_geometry,
                    )
                )
                for fused, visual, render_hints, background_info, render_geometry, skip_translation in analyzed_blocks
            ]

            resolved_source_language = source_language or _detect_source_language(fusion_result.fused_blocks)

            renderer_result: RendererResult | None = None
            localized_image_path: str | None = None
            clean_background_path: str | None = None
            localized_image: Image.Image | None = None
            clean_background: Image.Image | None = None
            pipeline_success = True
            pipeline_completion_status = "complete"

            if render_localized:
                renderer_result = render_image_localization(
                    loaded.image,
                    text_blocks,
                    masks=None,
                    source_image_id=loaded.source_image_id,
                    save_assets=save_render_assets,
                    output_dir=self.output_dir,
                )
                localized_image = renderer_result.image
                clean_background = renderer_result.clean_background
                pipeline_completion_status = renderer_result.completion_status
                pipeline_success = renderer_result.completion_status != COMPLETION_STATUS_FAILED
                if save_render_assets:
                    localized_image_path = renderer_result.output_paths.get("localized_image")
                    clean_background_path = renderer_result.output_paths.get("clean_background")

            return PipelineResult(
                source_image_id=loaded.source_image_id,
                source_image=loaded.source_image,
                source_language=resolved_source_language,
                source_language_mode=source_language_mode,
                target_language=target_language,
                processed_at=datetime.now(timezone.utc),
                text_blocks=text_blocks,
                unmatched_vision_blocks=fusion_result.unmatched_vision_blocks,
                stats=_build_stats(
                    text_blocks=text_blocks,
                    ocr_block_count=len(ocr_blocks),
                    vision_block_count=len(vision_blocks),
                ),
                pipeline_success=pipeline_success,
                pipeline_completion_status=pipeline_completion_status,
                renderer=_renderer_diagnostics_from_result(renderer_result) if renderer_result else None,
                localized_image_path=localized_image_path,
                clean_background_path=clean_background_path,
                localized_image=localized_image,
                clean_background=clean_background,
            )
        except Exception as exc:
            return PipelineResult(
                source_image_id=loaded.source_image_id,
                source_image=loaded.source_image,
                source_language=source_language,
                source_language_mode=source_language_mode,
                target_language=target_language,
                processed_at=datetime.now(timezone.utc),
                text_blocks=[],
                unmatched_vision_blocks=[],
                stats=Stats(
                    total_blocks=0,
                    blocks_requiring_review=0,
                    avg_confidence=0.0,
                    ocr_blocks=0,
                    vision_blocks=0,
                ),
                pipeline_success=False,
                pipeline_completion_status="failed",
                renderer=RendererDiagnostics(success=False, completion_status="failed", fatal_error=str(exc)),
            )


def run_pipeline(
    image_path: str | Path,
    *,
    ocr_provider: OCRProvider,
    vision_provider: VisionProvider,
    translation_provider: OpenAITranslationProvider,
    fusion_settings: FusionSettings,
    output_dir: Path,
    target_language: str = "EN",
    source_language: str | None = None,
    source_language_mode: str = "auto",
    is_test_case: bool = False,
    render_localized: bool = True,
    save_render_assets: bool = False,
) -> PipelineResult:
    """Convenience wrapper around PipelineOrchestrator.run()."""
    orchestrator = PipelineOrchestrator(
        ocr_provider=ocr_provider,
        vision_provider=vision_provider,
        translation_provider=translation_provider,
        fusion_settings=fusion_settings,
        output_dir=output_dir,
    )
    return orchestrator.run(
        image_path,
        target_language=target_language,
        source_language=source_language,
        source_language_mode=source_language_mode,
        is_test_case=is_test_case,
        render_localized=render_localized,
        save_render_assets=save_render_assets,
    )


def _renderer_diagnostics_from_result(renderer_result: RendererResult) -> RendererDiagnostics:
    return RendererDiagnostics(
        success=renderer_result.success,
        completion_status=renderer_result.completion_status,
        rendered_block_count=renderer_result.rendered_block_count,
        skipped_block_count=renderer_result.skipped_block_count,
        failed_block_ids=list(renderer_result.failed_block_ids),
        warnings=list(renderer_result.warnings),
        fatal_error=renderer_result.fatal_error,
        output_paths=dict(renderer_result.output_paths),
        block_results=[
            RendererBlockDiagnostics(
                block_id=item.block_id,
                status=item.status,
                restore_success=item.restore_success,
                restore_method=item.restore_method,
                render_success=item.render_success,
                failure_reason=item.failure_reason,
                warnings=list(item.warnings),
            )
            for item in renderer_result.block_results
        ],
    )


def _index_vision_blocks(vision_blocks: list[RawVisionBlock]) -> dict[str, RawVisionBlock]:
    indexed: dict[str, RawVisionBlock] = {}
    for block in vision_blocks:
        if block.ocr_block_id:
            indexed[block.ocr_block_id] = block
    return indexed


def _detect_source_language(fused_blocks: list[FusedBlock]) -> str | None:
    languages = [block.source_language for block in fused_blocks if block.source_language]
    if not languages:
        return None
    return max(set(languages), key=languages.count)


def _finalize_review(review: ReviewInfo) -> ReviewInfo:
    if review.requires_review:
        return review
    return review.model_copy(update={"user_confirmed": True, "user_action": "approved"})


def _build_text_block(
    *,
    fused: FusedBlock,
    visual: VisualStyle,
    render_hints: RenderHints,
    translation: TranslationResultDetail | None,
    skip_translation: bool = False,
    background: BackgroundInfo | None = None,
    render_geometry: RenderGeometry | None = None,
) -> TextBlock:
    return TextBlock(
        id=fused.id,
        original_text=fused.text,
        translated_text=None
        if skip_translation
        else translation.translated_text
        if translation
        else None,
        source_language=fused.source_language or (translation.detected_language if translation else None),
        confidence=fused.confidence,
        bbox=fused.bbox,
        polygon=fused.polygon,
        ocr=fused.ocr,
        vision=fused.vision,
        visual=visual,
        render_hints=render_hints,
        review=_finalize_review(fused.review),
        background=background,
        render_geometry=render_geometry,
    )


def _build_stats(
    *,
    text_blocks: list[TextBlock],
    ocr_block_count: int,
    vision_block_count: int,
) -> Stats:
    blocks_requiring_review = sum(1 for block in text_blocks if block.review.requires_review)
    avg_confidence = (
        sum(block.confidence for block in text_blocks) / len(text_blocks) if text_blocks else 0.0
    )
    return Stats(
        total_blocks=len(text_blocks),
        blocks_requiring_review=blocks_requiring_review,
        avg_confidence=avg_confidence,
        ocr_blocks=ocr_block_count,
        vision_blocks=vision_block_count,
    )
