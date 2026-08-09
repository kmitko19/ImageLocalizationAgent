"""Application configuration loader.

Loads the OpenAI API key from .env (never hardcoded, never logged) and
non-secret tunables (model names, thresholds, paths) from config/settings.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass
class OpenAISettings:
    vision_model: str
    translation_model: str
    api_key: str


@dataclass
class FusionSettings:
    iou_threshold: float
    text_similarity_threshold: float
    text_mismatch_threshold: float
    low_ocr_confidence_threshold: float
    low_font_size_confidence_threshold: float


@dataclass
class PathsSettings:
    test_cases_dir: Path
    output_dir: Path


@dataclass
class Settings:
    openai: OpenAISettings
    ocr_languages: list[str]
    fusion: FusionSettings
    paths: PathsSettings
    gui_port: int
    source_languages: list[str] = field(default_factory=list)
    target_languages: list[str] = field(default_factory=list)


def load_settings() -> Settings:
    """Load settings.yaml + .env into a single Settings object.

    Raises RuntimeError with a clear message if OPENAI_API_KEY is missing,
    since OpenAI is a required component of the MVP.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    raw: dict[str, Any] = {}
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-your-key"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    openai_raw = raw.get("openai", {})
    fusion_raw = raw.get("fusion", {})
    paths_raw = raw.get("paths", {})
    gui_raw = raw.get("gui", {})
    languages_raw = raw.get("languages", {})

    return Settings(
        openai=OpenAISettings(
            vision_model=openai_raw.get("vision_model", "gpt-4o"),
            translation_model=openai_raw.get("translation_model", "gpt-4o-mini"),
            api_key=api_key,
        ),
        ocr_languages=raw.get("ocr", {}).get("languages", ["ru", "en"]),
        fusion=FusionSettings(
            iou_threshold=fusion_raw.get("iou_threshold", 0.25),
            text_similarity_threshold=fusion_raw.get("text_similarity_threshold", 0.55),
            text_mismatch_threshold=fusion_raw.get("text_mismatch_threshold", 0.5),
            low_ocr_confidence_threshold=fusion_raw.get("low_ocr_confidence_threshold", 0.5),
            low_font_size_confidence_threshold=fusion_raw.get(
                "low_font_size_confidence_threshold", 0.4
            ),
        ),
        paths=PathsSettings(
            test_cases_dir=PROJECT_ROOT / paths_raw.get("test_cases_dir", "data/input/test_cases"),
            output_dir=PROJECT_ROOT / paths_raw.get("output_dir", "data/output"),
        ),
        gui_port=gui_raw.get("server_port", 7860),
        source_languages=languages_raw.get("source", ["auto", "ru", "en"]),
        target_languages=languages_raw.get("target", ["en", "ru"]),
    )
