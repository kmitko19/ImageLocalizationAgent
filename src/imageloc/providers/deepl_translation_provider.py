"""Stub translation provider for DeepL.

Not wired into the MVP orchestrator and does not require a DeepL API key.
It exists only so `TranslationProvider` has a second implementation on
record, proving the interface isn't tailored to OpenAI alone. Implement this
in a later stage if literal, non-context-aware translation is ever needed.
"""

from __future__ import annotations

from imageloc.providers.base import TranslatedBlock, TranslationInput


class DeepLTranslationProvider:
    """Not implemented in the MVP. Do not instantiate or wire this up."""

    def __init__(self, api_key: str | None = None) -> None:
        raise NotImplementedError(
            "DeepLTranslationProvider is a placeholder for a future stage. "
            "The MVP uses OpenAITranslationProvider exclusively."
        )

    def translate(
        self,
        blocks: list[TranslationInput],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslatedBlock]:
        raise NotImplementedError
