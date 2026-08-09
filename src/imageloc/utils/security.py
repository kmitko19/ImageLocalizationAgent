"""Security helpers — предотвращение утечки OPENAI_API_KEY в логи и ошибки.

API-ключ хранится только в .env и никогда не должен попадать:
- в исходный код;
- JSON-результат;
- логи;
- Git.
"""

from __future__ import annotations

import os
import re


def get_api_key_pattern() -> re.Pattern[str]:
    """Return a regex that matches typical OpenAI API key prefixes."""
    return re.compile(r"sk-[A-Za-z0-9_-]+")


def sanitize_message(message: str) -> str:
    """Strip any API key-like substring from an error or log message."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    sanitized = message

    if api_key:
        sanitized = sanitized.replace(api_key, "[REDACTED]")

    return get_api_key_pattern().sub("[REDACTED]", sanitized)
