"""Image loading, normalization and source-image bookkeeping.

This module owns the only two things that need to happen before any AI
analysis starts:

1. Load the file into a normalized RGB PIL.Image the rest of the pipeline
   can rely on (no CMYK, no palette, no alpha surprises).
2. Assign a stable `source_image_id` and resolve where the source lives on
   disk, so a future rendering stage can find the original image again
   without repeating OCR/Vision analysis.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from imageloc.config import PROJECT_ROOT
from imageloc.models.result import SourceImage

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class LoadedImage:
    image: Image.Image
    source_image_id: str
    source_image: SourceImage


def _relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        # Path lives outside the project (e.g. a Gradio temp upload) — the
        # caller is responsible for copying it in before calling this.
        return str(path)


def load_image(input_path: str | Path, output_dir: Path, is_test_case: bool) -> LoadedImage:
    """Load an image and resolve its permanent, citable path.

    - `is_test_case=True`: the file already lives under a stable project
      path (e.g. data/input/test_cases/); it is referenced in place.
    - `is_test_case=False`: the file is a transient upload (e.g. a Gradio
      temp path); it is copied into `output_dir/{source_image_id}/` so it
      survives after the temp file is cleaned up.
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    source_image_id = str(uuid.uuid4())

    if is_test_case:
        resolved_path = _relative_to_project(input_path)
    else:
        dest_dir = output_dir / source_image_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"source{suffix}"
        shutil.copy2(input_path, dest_path)
        resolved_path = _relative_to_project(dest_path)

    with Image.open(input_path) as raw:
        raw.load()
        image_format = (raw.format or suffix.lstrip(".").upper())
        image = raw.convert("RGB")

    source_image = SourceImage(
        filename=input_path.name,
        path=resolved_path,
        width=image.width,
        height=image.height,
        format=image_format,
    )

    return LoadedImage(image=image, source_image_id=source_image_id, source_image=source_image)
