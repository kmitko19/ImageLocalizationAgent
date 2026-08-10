"""One-off helper to generate baseline/acceptance JSON from E2E results."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TC = ROOT / "data/input/test_cases"
ACCEPT = TC / "acceptance"
BASE = TC / "baselines"

CASES = [
    {
        "case_id": "01_complex_brochure_ru",
        "image_file": "01_complex_brochure_ru.jpg",
        "e2e": ROOT / "data/output/edb122a8-c010-42f5-a1e1-9a556e72dfbf/result.json",
        "required_texts": ["Беречь", "иербне"],
        "required_blocks": [
            {
                "block_id": "block_003",
                "review_reasons_contains_any": ["low_ocr_confidence"],
            },
            {
                "block_id": "block_015",
                "review_reasons_contains_any": ["text_mismatch"],
            },
        ],
        "review_reason_counts": {
            "vision_scope_fallback": {"gte": 20},
            "text_mismatch": {"gte": 15},
            "low_ocr_confidence": {"gte": 30},
        },
    },
    {
        "case_id": "02_text_and_nontext_ru",
        "image_file": "02_text_and_nontext_ru.jpg",
        "e2e": ROOT / "data/output/5884debc-ead0-49d0-ad8d-4e3c7b2d530a/result.json",
        "required_texts": ["Ваши", "за"],
        "required_blocks": [
            {
                "block_id": "block_018",
                "review_reasons_contains_any": ["low_ocr_confidence"],
            },
            {
                "block_id": "block_027",
                "review_reasons_contains_any": ["probable_non_text"],
            },
        ],
        "review_reason_counts": {
            "low_ocr_confidence": {"gte": 2},
            "probable_non_text": {"eq": 1},
        },
    },
]

MISSING = [
    ("03_text_on_photo_ru", "03_text_on_photo_ru.jpg", "Text over heterogeneous photo background"),
    ("04_decorative_headline_ru", "04_decorative_headline_ru.jpg", "Large decorative/stylized headline"),
    ("05_catalog_mixed_layout_ru", "05_catalog_mixed_layout_ru.jpg", "Mixed catalog layout"),
]


def summarize(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    reasons: Counter[str] = Counter()
    translated = 0
    for block in result["text_blocks"]:
        if block.get("translated_text"):
            translated += 1
        for reason in block.get("review", {}).get("reasons", []):
            reasons[reason] += 1
    stats = result["stats"]
    return {
        "stats": stats,
        "unmatched_vision_blocks_count": len(result.get("unmatched_vision_blocks", [])),
        "translated_blocks_count": translated,
        "review_reason_counts": dict(reasons),
        "source_image_dimensions": {
            "width": result["source_image"]["width"],
            "height": result["source_image"]["height"],
        },
        "target_language": result.get("target_language", "EN"),
        "source_language_mode": result.get("source_language_mode", "auto"),
    }


def main() -> None:
    ACCEPT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        metrics = summarize(case["e2e"])
        stats = metrics["stats"]
        expectations = {
            "pipeline_completes": True,
            "stats": {
                "total_blocks": {"eq": stats["total_blocks"]},
                "ocr_blocks": {"eq": stats["ocr_blocks"]},
                "vision_blocks": {"eq": stats["vision_blocks"]},
                "blocks_requiring_review": {"eq": stats["blocks_requiring_review"]},
            },
            "unmatched_vision_blocks_count": {"eq": metrics["unmatched_vision_blocks_count"]},
            "translated_blocks_min": metrics["translated_blocks_count"],
            "review_reason_counts": case["review_reason_counts"],
            "required_texts": case["required_texts"],
            "required_blocks": case["required_blocks"],
        }

        baseline = {
            "case_id": case["case_id"],
            "image_file": case["image_file"],
            "source_e2e_result": case["e2e"].relative_to(ROOT).as_posix(),
            "extracted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_image_dimensions": metrics["source_image_dimensions"],
            "stats": stats,
            "unmatched_vision_blocks_count": metrics["unmatched_vision_blocks_count"],
            "translated_blocks_count": metrics["translated_blocks_count"],
            "review_reason_counts": metrics["review_reason_counts"],
            "required_texts": case["required_texts"],
            "required_blocks": case["required_blocks"],
        }
        baseline_path = BASE / f"{case['case_id']}.baseline.json"
        baseline_path.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        acceptance = {
            "case_id": case["case_id"],
            "image_file": case["image_file"],
            "image_status": "available",
            "baseline_file": f"baselines/{case['case_id']}.baseline.json",
            "target_language": metrics["target_language"],
            "source_language_mode": metrics["source_language_mode"],
            "automated_run": True,
            "expectations": expectations,
        }
        acc_path = ACCEPT / f"{case['case_id']}.acceptance.json"
        acc_path.write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for case_id, image_file, description in MISSING:
        missing_marker = TC / f"{image_file}.MISSING"
        missing_marker.write_text(
            f"Image missing for test case {case_id}.\n\n"
            f"Purpose: {description}\n\n"
            f"Add {image_file} manually when available.\n",
            encoding="utf-8",
        )
        acceptance = {
            "case_id": case_id,
            "image_file": image_file,
            "image_status": "missing",
            "baseline_file": None,
            "target_language": "EN",
            "source_language_mode": "auto",
            "automated_run": False,
            "expectations": {
                "pipeline_completes": True,
                "notes": description,
            },
        }
        acc_path = ACCEPT / f"{case_id}.acceptance.json"
        acc_path.write_text(
            json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
