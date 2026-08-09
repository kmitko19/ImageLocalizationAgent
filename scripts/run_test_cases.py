"""Run fixed test cases offline against baseline/acceptance metadata.

Default mode compares stored E2E reference results — no external API calls.
Use ``--run-api`` explicitly to execute the live pipeline (OpenAI + EasyOCR).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from imageloc.config import load_settings  # noqa: E402
from imageloc.utils.test_cases import (  # noqa: E402
    CaseStatus,
    EvaluationResult,
    TestCaseSpec,
    discover_test_cases,
    evaluate_offline_case,
    evaluate_result,
    resolve_image_status,
)


@dataclass
class CaseRunRow:
    case_id: str
    image_status: str
    baseline_status: str
    result: str
    message: str
    evaluation: EvaluationResult


def _baseline_status(spec: TestCaseSpec) -> str:
    if spec.baseline_path is not None and spec.baseline_path.exists():
        return "available"
    return "missing"


def _display_result(spec: TestCaseSpec, evaluation: EvaluationResult) -> str:
    image_status = resolve_image_status(spec)
    if image_status == "missing":
        return "SKIP"
    if evaluation.status == CaseStatus.SKIP:
        return "SKIP"
    return evaluation.status.value


def run_offline(specs: list[TestCaseSpec]) -> list[CaseRunRow]:
    rows: list[CaseRunRow] = []
    for spec in specs:
        image_status = resolve_image_status(spec)
        baseline_status = _baseline_status(spec)
        evaluation = evaluate_offline_case(spec, project_root=ROOT)
        rows.append(
            CaseRunRow(
                case_id=spec.case_id,
                image_status=image_status,
                baseline_status=baseline_status,
                result=_display_result(spec, evaluation),
                message=evaluation.message,
                evaluation=evaluation,
            )
        )
    return rows


def run_api(specs: list[TestCaseSpec]) -> list[CaseRunRow]:
    from imageloc.app import build_orchestrator

    settings = load_settings()
    orchestrator = build_orchestrator(settings)
    runs_root = settings.paths.output_dir / "test_case_runs"
    rows: list[CaseRunRow] = []

    for spec in specs:
        image_status = resolve_image_status(spec)
        baseline_status = _baseline_status(spec)

        if image_status == "missing" or not spec.acceptance.get("automated_run", False):
            evaluation = EvaluationResult(
                case_id=spec.case_id,
                status=CaseStatus.SKIP,
                message="Image or baseline missing — API SKIP",
            )
            rows.append(
                CaseRunRow(
                    case_id=spec.case_id,
                    image_status=image_status,
                    baseline_status=baseline_status,
                    result="SKIP",
                    message=evaluation.message,
                    evaluation=evaluation,
                )
            )
            continue

        if spec.image_path is None or not spec.image_path.exists():
            evaluation = EvaluationResult(
                case_id=spec.case_id,
                status=CaseStatus.SKIP,
                message=f"Image not found: {spec.image_path}",
            )
            rows.append(
                CaseRunRow(
                    case_id=spec.case_id,
                    image_status=image_status,
                    baseline_status=baseline_status,
                    result="SKIP",
                    message=evaluation.message,
                    evaluation=evaluation,
                )
            )
            continue

        acceptance = spec.acceptance
        target_language = acceptance.get("target_language", "EN")
        source_language_mode = acceptance.get("source_language_mode", "auto")
        source_language = None if source_language_mode == "auto" else source_language_mode.upper()

        pipeline_result = orchestrator.run(
            spec.image_path,
            target_language=target_language,
            source_language=source_language,
            source_language_mode=source_language_mode,
            is_test_case=True,
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = runs_root / spec.case_id / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "result.json"
        output_path.write_text(
            pipeline_result.model_dump_json(indent=2),
            encoding="utf-8",
        )

        evaluation = evaluate_result(pipeline_result, acceptance)
        rows.append(
            CaseRunRow(
                case_id=spec.case_id,
                image_status=image_status,
                baseline_status=baseline_status,
                result=evaluation.status.value,
                message=f"{evaluation.message}; saved to {output_path.relative_to(ROOT).as_posix()}",
                evaluation=evaluation,
            )
        )

    return rows


def print_summary(rows: list[CaseRunRow], *, mode: str) -> None:
    print(f"\nTest cases summary ({mode})")
    print(f"{'case_id':<32} | {'image_status':<12} | {'baseline':<10} | result")
    print("-" * 78)
    for row in rows:
        print(
            f"{row.case_id:<32} | {row.image_status:<12} | "
            f"{row.baseline_status:<10} | {row.result}"
        )

    passed = sum(1 for row in rows if row.result == "PASS")
    skipped = sum(1 for row in rows if row.result == "SKIP")
    failed = sum(1 for row in rows if row.result == "FAIL")
    print(f"\nPassed: {passed}  Skipped: {skipped}  Failed: {failed}")

    for row in rows:
        if row.result == "FAIL":
            print(f"\nFAIL {row.case_id}: {row.message}")
            for failure in row.evaluation.failures:
                print(f"  - {failure}")


def exit_code(rows: list[CaseRunRow]) -> int:
    for row in rows:
        if row.result == "FAIL":
            return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ImageLocalizationAgent test cases.")
    parser.add_argument(
        "--run-api",
        action="store_true",
        help="Execute the live pipeline (OpenAI/EasyOCR). Default is offline evaluation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = discover_test_cases()
    if not specs:
        print("No test cases found.", file=sys.stderr)
        return 1

    mode = "api" if args.run_api else "offline"
    rows = run_api(specs) if args.run_api else run_offline(specs)

    if args.json:
        payload = [
            {
                "case_id": row.case_id,
                "image_status": row.image_status,
                "baseline": row.baseline_status,
                "result": row.result,
                "message": row.message,
            }
            for row in rows
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_summary(rows, mode=mode)

    return exit_code(rows)


if __name__ == "__main__":
    raise SystemExit(main())
