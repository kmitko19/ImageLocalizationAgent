"""Fixed test-case manifest, baseline metrics, and acceptance evaluation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from imageloc.config import PROJECT_ROOT
from imageloc.models.result import PipelineResult
from imageloc.utils.json_export import load_result

ACCEPTANCE_SUFFIX = ".acceptance.json"
BASELINE_SUFFIX = ".baseline.json"
MISSING_SUFFIX = ".MISSING"


class CaseStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class TestCaseSpec:
    case_id: str
    acceptance_path: Path
    acceptance: dict[str, Any]
    image_path: Path | None
    missing_marker_path: Path | None
    baseline_path: Path | None
    baseline: dict[str, Any] | None


@dataclass
class EvaluationResult:
    case_id: str
    status: CaseStatus
    message: str
    failures: list[str] = field(default_factory=list)


def test_cases_dir(settings_paths_test_cases_dir: Path | None = None) -> Path:
    if settings_paths_test_cases_dir is not None:
        return settings_paths_test_cases_dir
    return PROJECT_ROOT / "data" / "input" / "test_cases"


def acceptance_dir(test_cases_root: Path | None = None) -> Path:
    return (test_cases_root or test_cases_dir()) / "acceptance"


def baselines_dir(test_cases_root: Path | None = None) -> Path:
    return (test_cases_root or test_cases_dir()) / "baselines"


def discover_test_cases(test_cases_root: Path | None = None) -> list[TestCaseSpec]:
    """Load all acceptance manifests sorted by case id."""
    root = test_cases_root or test_cases_dir()
    specs: list[TestCaseSpec] = []
    for acceptance_path in sorted(acceptance_dir(root).glob(f"*{ACCEPTANCE_SUFFIX}")):
        acceptance = load_acceptance(acceptance_path)
        case_id = acceptance["case_id"]
        image_file = acceptance.get("image_file")
        image_path = root / image_file if image_file else None
        missing_marker_path = root / f"{image_file}{MISSING_SUFFIX}" if image_file else None

        baseline_path: Path | None = None
        baseline: dict[str, Any] | None = None
        baseline_rel = acceptance.get("baseline_file")
        if baseline_rel:
            baseline_path = root / baseline_rel
            baseline = load_baseline(baseline_path)

        specs.append(
            TestCaseSpec(
                case_id=case_id,
                acceptance_path=acceptance_path,
                acceptance=acceptance,
                image_path=image_path,
                missing_marker_path=missing_marker_path,
                baseline_path=baseline_path,
                baseline=baseline,
            )
        )
    return specs


def load_acceptance(path: str | Path) -> dict[str, Any]:
    acceptance_path = Path(path)
    data = json.loads(acceptance_path.read_text(encoding="utf-8"))
    validate_acceptance(data, acceptance_path)
    return data


def load_baseline(path: str | Path) -> dict[str, Any]:
    baseline_path = Path(path)
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    validate_baseline(data, baseline_path)
    return data


def validate_acceptance(data: dict[str, Any], path: Path | None = None) -> None:
    label = str(path) if path else "acceptance"
    required = ("case_id", "image_file", "image_status", "expectations")
    for key in required:
        if key not in data:
            raise ValueError(f"{label}: missing required field '{key}'")
    if data["image_status"] not in {"available", "missing"}:
        raise ValueError(f"{label}: image_status must be 'available' or 'missing'")
    if data["image_status"] == "available" and not data.get("baseline_file"):
        raise ValueError(f"{label}: available cases must reference baseline_file")


def validate_baseline(data: dict[str, Any], path: Path | None = None) -> None:
    label = str(path) if path else "baseline"
    required = (
        "case_id",
        "image_file",
        "source_e2e_result",
        "stats",
        "unmatched_vision_blocks_count",
        "translated_blocks_count",
        "review_reason_counts",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"{label}: missing required field '{key}'")


def resolve_image_status(spec: TestCaseSpec) -> str:
    declared = spec.acceptance.get("image_status", "missing")
    if declared == "missing":
        return "missing"
    if spec.image_path and spec.image_path.exists():
        return "available"
    if spec.missing_marker_path and spec.missing_marker_path.exists():
        return "missing"
    return "missing"


def resolve_reference_result_path(
    spec: TestCaseSpec,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path | None:
    if spec.baseline is None:
        return None
    rel = spec.baseline.get("source_e2e_result")
    if not rel:
        return None
    return project_root / rel


def extract_metrics(result: PipelineResult) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    translated_blocks = 0
    for block in result.text_blocks:
        if block.translated_text:
            translated_blocks += 1
        for reason in block.review.reasons:
            reason_counts[reason] += 1

    return {
        "stats": result.stats.model_dump(),
        "unmatched_vision_blocks_count": len(result.unmatched_vision_blocks),
        "translated_blocks_count": translated_blocks,
        "review_reason_counts": dict(reason_counts),
        "texts": [block.original_text for block in result.text_blocks],
        "blocks_by_id": {block.id: block for block in result.text_blocks},
    }


def _check_constraint(actual: int | float, constraint: dict[str, Any], label: str) -> list[str]:
    failures: list[str] = []
    if "eq" in constraint and actual != constraint["eq"]:
        failures.append(f"{label}: expected eq {constraint['eq']}, got {actual}")
    if "gte" in constraint and actual < constraint["gte"]:
        failures.append(f"{label}: expected gte {constraint['gte']}, got {actual}")
    if "lte" in constraint and actual > constraint["lte"]:
        failures.append(f"{label}: expected lte {constraint['lte']}, got {actual}")
    return failures


def evaluate_metrics(
    metrics: dict[str, Any],
    expectations: dict[str, Any],
    *,
    case_id: str,
) -> EvaluationResult:
    failures: list[str] = []

    stats_expectations = expectations.get("stats", {})
    for stat_name, constraint in stats_expectations.items():
        actual = metrics["stats"].get(stat_name)
        if actual is None:
            failures.append(f"stats.{stat_name}: missing in result")
            continue
        failures.extend(_check_constraint(actual, constraint, f"stats.{stat_name}"))

    unmatched_constraint = expectations.get("unmatched_vision_blocks_count")
    if unmatched_constraint:
        failures.extend(
            _check_constraint(
                metrics["unmatched_vision_blocks_count"],
                unmatched_constraint,
                "unmatched_vision_blocks_count",
            )
        )

    translated_min = expectations.get("translated_blocks_min")
    if translated_min is not None:
        actual = metrics["translated_blocks_count"]
        if actual < translated_min:
            failures.append(
                f"translated_blocks_min: expected >= {translated_min}, got {actual}"
            )

    reason_expectations = expectations.get("review_reason_counts", {})
    for reason, constraint in reason_expectations.items():
        actual = metrics["review_reason_counts"].get(reason, 0)
        failures.extend(_check_constraint(actual, constraint, f"review_reason_counts.{reason}"))

    required_texts = expectations.get("required_texts", [])
    present_texts = set(metrics.get("texts", []))
    for text in required_texts:
        if text not in present_texts:
            failures.append(f"required_texts: missing text {text!r}")

    blocks_by_id = metrics.get("blocks_by_id", {})
    for block_spec in expectations.get("required_blocks", []):
        block_id = block_spec["block_id"]
        block = blocks_by_id.get(block_id)
        if block is None:
            failures.append(f"required_blocks: missing block {block_id}")
            continue
        expected_any = block_spec.get("review_reasons_contains_any") or block_spec.get(
            "review_reasons_contains", []
        )
        if expected_any:
            reasons = set(block.review.reasons)
            if not reasons.intersection(expected_any):
                failures.append(
                    f"required_blocks.{block_id}: expected one of {expected_any}, got {sorted(reasons)}"
                )

    if failures:
        return EvaluationResult(
            case_id=case_id,
            status=CaseStatus.FAIL,
            message=f"{len(failures)} expectation(s) failed",
            failures=failures,
        )
    return EvaluationResult(
        case_id=case_id,
        status=CaseStatus.PASS,
        message="All expectations met",
    )


def evaluate_result(result: PipelineResult, acceptance: dict[str, Any]) -> EvaluationResult:
    expectations = acceptance.get("expectations", {})
    metrics = extract_metrics(result)
    return evaluate_metrics(metrics, expectations, case_id=acceptance["case_id"])


def compare_metrics_to_baseline(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> EvaluationResult:
    """Ensure extracted metrics match stored baseline aggregates."""
    failures: list[str] = []
    case_id = baseline["case_id"]

    for stat_name, expected_value in baseline["stats"].items():
        actual = metrics["stats"].get(stat_name)
        if actual != expected_value:
            failures.append(f"baseline.stats.{stat_name}: expected {expected_value}, got {actual}")

    if metrics["unmatched_vision_blocks_count"] != baseline["unmatched_vision_blocks_count"]:
        failures.append(
            "baseline.unmatched_vision_blocks_count mismatch: "
            f"expected {baseline['unmatched_vision_blocks_count']}, "
            f"got {metrics['unmatched_vision_blocks_count']}"
        )

    if metrics["translated_blocks_count"] != baseline["translated_blocks_count"]:
        failures.append(
            "baseline.translated_blocks_count mismatch: "
            f"expected {baseline['translated_blocks_count']}, "
            f"got {metrics['translated_blocks_count']}"
        )

    for reason, expected_count in baseline["review_reason_counts"].items():
        actual = metrics["review_reason_counts"].get(reason, 0)
        if actual != expected_count:
            failures.append(
                f"baseline.review_reason_counts.{reason}: expected {expected_count}, got {actual}"
            )

    if failures:
        return EvaluationResult(
            case_id=case_id,
            status=CaseStatus.FAIL,
            message="Baseline metrics mismatch",
            failures=failures,
        )
    return EvaluationResult(
        case_id=case_id,
        status=CaseStatus.PASS,
        message="Baseline metrics match reference result",
    )


def evaluate_offline_case(spec: TestCaseSpec, *, project_root: Path = PROJECT_ROOT) -> EvaluationResult:
    image_status = resolve_image_status(spec)
    if image_status == "missing" or not spec.acceptance.get("automated_run", False):
        return EvaluationResult(
            case_id=spec.case_id,
            status=CaseStatus.SKIP,
            message="Image or baseline missing — offline SKIP",
        )

    reference_path = resolve_reference_result_path(spec, project_root=project_root)
    if reference_path is None or not reference_path.exists():
        return EvaluationResult(
            case_id=spec.case_id,
            status=CaseStatus.SKIP,
            message=f"Reference E2E result not found: {reference_path}",
        )

    result = load_result(reference_path)
    baseline_check = compare_metrics_to_baseline(extract_metrics(result), spec.baseline or {})
    if baseline_check.status != CaseStatus.PASS:
        return baseline_check

    return evaluate_result(result, spec.acceptance)
