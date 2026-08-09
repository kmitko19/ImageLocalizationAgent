"""Offline integration tests for fixed test-case assets and runner."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from imageloc.config import PROJECT_ROOT
from imageloc.utils.test_cases import (
    CaseStatus,
    discover_test_cases,
    evaluate_offline_case,
    load_acceptance,
    load_baseline,
    resolve_image_status,
)

RUNNER = PROJECT_ROOT / "scripts" / "run_test_cases.py"
EXPECTED_CASE_IDS = {
    "01_complex_brochure_ru",
    "02_text_and_nontext_ru",
    "03_text_on_photo_ru",
    "04_decorative_headline_ru",
    "05_catalog_mixed_layout_ru",
}


@pytest.fixture(scope="module")
def specs():
    return discover_test_cases()


@pytest.fixture(scope="module")
def specs_by_id(specs):
    return {spec.case_id: spec for spec in specs}


def test_discover_all_five_test_cases(specs):
    assert len(specs) == 5
    assert {spec.case_id for spec in specs} == EXPECTED_CASE_IDS


def test_available_cases_01_and_02(specs_by_id):
    for case_id in ("01_complex_brochure_ru", "02_text_and_nontext_ru"):
        spec = specs_by_id[case_id]
        assert resolve_image_status(spec) == "available"
        assert spec.acceptance["image_status"] == "available"
        assert spec.image_path is not None
        assert spec.image_path.exists()


def test_missing_cases_03_to_05(specs_by_id):
    for case_id in (
        "03_text_on_photo_ru",
        "04_decorative_headline_ru",
        "05_catalog_mixed_layout_ru",
    ):
        spec = specs_by_id[case_id]
        assert resolve_image_status(spec) == "missing"
        assert spec.acceptance["image_status"] == "missing"
        assert spec.missing_marker_path is not None
        assert spec.missing_marker_path.exists()
        assert spec.image_path is not None
        assert not spec.image_path.exists()


def test_baseline_present_for_available_cases(specs_by_id):
    for case_id in ("01_complex_brochure_ru", "02_text_and_nontext_ru"):
        spec = specs_by_id[case_id]
        assert spec.baseline_path is not None
        assert spec.baseline_path.exists()
        assert spec.baseline is not None
        assert spec.acceptance.get("baseline_file")


def test_baseline_absent_for_missing_cases(specs_by_id):
    for case_id in (
        "03_text_on_photo_ru",
        "04_decorative_headline_ru",
        "05_catalog_mixed_layout_ru",
    ):
        spec = specs_by_id[case_id]
        assert spec.baseline_path is None
        assert spec.baseline is None
        assert spec.acceptance.get("baseline_file") is None


def test_all_acceptance_json_valid(specs):
    for spec in specs:
        data = load_acceptance(spec.acceptance_path)
        assert data["case_id"] == spec.case_id


def test_baseline_json_valid_for_available_cases(specs_by_id):
    for case_id in ("01_complex_brochure_ru", "02_text_and_nontext_ru"):
        spec = specs_by_id[case_id]
        data = load_baseline(spec.baseline_path)
        assert data["case_id"] == case_id


def test_offline_evaluator_passes_case_01(specs_by_id):
    result = evaluate_offline_case(specs_by_id["01_complex_brochure_ru"])
    assert result.status == CaseStatus.PASS


def test_offline_evaluator_passes_case_02(specs_by_id):
    result = evaluate_offline_case(specs_by_id["02_text_and_nontext_ru"])
    assert result.status == CaseStatus.PASS


def test_missing_cases_skip_not_fail(specs_by_id):
    for case_id in (
        "03_text_on_photo_ru",
        "04_decorative_headline_ru",
        "05_catalog_mixed_layout_ru",
    ):
        result = evaluate_offline_case(specs_by_id[case_id])
        assert result.status == CaseStatus.SKIP
        assert result.status != CaseStatus.FAIL


def test_offline_runner_does_not_require_api_key():
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "Failed: 0" in completed.stdout


def test_offline_runner_summary_table(specs_by_id):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout
    assert completed.returncode == 0
    assert "01_complex_brochure_ru" in stdout
    assert "03_text_on_photo_ru" in stdout
    assert "| PASS" in stdout or "| PASS\n" in stdout
    assert "| SKIP" in stdout
