import json
from pathlib import Path

import pytest

from app.evaluation.citations import (
    CitationSafetySuite,
    evaluate_citation_safety,
    load_citation_safety_suite,
    write_citation_report,
)

FIXTURE = Path("evaluation/suites/v0_5/citation_safety_fixtures.json")


def test_fixture_covers_the_four_citation_failure_dimensions() -> None:
    suite = load_citation_safety_suite(FIXTURE)
    report = evaluate_citation_safety(suite)

    assert report.case_count == 5
    assert report.benchmark_status == "fixture"
    assert report.metrics["citation_precision"] == pytest.approx(1 / 3)
    assert report.metrics["citation_completeness"] == 0.5
    assert report.metrics["unsupported_claim_rate"] == 0.75
    assert report.metrics["invalid_citation_rate"] == pytest.approx(1 / 3)
    assert report.metrics["citation_required_claims"] == 4


def test_case_metrics_use_not_applicable_for_empty_denominators() -> None:
    report = evaluate_citation_safety(load_citation_safety_suite(FIXTURE))
    results = {case.case_id: case.metrics for case in report.results}

    assert results["missing_citation"]["citation_precision"] is None
    assert results["citation_not_required"]["citation_completeness"] is None
    assert results["citation_not_required"]["unsupported_claim_rate"] is None
    assert results["citation_not_required"]["invalid_citation_rate"] is None


def test_report_writer_preserves_metrics_and_human_readable_output(tmp_path: Path) -> None:
    report = evaluate_citation_safety(load_citation_safety_suite(FIXTURE))

    write_citation_report(report, tmp_path)

    payload = json.loads((tmp_path / "citation_report.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "citation_report.md").read_text(encoding="utf-8")
    assert payload["metrics"]["unsupported_claim_rate"] == 0.75
    assert "Citation precision" in markdown
    assert "not applicable" not in markdown
    assert "| missing_citation | n/a |" in markdown


def test_unknown_gold_support_is_rejected_but_invalid_prediction_is_measured() -> None:
    valid_invalid_prediction = {
        "contract_version": "1.0.0",
        "suite_id": "test-suite",
        "dataset_version": "0.0.1",
        "benchmark_status": "fixture",
        "description": "test",
        "cases": [
            {
                "case_id": "case",
                "available_evidence_ids": ["e1"],
                "claims": [
                    {
                        "claim_id": "claim",
                        "text": "claim",
                        "requires_citation": True,
                        "cited_evidence_ids": ["invalid"],
                        "supporting_evidence_ids": ["e1"],
                    }
                ],
            }
        ],
    }
    suite = CitationSafetySuite.model_validate(valid_invalid_prediction)
    assert evaluate_citation_safety(suite).metrics["invalid_citation_rate"] == 1.0

    valid_invalid_prediction["cases"][0]["claims"][0]["supporting_evidence_ids"] = [
        "unknown"
    ]
    with pytest.raises(ValueError, match="unknown supporting evidence"):
        CitationSafetySuite.model_validate(valid_invalid_prediction)


def test_duplicate_citation_assignment_is_rejected() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["cases"][0]["claims"][0]["cited_evidence_ids"] = ["e1", "e1"]

    with pytest.raises(ValueError, match="must be unique"):
        CitationSafetySuite.model_validate(payload)
