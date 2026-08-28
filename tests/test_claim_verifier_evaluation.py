import json
from pathlib import Path

import pytest

from app.evaluation.claim_verifier import (
    load_claim_verifier_suite,
    run_claim_verifier_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "evaluation" / "suites" / "v0_5" / "claim_verifier_development.json"


class ExpectedGenerator:
    def __init__(self, suite) -> None:
        self.suite = suite
        self.batch_calls = 4

    def __call__(self, prompts: list[str]) -> list[str]:
        assert all("verifier-approved evidence" in prompt for prompt in prompts)
        return [case.expected.model_dump_json() for case in self.suite.cases]


def test_development_suite_loads_all_controlled_challenges() -> None:
    suite = load_claim_verifier_suite(SUITE)
    assert len(suite.cases) == 7
    assert suite.benchmark_status == "development"
    assert {case.case_id for case in suite.cases} >= {
        "supported_numeric", "partial_numeric", "wrong_citation",
        "missing_citation", "compound_claims", "mixed_citations",
    }


def test_perfect_predictions_score_all_structural_metrics() -> None:
    suite = load_claim_verifier_suite(SUITE)
    report = run_claim_verifier_benchmark(
        suite, ExpectedGenerator(suite), model="test", model_revision="pinned"
    )
    assert report.schema_valid_rate == 1.0
    assert report.exact_case_rate == 1.0
    assert report.extraction_exact_case_rate == 1.0
    assert report.claim_verdict_accuracy == 1.0
    assert report.evidence_relationship_accuracy == 1.0
    assert report.generation_batches == 4
    assert report.citation_metrics["citation_precision"] == pytest.approx(4 / 7)


def test_invalid_predictions_fail_closed_and_count_against_metrics() -> None:
    suite = load_claim_verifier_suite(SUITE).model_copy(
        update={"cases": load_claim_verifier_suite(SUITE).cases[:1]}
    )
    report = run_claim_verifier_benchmark(
        suite, lambda prompts: [json.dumps({"answer": "changed"})], model="test"
    )
    assert report.schema_valid_rate == 0.0
    assert report.exact_case_rate == 0.0
    assert report.claim_verdict_accuracy == 0.0
    assert report.evidence_relationship_accuracy == 0.0
    assert report.results[0].parse_error
    assert report.results[0].raw_response
    assert report.citation_eligible_cases == 1
    assert report.citation_metrics["citation_completeness"] == 0.0


def test_generator_response_count_is_checked() -> None:
    suite = load_claim_verifier_suite(SUITE)
    with pytest.raises(ValueError, match="wrong response count"):
        run_claim_verifier_benchmark(suite, lambda prompts: [], model="test")
