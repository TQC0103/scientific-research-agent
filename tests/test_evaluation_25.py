import json
import re
from pathlib import Path

import pytest

from app.evaluation.loader import load_suite

ROOT = Path(__file__).resolve().parents[1]
R10_PATH = ROOT / "evaluation" / "suites" / "v0_5" / "development_10.json"
R25_PATH = ROOT / "evaluation" / "suites" / "v0_5" / "development_25.json"
R25_SOURCES_PATH = (
    ROOT / "evaluation" / "suites" / "v0_5" / "development_25_sources.json"
)


def test_development_25_preserves_reviewed_r10_slice_exactly() -> None:
    r10 = load_suite(R10_PATH)
    r25 = load_suite(R25_PATH)

    assert len(r10.cases) == 10
    assert len(r25.cases) == 25
    assert [case.model_dump(mode="json") for case in r25.cases[:10]] == [
        case.model_dump(mode="json") for case in r10.cases
    ]


def test_development_25_has_expected_decisions_papers_and_review_state() -> None:
    suite = load_suite(R25_PATH)

    assert sum(case.expected.decision.value == "answer" for case in suite.cases) == 22
    assert sum(case.expected.decision.value == "abstain" for case in suite.cases) == 3
    assert {paper.versioned_id for case in suite.cases for paper in case.papers} == {
        "1512.03385v1",
        "1706.03762v7",
        "1810.04805v2",
        "2005.11401v4",
        "2106.09685v2",
    }
    assert all(case.evaluation_split.value == "development" for case in suite.cases)
    assert all(case.annotation.adjudicated for case in suite.cases)
    assert all((case.annotation.reviewer_count or 0) >= 1 for case in suite.cases)

    with pytest.raises(ValueError, match="Only a frozen suite"):
        suite.assert_publishable()


def test_partial_resource_case_requires_abstention_without_invented_latency_factor() -> None:
    suite = load_suite(R25_PATH)
    case = next(
        case
        for case in suite.cases
        if case.case_id == "lora_all_resource_reduction_factors_missing"
    )

    assert case.expected.decision.value == "abstain"
    assert case.expected.abstention_reason.value == "evidence_missing"
    supported = {support for evidence in case.gold_evidence for support in evidence.supports}
    assert supported == {"trainable_factor_reported", "memory_factor_reported"}
    assert "latency_factor_not_reported" not in supported
    assert case.challenge is not None
    assert {kind.value for kind in case.challenge.kinds} >= {
        "negative",
        "adversarial",
        "partial_evidence",
        "evidence_missing",
    }


def test_development_25_source_manifest_covers_every_pinned_revision() -> None:
    suite = load_suite(R25_PATH)
    manifest = json.loads(R25_SOURCES_PATH.read_text(encoding="utf-8"))
    entries = {entry["versioned_id"]: entry for entry in manifest["sources"]}

    assert manifest["suite_id"] == suite.suite_id
    assert set(entries) == {
        paper.versioned_id for case in suite.cases for paper in case.papers
    }
    for entry in entries.values():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["pdf_sha256"])
        assert entry["page_count"] > 0
        assert entry["abstract_url"].endswith(entry["versioned_id"])
        assert entry["pdf_url"].endswith(entry["versioned_id"])
