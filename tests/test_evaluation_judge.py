import json
from pathlib import Path

import pytest

from app.evaluation.judge import (
    JUDGE_CONTRACT_VERSION,
    build_judge_prompt,
    build_judge_report,
    parse_judge_response,
)
from app.evaluation.loader import load_suite

DEVELOPMENT_SUITE = Path("evaluation/suites/v0_5/development_10.json")


def _response(case_id: str, *, human_review_required: bool = False) -> str:
    payload = {
        "contract_version": JUDGE_CONTRACT_VERSION,
        "case_id": case_id,
        "verdict": "pass",
        "scores": {
            "question_clarity": 5,
            "evidence_entailment": 4,
            "answer_alignment": 4,
            "citation_specificity": 5,
            "challenge_validity": 4,
        },
        "findings": [],
        "rationale": "The annotation is internally coherent.",
        "human_review_required": human_review_required,
    }
    return f"```json\n{json.dumps(payload)}\n```"


def test_development_suite_has_required_case_mix_and_is_not_publishable() -> None:
    suite = load_suite(DEVELOPMENT_SUITE)
    assert len(suite.cases) == 10
    assert sum(case.expected.decision.value == "answer" for case in suite.cases) == 8
    assert sum(case.expected.decision.value == "abstain" for case in suite.cases) == 2
    assert {case.question_type for case in suite.cases} >= {
        "single_paper_fact",
        "method",
        "result",
        "multi_paper_comparison",
        "evidence_missing",
        "unsupported_question",
    }
    assert all(case.evaluation_split.value == "development" for case in suite.cases)
    assert all(
        case.annotation.reviewer_count == 1 and case.annotation.adjudicated
        for case in suite.cases
    )
    assert any(
        case.challenge and "partial_evidence" in {kind.value for kind in case.challenge.kinds}
        for case in suite.cases
    )
    with pytest.raises(ValueError, match="Only a frozen suite"):
        suite.assert_publishable()


def test_prompt_contains_case_evidence_and_advisory_boundary() -> None:
    case = load_suite(DEVELOPMENT_SUITE).cases[0]
    prompt = build_judge_prompt(case)
    assert case.case_id in prompt
    assert case.gold_evidence[0].quote in prompt
    assert "never counts as independent human adjudication" in prompt


def test_parse_forces_human_review_for_abstention() -> None:
    case = load_suite(DEVELOPMENT_SUITE).cases[7]
    result = parse_judge_response(case, _response(case.case_id))
    assert result.human_review_required is True


def test_parse_rejects_wrong_case_id() -> None:
    case = load_suite(DEVELOPMENT_SUITE).cases[0]
    with pytest.raises(ValueError, match="expected"):
        parse_judge_response(case, _response("wrong_case"))


def test_report_requires_one_result_per_case_and_aggregates_scores() -> None:
    suite = load_suite(DEVELOPMENT_SUITE)
    results = [
        parse_judge_response(case, _response(case.case_id)) for case in suite.cases
    ]
    report = build_judge_report(suite, "fake/model", results)
    assert report.case_count == 10
    assert report.model_calls == 10
    assert report.verdict_counts == {"pass": 10}
    assert report.mean_scores["question_clarity"] == 5.0
    assert report.human_review_required_count == 2
    with pytest.raises(ValueError, match="every suite case"):
        build_judge_report(suite, "fake/model", results[:-1])
