import json
from pathlib import Path

import pytest

from app.evaluation.loader import load_suite
from app.evaluation.retrieval import (
    RetrievedChunk,
    evaluate_case_retrieval,
    evaluate_retrieval,
    evidence_matches,
    load_retrieval_jsonl,
    quote_token_recall,
)

SUITE_PATH = Path("evaluation/suites/v0_5/development_10.json")


def _chunk(versioned_id: str, text: str, *, page: int = 4) -> dict:
    return {"versioned_id": versioned_id, "text": text, "page": page}


def test_match_requires_revision_and_quote_content_not_page_alone() -> None:
    case = load_suite(SUITE_PATH).cases[0]
    evidence = case.gold_evidence[0]
    exact = RetrievedChunk.model_validate(
        _chunk(evidence.versioned_id, f"Context before. {evidence.quote} Context after.")
    )
    wrong_revision = exact.model_copy(update={"versioned_id": "1706.03762v6"})
    same_page_wrong_text = RetrievedChunk.model_validate(
        _chunk(evidence.versioned_id, "This unrelated passage happens to be on page four.")
    )
    assert evidence_matches(evidence, exact)
    assert not evidence_matches(evidence, wrong_revision)
    assert not evidence_matches(evidence, same_page_wrong_text)


def test_quote_matching_tolerates_small_extraction_differences() -> None:
    quote = "One two three four five six seven eight nine ten."
    extracted = "one, two three four five six seven eight nine"
    assert quote_token_recall(quote, extracted) == 0.9


def test_case_metrics_report_group_recall_precision_mrr_and_coverage() -> None:
    case = load_suite(SUITE_PATH).cases[0]
    evidence = case.gold_evidence[0]
    result = evaluate_case_retrieval(
        case,
        [
            _chunk(evidence.versioned_id, "Unrelated first result."),
            _chunk(evidence.versioned_id, evidence.quote),
            _chunk(evidence.versioned_id, "Another unrelated result."),
        ],
        top_k=3,
    )
    assert result.eligible
    assert result.recall_at_k == 1.0
    assert result.precision_at_k == pytest.approx(1 / 3)
    assert result.reciprocal_rank == 0.5
    assert result.gold_evidence_coverage == 1.0
    assert result.required_paper_coverage == 1.0
    assert result.macro_paper_recall == 1.0
    assert result.diagnostics == []


def test_multi_paper_case_reports_missing_paper_coverage() -> None:
    suite = load_suite(SUITE_PATH)
    case = next(item for item in suite.cases if len(item.papers) == 2)
    transformer_evidence = next(
        item for item in case.gold_evidence if item.versioned_id == "1706.03762v7"
    )
    result = evaluate_case_retrieval(
        case,
        [_chunk(transformer_evidence.versioned_id, transformer_evidence.quote)],
        top_k=1,
    )
    assert result.required_paper_coverage == 0.5
    assert result.macro_paper_recall == 0.5
    assert result.missing_required_paper_ids == ["1810.04805v2"]
    assert "missing_required_paper_coverage" in result.diagnostics


def test_abstention_without_gold_is_not_retrieval_eligible() -> None:
    suite = load_suite(SUITE_PATH)
    case = next(item for item in suite.cases if not item.gold_evidence)
    result = evaluate_case_retrieval(case, [], top_k=5)
    assert not result.eligible
    assert result.recall_at_k is None
    assert result.precision_at_k is None
    assert result.reciprocal_rank is None
    assert result.diagnostics == ["not_applicable_no_gold_evidence"]


def test_aggregate_excludes_no_gold_cases_and_counts_missing_rows() -> None:
    suite = load_suite(SUITE_PATH)
    first = suite.cases[0]
    evidence = first.gold_evidence[0]
    report = evaluate_retrieval(
        suite,
        {first.case_id: [_chunk(evidence.versioned_id, evidence.quote)]},
        config_name="synthetic-test",
        top_k=1,
    )
    expected_eligible = sum(bool(case.gold_evidence) for case in suite.cases)
    assert report.aggregate.eligible_cases == expected_eligible
    assert report.aggregate.ineligible_cases == len(suite.cases) - expected_eligible
    assert report.aggregate.missing_case_predictions == len(suite.cases) - 1
    assert report.aggregate.recall_at_k == pytest.approx(1 / expected_eligible)


def test_jsonl_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    source = tmp_path / "retrieval.jsonl"
    row = {"case_id": "duplicate", "retrieved": []}
    source.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate retrieval case_id"):
        load_retrieval_jsonl(source)


def test_aggregate_rejects_unknown_case_id() -> None:
    suite = load_suite(SUITE_PATH)
    with pytest.raises(ValueError, match="unknown cases"):
        evaluate_retrieval(suite, {"unknown": []}, config_name="bad")
