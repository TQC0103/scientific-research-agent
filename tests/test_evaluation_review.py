import json
from pathlib import Path

import pytest

from scripts.render_evaluation_review import render

SUITE = Path("evaluation/suites/v0_5/development_10.json")


def _result(case_id: str, verdict: str) -> dict:
    return {
        "case_id": case_id,
        "verdict": verdict,
        "scores": {
            "question_clarity": 5,
            "evidence_entailment": 4,
            "answer_alignment": 4,
            "citation_specificity": 5,
            "challenge_validity": 4,
        },
        "findings": [],
        "rationale": "Review rationale.",
        "human_review_required": verdict != "pass",
    }


def test_render_combines_questions_answers_evidence_and_judge_findings() -> None:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    results = [
        _result(case["case_id"], "needs_revision" if index == 1 else "pass")
        for index, case in enumerate(suite["cases"])
    ]
    report = {
        "judge_model": "fake/model",
        "case_count": len(results),
        "verdict_counts": {"pass": 9, "needs_revision": 1},
        "results": results,
    }

    page = render(suite, report)

    assert suite["cases"][0]["question"] in page
    assert suite["cases"][0]["expected"]["reference_answer"] in page
    assert suite["cases"][0]["gold_evidence"][0]["quote"] in page
    assert page.index(suite["cases"][1]["case_id"]) < page.index(
        suite["cases"][0]["case_id"]
    )


def test_review_cli_rejects_mismatched_case_ids(tmp_path: Path, monkeypatch) -> None:
    from scripts import render_evaluation_review as review

    suite_path = tmp_path / "suite.json"
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "review.html"
    suite_path.write_text('{"cases":[{"case_id":"one"}]}', encoding="utf-8")
    report_path.write_text('{"results":[{"case_id":"two"}]}', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "render_evaluation_review.py",
            "--suite",
            str(suite_path),
            "--report",
            str(report_path),
            "--output",
            str(output_path),
        ],
    )
    with pytest.raises(ValueError, match="case IDs do not match"):
        review.main()
