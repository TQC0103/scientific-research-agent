import json
from pathlib import Path

import pytest

from app.evaluation.external import ExternalDatasetError, load_qasper, load_scifact
from app.evaluation.metrics import (
    binary_sufficiency_metrics,
    classification_metrics,
    qasper_metrics,
)


def test_qasper_adapter_preserves_multiple_references_and_unanswerable(tmp_path: Path) -> None:
    payload = {
        "paper-a": {
            "title": "Paper A",
            "qas": [
                {
                    "question_id": "q1",
                    "question": "What is reported?",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["Result A"],
                                "free_form_answer": "",
                                "yes_no": None,
                                "evidence": ["Evidence paragraph", "FLOAT SELECTED: figure 1"],
                            }
                        },
                        {
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": [],
                                "free_form_answer": "The paper reports result A.",
                                "yes_no": None,
                                "evidence": ["Evidence paragraph"],
                            }
                        },
                    ],
                },
                {
                    "question_id": "q2",
                    "question": "What is missing?",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": True,
                                "extractive_spans": [],
                                "free_form_answer": "",
                                "yes_no": None,
                                "evidence": [],
                            }
                        }
                    ],
                },
            ],
        }
    }
    source = tmp_path / "qasper.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    cases = load_qasper(source, source_split="dev")
    assert len(cases) == 2
    assert [reference.answer_type for reference in cases[0].references] == [
        "extractive",
        "abstractive",
    ]
    assert cases[0].references[0].evidence == ["Evidence paragraph"]
    assert cases[1].references[0].answer == "Unanswerable"


def test_qasper_metrics_take_best_reference() -> None:
    from app.evaluation.external import QasperCase, QasperReference

    case = QasperCase(
        paper_id="paper-a",
        title="Paper A",
        question_id="q1",
        question="Question",
        source_split="dev",
        references=[
            QasperReference(answer="Result A", answer_type="extractive", evidence=["p1"]),
            QasperReference(
                answer="A longer paraphrase", answer_type="abstractive", evidence=["p2"]
            ),
        ],
    )
    result = qasper_metrics([case], {"q1": {"answer": "Result A", "evidence": ["p1"]}})
    assert result == {
        "answer_f1": 1.0,
        "evidence_f1": 1.0,
        "missing_predictions": 0,
        "eligible_cases": 1,
    }


def test_scifact_adapter_resolves_gold_rationale_sentences(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "doc_id": 10,
                "title": "Document",
                "abstract": ["Background.", "Direct supporting sentence."],
                "structured": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    claims = tmp_path / "claims_dev.jsonl"
    claims.write_text(
        json.dumps(
            {
                "id": 7,
                "claim": "The document supports this claim.",
                "evidence": {"10": [{"sentences": [1], "label": "SUPPORT"}]},
                "cited_doc_ids": [10],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    case = load_scifact(corpus, claims, source_split="dev")[0]
    assert case.label == "SUPPORT"
    assert case.documents[0].doc_id == 10
    assert case.documents[0].abstract[1] == "Direct supporting sentence."
    assert case.rationales[0].sentences == ["Direct supporting sentence."]


def test_scifact_adapter_rejects_missing_evidence_document(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("", encoding="utf-8")
    claims = tmp_path / "claims_dev.jsonl"
    claims.write_text(
        json.dumps(
            {
                "id": 7,
                "claim": "Claim",
                "evidence": {"10": [{"sentences": [0], "label": "SUPPORT"}]},
                "cited_doc_ids": [10],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExternalDatasetError, match="document 10 is missing"):
        load_scifact(corpus, claims, source_split="dev")


def test_classification_and_sufficiency_metrics_report_denominators() -> None:
    classification = classification_metrics(
        ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"],
        ["SUPPORT", "SUPPORT", "NOT_ENOUGH_INFO"],
    )
    assert classification["accuracy"] == pytest.approx(2 / 3)
    assert classification["eligible_cases"] == 3

    sufficiency = binary_sufficiency_metrics(
        [True, False, False, True], [True, True, False, False]
    )
    assert sufficiency["accuracy"] == 0.5
    assert sufficiency["false_positive_rate"] == 0.5
    assert sufficiency["false_negative_rate"] == 0.5
