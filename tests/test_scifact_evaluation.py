import json

import pytest

from app.evaluation.external import SciFactCase, SciFactDocument, SciFactRationale
from app.evaluation.scifact import (
    build_scifact_prompt,
    parse_scifact_response,
    run_scifact_benchmark,
)


def _cases() -> list[SciFactCase]:
    document = SciFactDocument(
        doc_id=10, title="Study", abstract=["Background.", "Treatment reduced fever."]
    )
    return [
        SciFactCase(
            claim_id=1, claim="Treatment reduced fever.", source_split="dev",
            label="SUPPORT", cited_doc_ids=[10], documents=[document],
            rationales=[SciFactRationale(doc_id=10, label="SUPPORT",
                                         sentence_indices=[1],
                                         sentences=["Treatment reduced fever."])]
        ),
        SciFactCase(
            claim_id=2, claim="Treatment increased fever.", source_split="dev",
            label="CONTRADICT", cited_doc_ids=[10], documents=[document],
            rationales=[SciFactRationale(doc_id=10, label="CONTRADICT",
                                         sentence_indices=[1],
                                         sentences=["Treatment reduced fever."])]
        ),
        SciFactCase(
            claim_id=3, claim="Treatment changed blood pressure.", source_split="dev",
            label="NOT_ENOUGH_INFO", cited_doc_ids=[10], documents=[document], rationales=[]
        ),
    ]


class PerfectGenerator:
    batch_calls = 1

    def __call__(self, prompts: list[str]) -> list[str]:
        assert all("NOT_ENOUGH_INFO" in prompt for prompt in prompts)
        return [
            json.dumps({"label": "SUPPORT", "rationales": [
                {"doc_id": 10, "sentence_indices": [1]}], "reason": "Direct support."}),
            json.dumps({"label": "CONTRADICT", "rationales": [
                {"doc_id": 10, "sentence_indices": [1]}], "reason": "Direct conflict."}),
            json.dumps({"label": "NOT_ENOUGH_INFO", "rationales": [],
                        "reason": "No blood pressure evidence."}),
        ]


def test_prompt_uses_native_labels_and_numbered_document_sentences() -> None:
    prompt = build_scifact_prompt(_cases()[0])
    assert "[10:1] Treatment reduced fever." in prompt
    assert "A topical passage" in prompt
    assert "partial" not in prompt.casefold()


def test_perfect_native_predictions_score_label_and_rationale_metrics() -> None:
    report = run_scifact_benchmark(_cases(), PerfectGenerator(), model="test")
    assert report.label_metrics["accuracy"] == 1.0
    assert report.label_metrics["macro_f1"] == 1.0
    assert report.rationale_sentence_f1 == 1.0
    assert report.rationale_exact_match == 1.0
    assert report.joint_label_rationale_exact_match == 1.0
    assert report.nei_spurious_rationale_rate == 0.0
    assert report.parse_failure_count == 0


def test_parse_rejects_unknown_sentence_and_nei_rationale() -> None:
    case = _cases()[0]
    with pytest.raises(ValueError, match="unavailable sentence"):
        parse_scifact_response(
            '{"label":"SUPPORT","rationales":[{"doc_id":10,'
            '"sentence_indices":[9]}],"reason":"x"}', case
        )
    with pytest.raises(ValueError, match="cannot select"):
        parse_scifact_response(
            '{"label":"NOT_ENOUGH_INFO","rationales":[{"doc_id":10,'
            '"sentence_indices":[1]}],"reason":"x"}', case
        )


def test_parser_skips_an_echoed_schema_object_before_prediction() -> None:
    response = (
        '{"type":"object","properties":{"label":{"type":"string"}}}\n'
        '{"label":"SUPPORT","rationales":[{"doc_id":10,'
        '"sentence_indices":[1]}],"reason":"Direct support."}'
    )
    prediction = parse_scifact_response(response, _cases()[0])
    assert prediction.label == "SUPPORT"


def test_parser_accepts_metric_complete_prediction_without_optional_reason() -> None:
    prediction = parse_scifact_response(
        '{"label":"NOT_ENOUGH_INFO","rationales":[]}', _cases()[2]
    )
    assert prediction.reason is None


def test_invalid_response_fails_closed_to_nei_and_is_counted() -> None:
    report = run_scifact_benchmark(
        [_cases()[0]], lambda prompts: ["not json"], model="test"
    )
    assert report.parse_failure_count == 1
    assert report.results[0].predicted_label == "NOT_ENOUGH_INFO"
    assert report.label_metrics["accuracy"] == 0.0
    assert report.support_detection_metrics["false_negative_rate"] == 1.0
