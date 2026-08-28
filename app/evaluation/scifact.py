"""Native-label SciFact oracle-document classification and rationale evaluation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from app.evaluation.external import SciFactCase
from app.evaluation.metrics import binary_sufficiency_metrics, classification_metrics
from app.evaluation.models import StrictModel

SCIFACT_RUNNER_CONTRACT_VERSION = "1.0.0"
SciFactLabel = Literal["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
TextGenerator = Callable[[list[str]], list[str]]


class PredictedRationale(StrictModel):
    doc_id: int
    sentence_indices: list[int] = Field(min_length=1)

    @field_validator("sentence_indices")
    @classmethod
    def indices_are_unique(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)) or any(index < 0 for index in value):
            raise ValueError("Rationale sentence indices must be unique and non-negative.")
        return value


class SciFactPrediction(StrictModel):
    label: SciFactLabel
    rationales: list[PredictedRationale]
    reason: str | None = Field(default=None, min_length=1)


class SciFactCaseResult(StrictModel):
    claim_id: int
    claim: str
    gold_label: SciFactLabel
    predicted_label: SciFactLabel
    label_correct: bool
    schema_valid: bool
    rationale_f1: float | None
    rationale_exact: bool | None
    joint_exact: bool | None
    predicted_rationales: list[PredictedRationale]
    parse_error: str | None
    raw_response: str


class SciFactReport(StrictModel):
    contract_version: str = SCIFACT_RUNNER_CONTRACT_VERSION
    source_split: str
    evaluation_mode: Literal["oracle_documents"] = "oracle_documents"
    model: str
    model_revision: str | None
    case_count: int
    model_calls: int
    generation_batches: int
    parse_failure_count: int
    latency_seconds: float
    label_metrics: dict[str, Any]
    support_detection_metrics: dict[str, float | int]
    rationale_eligible_cases: int
    rationale_sentence_f1: float
    rationale_exact_match: float
    joint_label_rationale_exact_match: float
    nei_spurious_rationale_rate: float
    results: list[SciFactCaseResult]


def build_scifact_prompt(case: SciFactCase) -> str:
    documents = []
    for document in case.documents:
        sentences = "\n".join(
            f"[{document.doc_id}:{index}] {sentence}"
            for index, sentence in enumerate(document.abstract)
        )
        documents.append(f"Document {document.doc_id}: {document.title}\n{sentences}")
    return f"""Classify one scientific claim using only the supplied SciFact documents.

Use the native labels exactly:
- SUPPORT: at least one selected sentence set establishes every material part of the claim.
- CONTRADICT: at least one selected sentence set directly refutes a material part.
- NOT_ENOUGH_INFO: the documents neither establish nor directly refute the full claim.

A topical passage, an omitted quantity, or evidence supporting only a strict subset is
NOT_ENOUGH_INFO unless another sentence directly contradicts the missing part. Do not use
outside knowledge. Select only the minimal sentence indices needed for SUPPORT or CONTRADICT.
Return an empty rationales list for NOT_ENOUGH_INFO. Every doc_id and sentence index must come
from the supplied documents. A brief reason is useful but optional. Return exactly one JSON
object and no Markdown or commentary:
{{"label":"SUPPORT|CONTRADICT|NOT_ENOUGH_INFO","rationales":[{{"doc_id":123,
"sentence_indices":[0]}}],"reason":"brief evidence-based reason"}}

Claim:
{case.claim}

Supplied cited documents:
{chr(10).join(documents)}
"""


def _extract_json(content: Any) -> dict[str, Any]:
    text = str(content).strip()
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {"label", "rationales"} <= value.keys():
            return value
    raise ValueError("SciFact classifier did not return a JSON object.")


def parse_scifact_response(content: Any, case: SciFactCase) -> SciFactPrediction:
    prediction = SciFactPrediction.model_validate(_extract_json(content))
    documents = {document.doc_id: len(document.abstract) for document in case.documents}
    seen_documents = set()
    for rationale in prediction.rationales:
        if rationale.doc_id in seen_documents:
            raise ValueError("Each predicted document may have only one rationale entry.")
        seen_documents.add(rationale.doc_id)
        sentence_count = documents.get(rationale.doc_id)
        if sentence_count is None:
            raise ValueError(f"Prediction references unavailable document {rationale.doc_id}.")
        if any(index >= sentence_count for index in rationale.sentence_indices):
            raise ValueError(f"Prediction references an unavailable sentence in {rationale.doc_id}.")
    if prediction.label == "NOT_ENOUGH_INFO" and prediction.rationales:
        raise ValueError("NOT_ENOUGH_INFO predictions cannot select rationale sentences.")
    return prediction


def _pairs(prediction: SciFactPrediction) -> set[tuple[int, int]]:
    return {
        (rationale.doc_id, index)
        for rationale in prediction.rationales
        for index in rationale.sentence_indices
    }


def _rationale_scores(case: SciFactCase, prediction: SciFactPrediction) -> tuple[float, bool]:
    predicted = _pairs(prediction)
    references = [
        {(rationale.doc_id, index) for index in rationale.sentence_indices}
        for rationale in case.rationales
    ]
    scores = []
    exact = False
    for reference in references:
        overlap = len(predicted & reference)
        if not predicted or not reference or not overlap:
            score = 0.0
        else:
            precision = overlap / len(predicted)
            recall = overlap / len(reference)
            score = 2 * precision * recall / (precision + recall)
        scores.append(score)
        exact = exact or predicted == reference
    return max(scores, default=0.0), exact


def run_scifact_benchmark(
    cases: list[SciFactCase],
    generator: TextGenerator,
    *,
    model: str,
    model_revision: str | None = None,
) -> SciFactReport:
    if not cases:
        raise ValueError("SciFact benchmark requires at least one case.")
    started = time.perf_counter()
    prompts = [build_scifact_prompt(case) for case in cases]
    responses = generator(prompts)
    if len(responses) != len(cases):
        raise ValueError("SciFact generator returned the wrong response count.")
    results = []
    for case, response in zip(cases, responses, strict=True):
        error = None
        try:
            prediction = parse_scifact_response(response, case)
            valid = True
        except (TypeError, ValueError) as exc:
            prediction = SciFactPrediction(
                label="NOT_ENOUGH_INFO", rationales=[], reason="Fail-closed parse fallback."
            )
            valid = False
            error = str(exc)
        rationale_f1 = rationale_exact = joint_exact = None
        if case.rationales:
            rationale_f1, rationale_exact = _rationale_scores(case, prediction)
            joint_exact = prediction.label == case.label and rationale_exact
        results.append(
            SciFactCaseResult(
                claim_id=case.claim_id,
                claim=case.claim,
                gold_label=case.label,
                predicted_label=prediction.label,
                label_correct=prediction.label == case.label,
                schema_valid=valid,
                rationale_f1=rationale_f1,
                rationale_exact=rationale_exact,
                joint_exact=joint_exact,
                predicted_rationales=prediction.rationales,
                parse_error=error,
                raw_response=response,
            )
        )
    gold = [result.gold_label for result in results]
    predicted = [result.predicted_label for result in results]
    rationale_results = [result for result in results if result.rationale_f1 is not None]
    nei_results = [result for result in results if result.gold_label == "NOT_ENOUGH_INFO"]
    return SciFactReport(
        source_split=cases[0].source_split,
        model=model,
        model_revision=model_revision,
        case_count=len(cases),
        model_calls=len(cases),
        generation_batches=int(getattr(generator, "batch_calls", 1)),
        parse_failure_count=sum(not result.schema_valid for result in results),
        latency_seconds=time.perf_counter() - started,
        label_metrics=classification_metrics(gold, predicted),
        support_detection_metrics=binary_sufficiency_metrics(
            [label == "SUPPORT" for label in gold],
            [label == "SUPPORT" for label in predicted],
        ),
        rationale_eligible_cases=len(rationale_results),
        rationale_sentence_f1=(
            sum(result.rationale_f1 or 0.0 for result in rationale_results)
            / len(rationale_results)
            if rationale_results
            else 0.0
        ),
        rationale_exact_match=(
            sum(bool(result.rationale_exact) for result in rationale_results)
            / len(rationale_results)
            if rationale_results
            else 0.0
        ),
        joint_label_rationale_exact_match=(
            sum(bool(result.joint_exact) for result in rationale_results)
            / len(rationale_results)
            if rationale_results
            else 0.0
        ),
        nei_spurious_rationale_rate=(
            sum(bool(result.predicted_rationales) for result in nei_results) / len(nei_results)
            if nei_results
            else 0.0
        ),
        results=results,
    )


def render_scifact_markdown(report: SciFactReport) -> str:
    lines = [
        "# SciFact oracle-document report",
        "",
        "External dev diagnostic with native labels; retrieval and LangGraph are not evaluated.",
        "",
        f"- Model: `{report.model}`",
        f"- Cases: `{report.case_count}`",
        f"- Label accuracy: `{report.label_metrics['accuracy']:.4f}`",
        f"- Label macro F1: `{report.label_metrics['macro_f1']:.4f}`",
        f"- Rationale sentence F1: `{report.rationale_sentence_f1:.4f}`",
        f"- Joint label+rationale exact: `{report.joint_label_rationale_exact_match:.4f}`",
        f"- Parse failures: `{report.parse_failure_count}`",
        "",
        "| Claim | Gold | Predicted | Rationale F1 | Parse error |",
        "|---:|---|---|---:|---|",
    ]
    for result in report.results:
        rationale = "n/a" if result.rationale_f1 is None else f"{result.rationale_f1:.4f}"
        lines.append(
            f"| {result.claim_id} | {result.gold_label} | {result.predicted_label} | "
            f"{rationale} | {result.parse_error or '—'} |"
        )
    return "\n".join(lines) + "\n"


def write_scifact_outputs(report: SciFactReport, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "scifact_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination / "scifact_report.md").write_text(
        render_scifact_markdown(report), encoding="utf-8"
    )
