"""Controlled development benchmark for atomic claim verification."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from app.evaluation.citations import (
    CitationClaimRecord,
    CitationSafetyCase,
    CitationSafetySuite,
    citation_case_from_claim_verification,
    evaluate_citation_safety,
)
from app.evaluation.models import StrictModel
from app.models.claim_verifier import build_claim_verifier_prompt, parse_claim_verifier_response
from app.models.claims import ClaimVerificationBundle

CLAIM_VERIFIER_BENCHMARK_VERSION = "1.0.0"
TextGenerator = Callable[[list[str]], list[str]]


class ClaimVerifierCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    evidence: list[dict[str, Any]] = Field(min_length=1)
    expected: ClaimVerificationBundle
    challenge: str = Field(min_length=1)

    @model_validator(mode="after")
    def immutable_inputs_match_expected(self) -> ClaimVerifierCase:
        if self.expected.answer != self.answer:
            raise ValueError("Expected bundle must echo the case answer.")
        if self.expected.evidence_count != len(self.evidence):
            raise ValueError("Expected evidence_count must match case evidence.")
        return self


class ClaimVerifierSuite(StrictModel):
    contract_version: str
    suite_id: str
    dataset_version: str
    benchmark_status: Literal["development"]
    description: str
    cases: list[ClaimVerifierCase] = Field(min_length=1)

    @model_validator(mode="after")
    def contract_and_cases_are_valid(self) -> ClaimVerifierSuite:
        if self.contract_version != CLAIM_VERIFIER_BENCHMARK_VERSION:
            raise ValueError(
                "Unsupported claim-verifier benchmark contract_version; expected "
                f"{CLAIM_VERIFIER_BENCHMARK_VERSION}."
            )
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Claim-verifier case_id values must be unique.")
        return self


class ClaimVerifierCaseResult(StrictModel):
    case_id: str
    challenge: str
    schema_valid: bool
    exact_match: bool
    extraction_exact: bool
    parse_error: str | None = None
    raw_response: str
    expected: ClaimVerificationBundle
    predicted: ClaimVerificationBundle | None = None


class ClaimVerifierReport(StrictModel):
    contract_version: str = CLAIM_VERIFIER_BENCHMARK_VERSION
    suite_id: str
    dataset_version: str
    benchmark_status: str
    model: str
    model_revision: str | None
    case_count: int
    model_calls: int
    generation_batches: int
    latency_seconds: float
    schema_valid_rate: float
    exact_case_rate: float
    extraction_exact_case_rate: float
    claim_verdict_accuracy: float
    evidence_relationship_accuracy: float
    citation_metrics: dict[str, float | int | None]
    citation_eligible_cases: int
    results: list[ClaimVerifierCaseResult]


def load_claim_verifier_suite(path: str | Path) -> ClaimVerifierSuite:
    try:
        return ClaimVerifierSuite.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load claim-verifier benchmark {path}: {exc}") from exc


def _extraction_signature(bundle: ClaimVerificationBundle) -> list[tuple[Any, ...]]:
    return [
        (claim.source_text, claim.requires_citation, claim.citation_labels)
        for claim in bundle.claims
    ]


def _assessment_signature(bundle: ClaimVerificationBundle) -> list[tuple[Any, ...]]:
    return [
        (
            assessment.claim_id,
            assessment.verdict,
            [
                (link.citation_label, link.relationship)
                for link in assessment.cited_evidence
            ],
        )
        for assessment in bundle.assessments
    ]


def _exact(expected: ClaimVerificationBundle, predicted: ClaimVerificationBundle) -> bool:
    return (
        _extraction_signature(expected) == _extraction_signature(predicted)
        and _assessment_signature(expected) == _assessment_signature(predicted)
    )


def _micro_accuracy(
    cases: list[ClaimVerifierCase], predictions: list[ClaimVerificationBundle | None]
) -> tuple[float, float]:
    verdict_correct = relationship_correct = 0
    verdict_total = relationship_total = 0
    for case, prediction in zip(cases, predictions, strict=True):
        predicted_assessments = prediction.assessments if prediction else []
        for index, expected in enumerate(case.expected.assessments):
            verdict_total += 1
            predicted = (
                predicted_assessments[index]
                if index < len(predicted_assessments)
                and predicted_assessments[index].claim_id == expected.claim_id
                else None
            )
            verdict_correct += bool(predicted and predicted.verdict == expected.verdict)
            for link_index, expected_link in enumerate(expected.cited_evidence):
                relationship_total += 1
                predicted_link = (
                    predicted.cited_evidence[link_index]
                    if predicted and link_index < len(predicted.cited_evidence)
                    else None
                )
                relationship_correct += bool(
                    predicted_link
                    and predicted_link.citation_label == expected_link.citation_label
                    and predicted_link.relationship == expected_link.relationship
                )
    return (
        verdict_correct / verdict_total if verdict_total else 0.0,
        relationship_correct / relationship_total if relationship_total else 0.0,
    )


def run_claim_verifier_benchmark(
    suite: ClaimVerifierSuite,
    generator: TextGenerator,
    *,
    model: str,
    model_revision: str | None = None,
) -> ClaimVerifierReport:
    started = time.perf_counter()
    prompts = [
        build_claim_verifier_prompt(case.answer, case.evidence, case.question)
        for case in suite.cases
    ]
    responses = generator(prompts)
    if len(responses) != len(suite.cases):
        raise ValueError("Claim-verifier generator returned the wrong response count.")

    predictions: list[ClaimVerificationBundle | None] = []
    errors: list[str | None] = []
    for case, response in zip(suite.cases, responses, strict=True):
        try:
            prediction = parse_claim_verifier_response(
                response,
                expected_answer=case.answer,
                evidence_count=len(case.evidence),
            )
            predictions.append(prediction)
            errors.append(None)
        except (TypeError, ValueError) as exc:
            predictions.append(None)
            errors.append(str(exc))

    results = []
    for case, prediction, error in zip(suite.cases, predictions, errors, strict=True):
        extraction_exact = bool(
            prediction
            and _extraction_signature(case.expected) == _extraction_signature(prediction)
        )
        results.append(
            ClaimVerifierCaseResult(
                case_id=case.case_id,
                challenge=case.challenge,
                schema_valid=prediction is not None,
                exact_match=bool(prediction and _exact(case.expected, prediction)),
                extraction_exact=extraction_exact,
                parse_error=error,
                raw_response=response,
                expected=case.expected,
                predicted=prediction,
            )
        )

    citation_cases = []
    for case, prediction in zip(suite.cases, predictions, strict=True):
        evidence_ids = [f"evidence_{index}" for index in range(1, len(case.evidence) + 1)]
        if prediction is not None:
            citation_cases.append(
                citation_case_from_claim_verification(
                    prediction, case_id=case.case_id, evidence_ids=evidence_ids
                )
            )
            continue
        citation_cases.append(
            CitationSafetyCase(
                case_id=case.case_id,
                available_evidence_ids=evidence_ids,
                claims=[
                    CitationClaimRecord(
                        claim_id=claim.claim_id,
                        text=claim.claim_text,
                        requires_citation=claim.requires_citation,
                        cited_evidence_ids=[],
                        supporting_evidence_ids=[],
                    )
                    for claim in case.expected.claims
                ],
            )
        )
    citation_report = evaluate_citation_safety(
        CitationSafetySuite(
            contract_version="1.0.0",
            suite_id=f"{suite.suite_id}-predictions",
            dataset_version=suite.dataset_version,
            benchmark_status="development",
            description="Citation metrics over schema-valid claim-verifier predictions.",
            cases=citation_cases,
        )
    )
    verdict_accuracy, relationship_accuracy = _micro_accuracy(suite.cases, predictions)
    case_count = len(suite.cases)
    return ClaimVerifierReport(
        suite_id=suite.suite_id,
        dataset_version=suite.dataset_version,
        benchmark_status=suite.benchmark_status,
        model=model,
        model_revision=model_revision,
        case_count=case_count,
        model_calls=case_count,
        generation_batches=int(getattr(generator, "batch_calls", 1)),
        latency_seconds=time.perf_counter() - started,
        schema_valid_rate=sum(result.schema_valid for result in results) / case_count,
        exact_case_rate=sum(result.exact_match for result in results) / case_count,
        extraction_exact_case_rate=(
            sum(result.extraction_exact for result in results) / case_count
        ),
        claim_verdict_accuracy=verdict_accuracy,
        evidence_relationship_accuracy=relationship_accuracy,
        citation_metrics=citation_report.metrics,
        citation_eligible_cases=len(citation_cases),
        results=results,
    )


def render_claim_verifier_markdown(report: ClaimVerifierReport) -> str:
    lines = [
        "# Claim verifier development benchmark",
        "",
        "Synthetic diagnostic only; this is not independently reviewed or held-out accuracy.",
        "",
        f"- Model: `{report.model}`",
        f"- Schema-valid rate: `{report.schema_valid_rate:.4f}`",
        f"- Exact-case rate: `{report.exact_case_rate:.4f}`",
        f"- Extraction exact-case rate: `{report.extraction_exact_case_rate:.4f}`",
        f"- Claim verdict accuracy: `{report.claim_verdict_accuracy:.4f}`",
        f"- Evidence relationship accuracy: `{report.evidence_relationship_accuracy:.4f}`",
        "",
        "| Case | Challenge | Schema | Extraction | Exact | Error |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in report.results:
        lines.append(
            f"| {result.case_id} | {result.challenge} | {result.schema_valid} | "
            f"{result.extraction_exact} | {result.exact_match} | {result.parse_error or '—'} |"
        )
    return "\n".join(lines) + "\n"


def write_claim_verifier_outputs(
    report: ClaimVerifierReport, output_dir: str | Path
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "claim_verifier_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination / "claim_verifier_report.md").write_text(
        render_claim_verifier_markdown(report), encoding="utf-8"
    )
