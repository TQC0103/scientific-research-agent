"""Controlled evidence-sufficiency benchmark for the production verifier prompt."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from app.evaluation.loader import load_suite
from app.evaluation.metrics import binary_sufficiency_metrics
from app.evaluation.models import EvaluationSuite, StrictModel
from app.models.verifier import (
    EvidenceVerification,
    build_verifier_prompt,
    parse_verifier_response,
)

VERIFIER_BENCHMARK_CONTRACT_VERSION = "1.0.0"


class VerifierBenchmarkCaseDefinition(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_case_id: str
    scope_instruction: str | None = None
    initial_query: str
    evidence_ids: list[str] = Field(min_length=1)
    expected_sufficient: bool
    expected_supported_evidence_ids: list[str]
    recovery_evidence_ids: list[str] = Field(min_length=1)
    recovery_expected_sufficient: bool
    recovery_expected_supported_evidence_ids: list[str]
    challenge: str

    @model_validator(mode="after")
    def references_are_in_their_snapshots(self) -> VerifierBenchmarkCaseDefinition:
        if not set(self.expected_supported_evidence_ids) <= set(self.evidence_ids):
            raise ValueError("Initial supported evidence must belong to the initial snapshot.")
        if not set(self.recovery_expected_supported_evidence_ids) <= set(
            self.recovery_evidence_ids
        ):
            raise ValueError("Recovery supported evidence must belong to the recovery snapshot.")
        return self


class VerifierBenchmarkDefinition(StrictModel):
    contract_version: str
    suite_id: str
    dataset_version: str
    benchmark_status: str = Field(pattern=r"^development$")
    source_suite_id: str
    description: str
    cases: list[VerifierBenchmarkCaseDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def contract_and_cases_are_valid(self) -> VerifierBenchmarkDefinition:
        if self.contract_version != VERIFIER_BENCHMARK_CONTRACT_VERSION:
            raise ValueError(
                "Unsupported verifier contract_version; expected "
                f"{VERIFIER_BENCHMARK_CONTRACT_VERSION}."
            )
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Verifier benchmark case_id values must be unique.")
        return self


class VerifierSnapshot(StrictModel):
    evidence_ids: list[str]
    evidence: list[dict[str, Any]]
    expected_sufficient: bool
    expected_supported_indices: list[int]


class VerifierBenchmarkCase(StrictModel):
    case_id: str
    source_case_id: str
    question: str
    scope_instruction: str | None
    initial_query: str
    challenge: str
    initial: VerifierSnapshot
    recovery: VerifierSnapshot


class MaterializedVerifierBenchmark(StrictModel):
    suite_id: str
    dataset_version: str
    benchmark_status: str
    description: str
    cases: list[VerifierBenchmarkCase]


class VerifierStageResult(StrictModel):
    expected_sufficient: bool
    predicted_sufficient: bool
    expected_supported_indices: list[int]
    predicted_supported_indices: list[int]
    query: str
    reason: str
    missing_information: list[str]
    suggested_query: str | None
    parse_error: str | None = None


class VerifierCaseResult(StrictModel):
    case_id: str
    source_case_id: str
    challenge: str
    initial: VerifierStageResult
    recovery: VerifierStageResult | None
    rewrite_proposed: bool
    final_expected_sufficient: bool
    final_predicted_sufficient: bool
    flow_correct: bool


class VerifierBenchmarkReport(StrictModel):
    contract_version: str = VERIFIER_BENCHMARK_CONTRACT_VERSION
    suite_id: str
    dataset_version: str
    benchmark_status: str
    model: str
    model_revision: str | None
    case_count: int
    model_calls: int
    generation_batches: int
    parse_failure_count: int
    latency_seconds: float
    initial_metrics: dict[str, float | int]
    flow_accuracy: float
    supported_selection: dict[str, float | int]
    rewrite_metrics: dict[str, float | int]
    abstention_metrics: dict[str, float | int]
    results: list[VerifierCaseResult]


TextGenerator = Callable[[list[str]], list[str]]


def _evidence_payload(item: Any) -> dict[str, Any]:
    return {
        "arxiv_id": item.paper_id,
        "versioned_id": item.versioned_id,
        "page": item.page,
        "section": item.section,
        "text": item.quote,
    }


def _snapshot(
    evidence_ids: list[str],
    supported_ids: list[str],
    expected_sufficient: bool,
    evidence_by_id: dict[str, Any],
) -> VerifierSnapshot:
    missing = set(evidence_ids) - set(evidence_by_id)
    if missing:
        raise ValueError(f"Unknown verifier evidence IDs: {sorted(missing)}")
    positions = {evidence_id: index for index, evidence_id in enumerate(evidence_ids, 1)}
    return VerifierSnapshot(
        evidence_ids=evidence_ids,
        evidence=[_evidence_payload(evidence_by_id[item]) for item in evidence_ids],
        expected_sufficient=expected_sufficient,
        expected_supported_indices=sorted(positions[item] for item in supported_ids),
    )


def load_verifier_benchmark(
    definition_path: str | Path,
    source_suite_path: str | Path,
) -> MaterializedVerifierBenchmark:
    try:
        definition = VerifierBenchmarkDefinition.model_validate_json(
            Path(definition_path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load verifier benchmark {definition_path}: {exc}") from exc
    source: EvaluationSuite = load_suite(source_suite_path)
    if definition.source_suite_id != source.suite_id:
        raise ValueError(
            f"Verifier source_suite_id {definition.source_suite_id!r} does not match "
            f"{source.suite_id!r}."
        )
    source_cases = {case.case_id: case for case in source.cases}
    evidence_by_id = {
        evidence.evidence_id: evidence
        for case in source.cases
        for evidence in case.gold_evidence
    }
    materialized = []
    for case in definition.cases:
        source_case = source_cases.get(case.source_case_id)
        if source_case is None:
            raise ValueError(f"Unknown verifier source_case_id: {case.source_case_id}")
        materialized.append(
            VerifierBenchmarkCase(
                case_id=case.case_id,
                source_case_id=case.source_case_id,
                question=source_case.question,
                scope_instruction=case.scope_instruction,
                initial_query=case.initial_query,
                challenge=case.challenge,
                initial=_snapshot(
                    case.evidence_ids,
                    case.expected_supported_evidence_ids,
                    case.expected_sufficient,
                    evidence_by_id,
                ),
                recovery=_snapshot(
                    case.recovery_evidence_ids,
                    case.recovery_expected_supported_evidence_ids,
                    case.recovery_expected_sufficient,
                    evidence_by_id,
                ),
            )
        )
    return MaterializedVerifierBenchmark(
        suite_id=definition.suite_id,
        dataset_version=definition.dataset_version,
        benchmark_status=definition.benchmark_status,
        description=definition.description,
        cases=materialized,
    )


def _materially_different(query: str | None, current: str) -> bool:
    return bool(query and query.strip() and query.strip().casefold() != current.strip().casefold())


def _fallback_query(result: EvidenceVerification, current: str) -> str:
    proposed = (result.suggested_query or "").strip()
    if _materially_different(proposed, current):
        return proposed
    missing = " ".join(result.missing_information).strip()
    return (
        f"{missing} specific mechanism terminology section table".strip()
        if missing
        else f"{current} specific evidence section table"
    )


def _fail_closed(error: Exception, fallback_query: str) -> EvidenceVerification:
    return EvidenceVerification(
        sufficient=False,
        reason=f"Verifier response failed validation: {error}",
        missing_information=["A valid structured evidence decision."],
        suggested_query=fallback_query,
        supported_evidence=[],
    )


def _stage_result(
    expected: VerifierSnapshot,
    result: EvidenceVerification,
    query: str,
    parse_error: str | None,
) -> VerifierStageResult:
    return VerifierStageResult(
        expected_sufficient=expected.expected_sufficient,
        predicted_sufficient=result.sufficient,
        expected_supported_indices=expected.expected_supported_indices,
        predicted_supported_indices=result.supported_evidence,
        query=query,
        reason=result.reason,
        missing_information=result.missing_information,
        suggested_query=result.suggested_query,
        parse_error=parse_error,
    )


def _selection_metrics(results: list[VerifierCaseResult]) -> dict[str, float | int]:
    true_positive = false_positive = false_negative = 0
    stages = []
    for case in results:
        stages.append(case.initial)
        if case.recovery is not None:
            stages.append(case.recovery)
    for stage in stages:
        expected = set(stage.expected_supported_indices)
        predicted = set(stage.predicted_supported_indices)
        true_positive += len(expected & predicted)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "precision": true_positive / precision_denominator if precision_denominator else 0.0,
        "recall": true_positive / recall_denominator if recall_denominator else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "evaluated_stages": len(stages),
    }


def run_verifier_benchmark(
    benchmark: MaterializedVerifierBenchmark,
    generator: TextGenerator,
    *,
    model: str,
    model_revision: str | None = None,
) -> VerifierBenchmarkReport:
    started = time.perf_counter()
    initial_prompts = [
        build_verifier_prompt(
            case.question,
            case.initial.evidence,
            case.initial_query,
            case.scope_instruction,
        )
        for case in benchmark.cases
    ]
    initial_responses = generator(initial_prompts)
    if len(initial_responses) != len(benchmark.cases):
        raise ValueError("Verifier generator returned the wrong initial response count.")

    initial_results: list[tuple[EvidenceVerification, str | None]] = []
    for case, response in zip(benchmark.cases, initial_responses, strict=True):
        try:
            parsed = parse_verifier_response(
                response,
                evidence_count=len(case.initial.evidence),
                fallback_query=case.initial_query,
            )
            initial_results.append((parsed, None))
        except (ValueError, TypeError) as exc:
            initial_results.append((_fail_closed(exc, case.initial_query), str(exc)))

    recovery_cases = []
    recovery_queries = []
    recovery_prompts = []
    for index, (case, (initial, _error)) in enumerate(
        zip(benchmark.cases, initial_results, strict=True)
    ):
        if initial.sufficient:
            continue
        query = _fallback_query(initial, case.initial_query)
        recovery_cases.append(index)
        recovery_queries.append(query)
        recovery_prompts.append(
            build_verifier_prompt(
                case.question,
                case.recovery.evidence,
                query,
                case.scope_instruction,
            )
        )

    recovery_results: dict[int, tuple[EvidenceVerification, str | None]] = {}
    if recovery_prompts:
        responses = generator(recovery_prompts)
        if len(responses) != len(recovery_prompts):
            raise ValueError("Verifier generator returned the wrong recovery response count.")
        for index, query, response in zip(
            recovery_cases, recovery_queries, responses, strict=True
        ):
            case = benchmark.cases[index]
            try:
                parsed = parse_verifier_response(
                    response,
                    evidence_count=len(case.recovery.evidence),
                    fallback_query=query,
                )
                recovery_results[index] = (parsed, None)
            except (ValueError, TypeError) as exc:
                recovery_results[index] = (_fail_closed(exc, query), str(exc))

    results = []
    for index, (case, (initial, initial_error)) in enumerate(
        zip(benchmark.cases, initial_results, strict=True)
    ):
        recovery_pair = recovery_results.get(index)
        recovery_stage = None
        final = initial.sufficient
        if recovery_pair is not None:
            recovery, recovery_error = recovery_pair
            recovery_stage = _stage_result(
                case.recovery,
                recovery,
                recovery_queries[recovery_cases.index(index)],
                recovery_error,
            )
            final = recovery.sufficient
        results.append(
            VerifierCaseResult(
                case_id=case.case_id,
                source_case_id=case.source_case_id,
                challenge=case.challenge,
                initial=_stage_result(
                    case.initial, initial, case.initial_query, initial_error
                ),
                recovery=recovery_stage,
                rewrite_proposed=_materially_different(
                    initial.suggested_query, case.initial_query
                ),
                final_expected_sufficient=case.recovery.expected_sufficient,
                final_predicted_sufficient=final,
                flow_correct=(
                    final
                    if case.initial.expected_sufficient
                    else (
                        not initial.sufficient
                        and recovery_stage is not None
                        and recovery_stage.predicted_sufficient
                        == case.recovery.expected_sufficient
                    )
                ),
            )
        )

    initial_gold = [case.initial.expected_sufficient for case in benchmark.cases]
    initial_predictions = [case.initial.predicted_sufficient for case in results]
    recoverable = [
        case
        for case in results
        if not case.initial.expected_sufficient and case.final_expected_sufficient
    ]
    abstentions = [case for case in results if not case.final_expected_sufficient]
    parse_failures = sum(
        bool(case.initial.parse_error)
        + bool(case.recovery and case.recovery.parse_error)
        for case in results
    )
    return VerifierBenchmarkReport(
        suite_id=benchmark.suite_id,
        dataset_version=benchmark.dataset_version,
        benchmark_status=benchmark.benchmark_status,
        model=model,
        model_revision=model_revision,
        case_count=len(results),
        model_calls=len(initial_prompts) + len(recovery_prompts),
        generation_batches=int(
            getattr(generator, "batch_calls", 1 + bool(recovery_prompts))
        ),
        parse_failure_count=parse_failures,
        latency_seconds=time.perf_counter() - started,
        initial_metrics=binary_sufficiency_metrics(initial_gold, initial_predictions),
        flow_accuracy=sum(case.flow_correct for case in results) / len(results),
        supported_selection=_selection_metrics(results),
        rewrite_metrics={
            "eligible_cases": len(recoverable),
            "proposal_rate": (
                sum(case.rewrite_proposed for case in recoverable) / len(recoverable)
                if recoverable
                else 0.0
            ),
            "execution_rate": (
                sum(case.recovery is not None for case in recoverable) / len(recoverable)
                if recoverable
                else 0.0
            ),
            "recovery_rate": (
                sum(
                    case.recovery is not None
                    and case.recovery.predicted_sufficient
                    for case in recoverable
                )
                / len(recoverable)
                if recoverable
                else 0.0
            ),
        },
        abstention_metrics={
            "eligible_cases": len(abstentions),
            "accuracy": (
                sum(
                    not case.initial.predicted_sufficient
                    and case.recovery is not None
                    and not case.recovery.predicted_sufficient
                    for case in abstentions
                )
                / len(abstentions)
                if abstentions
                else 0.0
            ),
        },
        results=results,
    )


def render_verifier_markdown(report: VerifierBenchmarkReport) -> str:
    lines = [
        "# Internal verifier benchmark",
        "",
        "Development diagnostic only; this is not held-out accuracy.",
        "",
        f"- Model: `{report.model}`",
        f"- Initial accuracy: `{report.initial_metrics['accuracy']:.4f}`",
        f"- Initial false-positive rate: `{report.initial_metrics['false_positive_rate']:.4f}`",
        f"- Initial false-negative rate: `{report.initial_metrics['false_negative_rate']:.4f}`",
        f"- Bounded-flow accuracy: `{report.flow_accuracy:.4f}`",
        f"- Rewrite recovery rate: `{report.rewrite_metrics['recovery_rate']:.4f}`",
        f"- Final abstention accuracy: `{report.abstention_metrics['accuracy']:.4f}`",
        f"- Supported-passage precision: `{report.supported_selection['precision']:.4f}`",
        f"- Supported-passage recall: `{report.supported_selection['recall']:.4f}`",
        "",
        "| Case | Initial gold/pred | Recovery gold/pred | Final | Rewrite | Parse error |",
        "|---|---|---|---|---|---|",
    ]
    for case in report.results:
        recovery = (
            f"{case.recovery.expected_sufficient}/{case.recovery.predicted_sufficient}"
            if case.recovery
            else "not run"
        )
        parse_error = case.initial.parse_error or (
            case.recovery.parse_error if case.recovery else None
        )
        lines.append(
            f"| {case.case_id} | {case.initial.expected_sufficient}/"
            f"{case.initial.predicted_sufficient} | {recovery} | "
            f"{case.final_expected_sufficient}/{case.final_predicted_sufficient} | "
            f"{case.rewrite_proposed} | {parse_error or '—'} |"
        )
    return "\n".join(lines) + "\n"


def write_verifier_outputs(report: VerifierBenchmarkReport, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "verifier_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination / "verifier_report.md").write_text(
        render_verifier_markdown(report), encoding="utf-8"
    )
