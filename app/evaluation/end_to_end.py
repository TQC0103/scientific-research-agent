"""Versioned end-to-end evaluation over the compiled production graph."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.agent.graph import MAX_CLAIM_REVISIONS
from app.config import settings
from app.evaluation.metrics import token_f1
from app.evaluation.models import EvaluationCase, EvaluationSuite, ExpectedDecision, StrictModel
from app.evaluation.retrieval import RetrievalCaseMetrics, evaluate_case_retrieval
from app.models.claim_verifier import answer_body
from app.models.claims import ClaimVerdict, ClaimVerificationBundle

END_TO_END_REPORT_VERSION = "1.0.0"
NODE_TRACE_KEY = "__evaluation_node_trace__"
GraphInvoker = Callable[[dict[str, Any], dict[str, Any]], Mapping[str, Any]]


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


# Baseline comparison is intentionally opt-in. A new trace field or aggregate
# value receives no quality direction until its meaning is explicitly registered.
METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "decision_accuracy": MetricDirection.HIGHER,
    "answer_case_decision_accuracy": MetricDirection.HIGHER,
    "abstention_accuracy": MetricDirection.HIGHER,
    "answer_f1": MetricDirection.HIGHER,
    "retrieval_recall_at_k": MetricDirection.HIGHER,
    "retrieval_precision_at_k": MetricDirection.HIGHER,
    "retrieval_mrr": MetricDirection.HIGHER,
    "gold_evidence_coverage": MetricDirection.HIGHER,
    "required_paper_coverage": MetricDirection.HIGHER,
    "macro_paper_recall": MetricDirection.HIGHER,
    "verifier_supported_claim_rate": MetricDirection.HIGHER,
    "citation_complete_claim_rate": MetricDirection.HIGHER,
    "post_revision_success_rate": MetricDirection.HIGHER,
    "execution_failure_rate": MetricDirection.LOWER,
    "citation_safety_failure_rate": MetricDirection.LOWER,
    "claim_verifier_failure_rate": MetricDirection.LOWER,
    "verifier_partial_claim_rate": MetricDirection.LOWER,
    "verifier_unsupported_claim_rate": MetricDirection.LOWER,
    "mean_latency_seconds": MetricDirection.LOWER,
}


class LlmCallCounts(StrictModel):
    evidence_verifier: int = Field(ge=0)
    synthesis: int = Field(ge=0)
    claim_verifier: int = Field(ge=0)
    claim_repair: int = Field(ge=0)
    total: int = Field(ge=0)

    @model_validator(mode="after")
    def total_agrees(self) -> LlmCallCounts:
        expected = (
            self.evidence_verifier
            + self.synthesis
            + self.claim_verifier
            + self.claim_repair
        )
        if self.total != expected:
            raise ValueError("LLM call total must equal the per-node counts.")
        return self


class GraphNodeEvent(StrictModel):
    sequence: int = Field(ge=1)
    node: str = Field(min_length=1)
    update: dict[str, Any]


class EndToEndCaseResult(StrictModel):
    case_id: str
    question: str
    expected_decision: str
    expected_abstention_reason: str | None
    predicted_decision: str
    decision_correct: bool
    answer: str
    answer_f1: float | None
    evidence_sufficient: bool
    synthesis_citation_valid: bool
    claim_verification_status: str
    claim_verification_attempt_count: int = Field(ge=0)
    claim_revision_count: int = Field(ge=0)
    claim_metrics: dict[str, float | int | None]
    retrieval: RetrievalCaseMetrics
    llm_calls: LlmCallCounts
    embedding_calls: int | None = None
    latency_seconds: float = Field(ge=0)
    failure_reasons: list[str]
    execution_error: str | None
    node_trace: list[GraphNodeEvent]
    trace: dict[str, Any]


class RuntimeMetadata(StrictModel):
    python: str
    platform: str
    git_commit: str | None
    git_dirty: bool | None
    ollama_model: str
    embedding_model: str
    max_retrieval_rewrites: int
    max_claim_revisions: int


class BaselineMetricDelta(StrictModel):
    metric: str
    direction: MetricDirection
    baseline: float
    current: float
    delta: float
    outcome: str


class BaselineComparison(StrictModel):
    baseline_run_id: str
    compared_metric_count: int
    improved_count: int
    regressed_count: int
    unchanged_count: int
    deltas: list[BaselineMetricDelta]


class EndToEndAggregate(StrictModel):
    contract_version: str = END_TO_END_REPORT_VERSION
    run_id: str = Field(min_length=1)
    generated_at: datetime
    suite_id: str
    dataset_version: str
    suite_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_ids: list[str]
    benchmark_status: str
    config_name: str = Field(min_length=1)
    retrieval_k_per_paper: int
    quote_token_recall_threshold: float
    case_count: int
    completed_cases: int
    execution_failures: int
    runtime: RuntimeMetadata
    metrics: dict[str, float | int | None]
    metric_directions: dict[str, MetricDirection]
    baseline_comparison: BaselineComparison | None = None

    @model_validator(mode="after")
    def contract_and_case_identity_are_valid(self) -> EndToEndAggregate:
        if self.contract_version != END_TO_END_REPORT_VERSION:
            raise ValueError(
                "Unsupported end-to-end contract_version; expected "
                f"{END_TO_END_REPORT_VERSION}."
            )
        if len(self.case_ids) != self.case_count or len(set(self.case_ids)) != len(
            self.case_ids
        ):
            raise ValueError("case_ids must be unique and match case_count.")
        return self


class EndToEndReport(StrictModel):
    aggregate: EndToEndAggregate
    cases: list[EndToEndCaseResult]

    @model_validator(mode="after")
    def counts_agree(self) -> EndToEndReport:
        if self.aggregate.case_count != len(self.cases):
            raise ValueError("Aggregate case_count must equal the result count.")
        if self.aggregate.case_ids != [case.case_id for case in self.cases]:
            raise ValueError("Aggregate case_ids must match per-case result order.")
        return self


def end_to_end_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            "https://github.com/TQC0103/scientific-research-agent/"
            "evaluation/schema/end-to-end-report.schema.json"
        ),
        **EndToEndReport.model_json_schema(),
    }


def _json_safe(value: Any) -> Any:
    """Keep arbitrary new graph-state fields serializable without interpreting them."""
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, Path, StrEnum)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _runtime_metadata() -> RuntimeMetadata:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        git_commit = completed.stdout.strip() if completed.returncode == 0 else None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        git_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        git_commit = None
        git_dirty = None
    return RuntimeMetadata(
        python=sys.version,
        platform=platform.platform(),
        git_commit=git_commit,
        git_dirty=git_dirty,
        ollama_model=settings.ollama_model,
        embedding_model=settings.ollama_embed_model,
        max_retrieval_rewrites=settings.max_retrieval_rewrites,
        max_claim_revisions=MAX_CLAIM_REVISIONS,
    )


def _mean(values: Sequence[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _retrieved_for_case(
    case: EvaluationCase,
    state: Mapping[str, Any],
    *,
    k_per_paper: int,
) -> tuple[list[dict[str, Any]], int]:
    by_paper = state.get("retrieved_chunks_by_paper") or {}
    ranked: list[dict[str, Any]] = []
    if isinstance(by_paper, Mapping):
        for paper in case.papers:
            items = by_paper.get(paper.paper_id, [])
            if isinstance(items, list):
                ranked.extend(items[:k_per_paper])
    if not ranked:
        raw = state.get("retrieved_chunks", [])
        if isinstance(raw, list):
            ranked = raw[: k_per_paper * len(case.papers)]
    return ranked, k_per_paper * len(case.papers)


def _claim_metrics(state: Mapping[str, Any]) -> dict[str, float | int | None]:
    payload = state.get("claim_verification")
    if not isinstance(payload, Mapping) or not payload:
        return {
            "claim_count": 0,
            "citation_required_claims": 0,
            "supported_claims": 0,
            "partial_claims": 0,
            "unsupported_claims": 0,
            "not_required_claims": 0,
            "citation_complete_claims": 0,
            "verifier_supported_claim_rate": None,
            "verifier_partial_claim_rate": None,
            "verifier_unsupported_claim_rate": None,
            "citation_complete_claim_rate": None,
        }
    try:
        bundle = ClaimVerificationBundle.model_validate(payload)
    except ValueError:
        return {
            "claim_count": 0,
            "citation_required_claims": 0,
            "supported_claims": 0,
            "partial_claims": 0,
            "unsupported_claims": 0,
            "not_required_claims": 0,
            "citation_complete_claims": 0,
            "verifier_supported_claim_rate": None,
            "verifier_partial_claim_rate": None,
            "verifier_unsupported_claim_rate": None,
            "citation_complete_claim_rate": None,
        }
    verdicts = [assessment.verdict for assessment in bundle.assessments]
    required_claims = [claim for claim in bundle.claims if claim.requires_citation]
    complete = sum(bool(claim.citation_labels) for claim in required_claims)
    supported = verdicts.count(ClaimVerdict.SUPPORTED)
    partial = verdicts.count(ClaimVerdict.PARTIAL)
    unsupported = verdicts.count(ClaimVerdict.UNSUPPORTED)
    factual = supported + partial + unsupported
    return {
        "claim_count": len(bundle.claims),
        "citation_required_claims": len(required_claims),
        "supported_claims": supported,
        "partial_claims": partial,
        "unsupported_claims": unsupported,
        "not_required_claims": verdicts.count(ClaimVerdict.NOT_REQUIRED),
        "citation_complete_claims": complete,
        "verifier_supported_claim_rate": _ratio(supported, factual),
        "verifier_partial_claim_rate": _ratio(partial, factual),
        "verifier_unsupported_claim_rate": _ratio(unsupported, factual),
        "citation_complete_claim_rate": _ratio(complete, len(required_claims)),
    }


def _llm_calls(state: Mapping[str, Any]) -> LlmCallCounts:
    attempts = state.get("retrieval_attempt_counts") or {}
    evidence_calls = (
        sum(int(value) for value in attempts.values())
        if isinstance(attempts, Mapping)
        else int(state.get("retrieval_attempt_count", 0) or 0)
    )
    synthesis = int(bool(state.get("evidence_sufficient")))
    claims = int(state.get("claim_verification_attempt_count", 0) or 0)
    repairs = int(state.get("claim_revision_count", 0) or 0)
    return LlmCallCounts(
        evidence_verifier=evidence_calls,
        synthesis=synthesis,
        claim_verifier=claims,
        claim_repair=repairs,
        total=evidence_calls + synthesis + claims + repairs,
    )


def _failure_reasons(
    case: EvaluationCase,
    state: Mapping[str, Any],
    retrieval: RetrievalCaseMetrics,
    *,
    predicted_decision: str,
    execution_error: str | None,
) -> list[str]:
    reasons: list[str] = []
    if execution_error:
        reasons.append("execution_error")
    elif predicted_decision != case.expected.decision.value:
        reasons.append(
            "unexpected_abstention"
            if predicted_decision == ExpectedDecision.ABSTAIN.value
            else "unexpected_answer"
        )
    if case.expected.decision == ExpectedDecision.ANSWER and not state.get(
        "evidence_sufficient"
    ):
        reasons.append("insufficient_evidence_for_answer_case")
    if state.get("evidence_sufficient") and not state.get("synthesis_citation_valid"):
        reasons.append("citation_safety_failure")
    if state.get("claim_verification_error"):
        reasons.append("claim_verifier_error")
    if state.get("claim_verification_status") == "abstained":
        reasons.append("claim_grounding_abstention")
    reasons.extend(
        f"retrieval:{item}"
        for item in retrieval.diagnostics
        if item != "not_applicable_no_gold_evidence"
    )
    if state.get("tool_errors"):
        reasons.append("tool_error")
    return list(dict.fromkeys(reasons))


def evaluate_end_to_end_case(
    case: EvaluationCase,
    invoke: GraphInvoker,
    *,
    retrieval_k_per_paper: int = 5,
    quote_token_recall_threshold: float = 0.8,
    recursion_limit: int = 30,
) -> EndToEndCaseResult:
    if retrieval_k_per_paper < 1:
        raise ValueError("retrieval_k_per_paper must be at least 1.")
    started = time.perf_counter()
    execution_error = None
    try:
        raw_state = invoke(
            {
                "user_query": case.question,
                "paper_ids": [paper.paper_id for paper in case.papers],
            },
            {"recursion_limit": recursion_limit},
        )
        state = dict(raw_state)
    except Exception as exc:  # noqa: BLE001 - one bad case must not abort the suite
        execution_error = f"{type(exc).__name__}: {exc}"
        state = {
            "user_query": case.question,
            "paper_ids": [paper.paper_id for paper in case.papers],
            "answer": "",
            "evidence_sufficient": False,
            "synthesis_citation_valid": False,
            "claim_verification_status": "execution_error",
            "claim_verification_attempt_count": 0,
            "claim_revision_count": 0,
            "tool_errors": [execution_error],
        }
    raw_node_trace = state.pop(NODE_TRACE_KEY, [])
    node_trace = []
    if isinstance(raw_node_trace, list):
        for sequence, event in enumerate(raw_node_trace, 1):
            if not isinstance(event, Mapping) or not event.get("node"):
                continue
            update = event.get("update")
            node_trace.append(
                GraphNodeEvent(
                    sequence=sequence,
                    node=str(event["node"]),
                    update=_json_safe(update) if isinstance(update, Mapping) else {},
                )
            )
    latency = time.perf_counter() - started
    retrieved, effective_k = _retrieved_for_case(
        case, state, k_per_paper=retrieval_k_per_paper
    )
    retrieval = evaluate_case_retrieval(
        case,
        retrieved,
        top_k=effective_k,
        quote_token_recall_threshold=quote_token_recall_threshold,
    )
    claim_status = str(state.get("claim_verification_status") or "not_run")
    predicted = (
        ExpectedDecision.ANSWER.value
        if claim_status == "verified"
        else ExpectedDecision.ABSTAIN.value
    )
    decision_correct = bool(
        not execution_error and predicted == case.expected.decision.value
    )
    answer = str(state.get("answer") or "")
    answer_score = (
        token_f1(answer_body(answer), case.expected.reference_answer)
        if case.expected.reference_answer
        else None
    )
    failures = _failure_reasons(
        case,
        state,
        retrieval,
        predicted_decision=predicted,
        execution_error=execution_error,
    )
    return EndToEndCaseResult(
        case_id=case.case_id,
        question=case.question,
        expected_decision=case.expected.decision.value,
        expected_abstention_reason=(
            case.expected.abstention_reason.value
            if case.expected.abstention_reason
            else None
        ),
        predicted_decision=predicted,
        decision_correct=decision_correct,
        answer=answer,
        answer_f1=answer_score,
        evidence_sufficient=bool(state.get("evidence_sufficient")),
        synthesis_citation_valid=bool(state.get("synthesis_citation_valid")),
        claim_verification_status=claim_status,
        claim_verification_attempt_count=int(
            state.get("claim_verification_attempt_count", 0) or 0
        ),
        claim_revision_count=int(state.get("claim_revision_count", 0) or 0),
        claim_metrics=_claim_metrics(state),
        retrieval=retrieval,
        llm_calls=_llm_calls(state),
        latency_seconds=latency,
        failure_reasons=failures,
        execution_error=execution_error,
        node_trace=node_trace,
        trace=_json_safe(state),
    )


def _aggregate_metrics(results: Sequence[EndToEndCaseResult]) -> dict[str, float | int | None]:
    answer_cases = [case for case in results if case.expected_decision == "answer"]
    abstention_cases = [case for case in results if case.expected_decision == "abstain"]
    completed = [case for case in results if case.execution_error is None]
    retrieval = [case.retrieval for case in results if case.retrieval.eligible]
    revised = [case for case in results if case.claim_revision_count > 0]
    claim_totals = {
        name: sum(int(case.claim_metrics[name]) for case in results)
        for name in (
            "supported_claims",
            "partial_claims",
            "unsupported_claims",
            "citation_required_claims",
            "citation_complete_claims",
        )
    }
    factual = (
        claim_totals["supported_claims"]
        + claim_totals["partial_claims"]
        + claim_totals["unsupported_claims"]
    )
    execution_failures = len(results) - len(completed)
    citation_failures = sum(
        case.evidence_sufficient and not case.synthesis_citation_valid for case in results
    )
    claim_failures = sum(
        bool(case.trace.get("claim_verification_error")) for case in results
    )
    return {
        "decision_accuracy": _ratio(sum(case.decision_correct for case in results), len(results)),
        "answer_case_decision_accuracy": _ratio(
            sum(case.decision_correct for case in answer_cases), len(answer_cases)
        ),
        "abstention_accuracy": _ratio(
            sum(case.decision_correct for case in abstention_cases), len(abstention_cases)
        ),
        "answer_f1": _mean([case.answer_f1 for case in answer_cases]),
        "retrieval_recall_at_k": _mean([case.recall_at_k for case in retrieval]),
        "retrieval_precision_at_k": _mean([case.precision_at_k for case in retrieval]),
        "retrieval_mrr": _mean([case.reciprocal_rank for case in retrieval]),
        "gold_evidence_coverage": _mean(
            [case.gold_evidence_coverage for case in retrieval]
        ),
        "required_paper_coverage": _mean(
            [case.required_paper_coverage for case in retrieval]
        ),
        "macro_paper_recall": _mean([case.macro_paper_recall for case in retrieval]),
        "verifier_supported_claim_rate": _ratio(
            claim_totals["supported_claims"], factual
        ),
        "verifier_partial_claim_rate": _ratio(claim_totals["partial_claims"], factual),
        "verifier_unsupported_claim_rate": _ratio(
            claim_totals["unsupported_claims"], factual
        ),
        "citation_complete_claim_rate": _ratio(
            claim_totals["citation_complete_claims"],
            claim_totals["citation_required_claims"],
        ),
        "revision_rate": _ratio(len(revised), len(results)),
        "post_revision_success_rate": _ratio(
            sum(case.claim_verification_status == "verified" for case in revised),
            len(revised),
        ),
        "citation_safety_failure_rate": _ratio(citation_failures, len(results)),
        "claim_verifier_failure_rate": _ratio(claim_failures, len(results)),
        "execution_failure_rate": _ratio(execution_failures, len(results)),
        "mean_latency_seconds": _mean([case.latency_seconds for case in results]),
        "total_latency_seconds": sum(case.latency_seconds for case in results),
        "llm_calls": sum(case.llm_calls.total for case in results),
        "embedding_calls": None,
        "cases_with_tool_errors": sum(bool(case.trace.get("tool_errors")) for case in results),
    }


def compare_with_baseline(
    current: EndToEndAggregate,
    baseline: EndToEndAggregate,
) -> BaselineComparison:
    if (
        current.suite_id,
        current.dataset_version,
        current.config_name,
    ) != (
        baseline.suite_id,
        baseline.dataset_version,
        baseline.config_name,
    ):
        raise ValueError("Baseline suite, dataset version, and config must match the current run.")
    if (
        current.case_count != baseline.case_count
        or current.case_ids != baseline.case_ids
        or current.suite_fingerprint != baseline.suite_fingerprint
    ):
        raise ValueError("Baseline and current run must contain the exact same suite cases.")
    deltas = []
    for metric, direction in METRIC_DIRECTIONS.items():
        current_value = current.metrics.get(metric)
        baseline_value = baseline.metrics.get(metric)
        if not isinstance(current_value, (int, float)) or not isinstance(
            baseline_value, (int, float)
        ):
            continue
        delta = float(current_value) - float(baseline_value)
        signed = delta if direction == MetricDirection.HIGHER else -delta
        outcome = "improved" if signed > 0 else "regressed" if signed < 0 else "unchanged"
        deltas.append(
            BaselineMetricDelta(
                metric=metric,
                direction=direction,
                baseline=float(baseline_value),
                current=float(current_value),
                delta=delta,
                outcome=outcome,
            )
        )
    return BaselineComparison(
        baseline_run_id=baseline.run_id,
        compared_metric_count=len(deltas),
        improved_count=sum(item.outcome == "improved" for item in deltas),
        regressed_count=sum(item.outcome == "regressed" for item in deltas),
        unchanged_count=sum(item.outcome == "unchanged" for item in deltas),
        deltas=deltas,
    )


def load_end_to_end_aggregate(path: str | Path) -> EndToEndAggregate:
    try:
        return EndToEndAggregate.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load end-to-end baseline {path}: {exc}") from exc


def run_end_to_end(
    suite: EvaluationSuite,
    invoke: GraphInvoker,
    *,
    config_name: str,
    retrieval_k_per_paper: int = 5,
    quote_token_recall_threshold: float = 0.8,
    recursion_limit: int = 30,
    baseline: EndToEndAggregate | None = None,
    run_id: str | None = None,
) -> EndToEndReport:
    generated_at = datetime.now(UTC)
    suite_fingerprint = hashlib.sha256(
        suite.model_dump_json(exclude_none=False).encode("utf-8")
    ).hexdigest()
    results = [
        evaluate_end_to_end_case(
            case,
            invoke,
            retrieval_k_per_paper=retrieval_k_per_paper,
            quote_token_recall_threshold=quote_token_recall_threshold,
            recursion_limit=recursion_limit,
        )
        for case in suite.cases
    ]
    aggregate = EndToEndAggregate(
        run_id=run_id or generated_at.strftime("e2e-%Y%m%dT%H%M%S.%fZ"),
        generated_at=generated_at,
        suite_id=suite.suite_id,
        dataset_version=suite.dataset_version,
        suite_fingerprint=suite_fingerprint,
        case_ids=[case.case_id for case in suite.cases],
        benchmark_status=suite.benchmark_status.value,
        config_name=config_name,
        retrieval_k_per_paper=retrieval_k_per_paper,
        quote_token_recall_threshold=quote_token_recall_threshold,
        case_count=len(results),
        completed_cases=sum(case.execution_error is None for case in results),
        execution_failures=sum(case.execution_error is not None for case in results),
        runtime=_runtime_metadata(),
        metrics=_aggregate_metrics(results),
        metric_directions=METRIC_DIRECTIONS,
    )
    if baseline is not None:
        aggregate.baseline_comparison = compare_with_baseline(aggregate, baseline)
    return EndToEndReport(aggregate=aggregate, cases=results)


def render_end_to_end_report(report: EndToEndReport) -> str:
    aggregate = report.aggregate

    def display(name: str) -> str:
        value = aggregate.metrics.get(name)
        if value is None:
            return "n/a"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = [
        "# End-to-end evaluation report",
        "",
        f"- Run: `{aggregate.run_id}`",
        f"- Suite: `{aggregate.suite_id}` (`{aggregate.dataset_version}`)",
        f"- Benchmark status: `{aggregate.benchmark_status}`",
        f"- Config: `{aggregate.config_name}`",
        f"- Cases: `{aggregate.case_count}`; execution failures: `{aggregate.execution_failures}`",
        f"- Decision accuracy: `{display('decision_accuracy')}`",
        f"- Answer F1: `{display('answer_f1')}`",
        f"- Retrieval Recall@K: `{display('retrieval_recall_at_k')}`",
        f"- Retrieval MRR: `{display('retrieval_mrr')}`",
        f"- Verifier-supported claim rate: `{display('verifier_supported_claim_rate')}`",
        f"- Citation-complete claim rate: `{display('citation_complete_claim_rate')}`",
        f"- Revision rate: `{display('revision_rate')}`",
        f"- Mean latency: `{display('mean_latency_seconds')}` seconds",
        f"- Counted LLM calls: `{display('llm_calls')}`",
        "- Embedding calls: `not instrumented`",
        "",
        "These numbers inherit the suite's publication status. Verifier-supported claim rate is",
        "the production verifier's judgment, not independently adjudicated entailment accuracy.",
        "",
        "## Cases",
        "",
        "| Case | Expected | Predicted | Correct | Recall | Claim status | Revisions | Failures |",
        "|---|---|---|---:|---:|---|---:|---|",
    ]
    for case in report.cases:
        recall = case.retrieval.recall_at_k
        failures = ", ".join(case.failure_reasons) or "—"
        lines.append(
            f"| {case.case_id} | {case.expected_decision} | {case.predicted_decision} | "
            f"{'yes' if case.decision_correct else 'no'} | "
            f"{'n/a' if recall is None else f'{recall:.4f}'} | "
            f"{case.claim_verification_status} | {case.claim_revision_count} | {failures} |"
        )
    if aggregate.baseline_comparison:
        comparison = aggregate.baseline_comparison
        lines.extend(
            [
                "",
                "## Baseline comparison",
                "",
                (
                    f"Baseline `{comparison.baseline_run_id}`: "
                    f"{comparison.improved_count} improved, "
                    f"{comparison.regressed_count} regressed, "
                    f"{comparison.unchanged_count} unchanged."
                ),
                "",
                "| Metric | Direction | Baseline | Current | Delta | Outcome |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for item in comparison.deltas:
            lines.append(
                f"| {item.metric} | {item.direction.value} | {item.baseline:.4f} | "
                f"{item.current:.4f} | {item.delta:+.4f} | {item.outcome} |"
            )
    return "\n".join(lines) + "\n"


def write_end_to_end_outputs(report: EndToEndReport, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination / "metrics.json").write_text(
        json.dumps(report.aggregate.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination / "per_case.jsonl").write_text(
        "".join(
            json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for case in report.cases
        ),
        encoding="utf-8",
    )
    (destination / "report.md").write_text(
        render_end_to_end_report(report), encoding="utf-8"
    )
