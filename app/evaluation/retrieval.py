"""Gold-evidence retrieval evaluation for the internal suite.

Matching is revision-safe and annotation-relative: an unlabeled retrieved chunk
is not asserted to be irrelevant, but it does not count toward Precision@K.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from app.evaluation.models import EvaluationCase, EvaluationSuite, GoldEvidence, StrictModel

RETRIEVAL_REPORT_VERSION = "1.0.0"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class RetrievedChunk(StrictModel):
    arxiv_id: str | None = None
    versioned_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    chunk_index: int | None = Field(default=None, ge=0)
    score: float | None = None
    retrieval_score: float | None = None
    fusion_method: str | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    reranker_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    lexical_rank: int | None = Field(default=None, ge=1)


class RetrievalInputRow(StrictModel):
    case_id: str = Field(min_length=1)
    retrieved: list[RetrievedChunk]


class RetrievalCaseMetrics(StrictModel):
    case_id: str
    eligible: bool
    top_k: int
    retrieved_count: int
    gold_evidence_count: int
    gold_group_count: int
    matched_evidence_ids: list[str]
    matched_evidence_group_ids: list[str]
    first_match_rank: int | None
    recall_at_k: float | None
    precision_at_k: float | None
    reciprocal_rank: float | None
    gold_evidence_coverage: float | None
    required_paper_coverage: float | None
    macro_paper_recall: float | None
    missing_required_paper_ids: list[str]
    unmatched_evidence_group_ids: list[str]
    diagnostics: list[str]


class RetrievalAggregate(StrictModel):
    contract_version: str = RETRIEVAL_REPORT_VERSION
    suite_id: str
    dataset_version: str
    config_name: str
    top_k: int
    quote_token_recall_threshold: float
    case_count: int
    eligible_cases: int
    ineligible_cases: int
    missing_case_predictions: int
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    gold_evidence_coverage: float | None
    required_paper_coverage: float | None
    macro_paper_recall: float | None


class RetrievalReport(StrictModel):
    aggregate: RetrievalAggregate
    cases: list[RetrievalCaseMetrics]

    @model_validator(mode="after")
    def case_count_agrees(self) -> RetrievalReport:
        if self.aggregate.case_count != len(self.cases):
            raise ValueError("Aggregate case_count must match the per-case results.")
        return self


def _normalized_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return TOKEN_PATTERN.findall(normalized)


def quote_token_recall(quote: str, chunk_text: str) -> float:
    """Return multiset token recall of a gold quote within a retrieved chunk."""
    gold = Counter(_normalized_tokens(quote))
    if not gold:
        return 0.0
    retrieved = Counter(_normalized_tokens(chunk_text))
    return sum((gold & retrieved).values()) / sum(gold.values())


def evidence_matches(
    evidence: GoldEvidence,
    chunk: RetrievedChunk,
    *,
    quote_token_recall_threshold: float = 0.8,
) -> bool:
    """Match evidence by pinned revision and normalized quote content.

    Page equality is deliberately not sufficient. It remains available in the
    input/output for diagnosis when PDF extraction shifts text boundaries.
    """
    if evidence.versioned_id != chunk.versioned_id:
        return False
    gold_tokens = _normalized_tokens(evidence.quote)
    chunk_tokens = _normalized_tokens(chunk.text)
    if not gold_tokens or not chunk_tokens:
        return False
    normalized_gold = " ".join(gold_tokens)
    normalized_chunk = " ".join(chunk_tokens)
    if normalized_gold in normalized_chunk:
        return True
    return quote_token_recall(evidence.quote, chunk.text) >= quote_token_recall_threshold


def evaluate_case_retrieval(
    case: EvaluationCase,
    retrieved: Sequence[RetrievedChunk | Mapping[str, Any]],
    *,
    top_k: int = 5,
    quote_token_recall_threshold: float = 0.8,
) -> RetrievalCaseMetrics:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0 < quote_token_recall_threshold <= 1:
        raise ValueError("quote_token_recall_threshold must be in (0, 1].")

    ranked = [
        item if isinstance(item, RetrievedChunk) else RetrievedChunk.model_validate(item)
        for item in retrieved[:top_k]
    ]
    gold = case.gold_evidence
    if not gold:
        return RetrievalCaseMetrics(
            case_id=case.case_id,
            eligible=False,
            top_k=top_k,
            retrieved_count=len(ranked),
            gold_evidence_count=0,
            gold_group_count=0,
            matched_evidence_ids=[],
            matched_evidence_group_ids=[],
            first_match_rank=None,
            recall_at_k=None,
            precision_at_k=None,
            reciprocal_rank=None,
            gold_evidence_coverage=None,
            required_paper_coverage=None,
            macro_paper_recall=None,
            missing_required_paper_ids=[],
            unmatched_evidence_group_ids=[],
            diagnostics=["not_applicable_no_gold_evidence"],
        )

    matches_by_rank: dict[int, list[GoldEvidence]] = {}
    matched_evidence: dict[str, GoldEvidence] = {}
    matched_groups: set[str] = set()
    for rank, chunk in enumerate(ranked, 1):
        matches = [
            item
            for item in gold
            if evidence_matches(
                item,
                chunk,
                quote_token_recall_threshold=quote_token_recall_threshold,
            )
        ]
        if matches:
            matches_by_rank[rank] = matches
            for item in matches:
                matched_evidence[item.evidence_id] = item
                matched_groups.add(item.evidence_group_id)

    all_groups = {item.evidence_group_id for item in gold}
    first_match_rank = min(matches_by_rank, default=None)
    relevant_retrieved = len(matches_by_rank)

    paper_groups: dict[str, set[str]] = defaultdict(set)
    matched_paper_groups: dict[str, set[str]] = defaultdict(set)
    for item in gold:
        paper_groups[item.versioned_id].add(item.evidence_group_id)
    for item in matched_evidence.values():
        matched_paper_groups[item.versioned_id].add(item.evidence_group_id)

    required_papers = case.expected.required_paper_ids
    covered_required = {
        paper_id for paper_id in required_papers if matched_paper_groups.get(paper_id)
    }
    missing_required = sorted(set(required_papers) - covered_required)
    paper_recalls = [
        len(matched_paper_groups.get(paper_id, set())) / len(groups)
        for paper_id, groups in sorted(paper_groups.items())
    ]

    diagnostics = []
    unmatched_groups = sorted(all_groups - matched_groups)
    if not matches_by_rank:
        diagnostics.append("no_gold_evidence_retrieved")
    elif unmatched_groups:
        diagnostics.append("incomplete_gold_group_coverage")
    if missing_required:
        diagnostics.append("missing_required_paper_coverage")
    if len(ranked) < top_k:
        diagnostics.append("retriever_returned_fewer_than_k")

    return RetrievalCaseMetrics(
        case_id=case.case_id,
        eligible=True,
        top_k=top_k,
        retrieved_count=len(ranked),
        gold_evidence_count=len(gold),
        gold_group_count=len(all_groups),
        matched_evidence_ids=sorted(matched_evidence),
        matched_evidence_group_ids=sorted(matched_groups),
        first_match_rank=first_match_rank,
        recall_at_k=len(matched_groups) / len(all_groups),
        precision_at_k=relevant_retrieved / top_k,
        reciprocal_rank=1 / first_match_rank if first_match_rank else 0.0,
        gold_evidence_coverage=len(matched_evidence) / len(gold),
        required_paper_coverage=(
            len(covered_required) / len(required_papers) if required_papers else None
        ),
        macro_paper_recall=sum(paper_recalls) / len(paper_recalls),
        missing_required_paper_ids=missing_required,
        unmatched_evidence_group_ids=unmatched_groups,
        diagnostics=diagnostics,
    )


def evaluate_retrieval(
    suite: EvaluationSuite,
    retrievals: Mapping[str, Sequence[RetrievedChunk | Mapping[str, Any]]],
    *,
    config_name: str,
    top_k: int = 5,
    quote_token_recall_threshold: float = 0.8,
) -> RetrievalReport:
    known_case_ids = {case.case_id for case in suite.cases}
    unknown_case_ids = set(retrievals) - known_case_ids
    if unknown_case_ids:
        raise ValueError(f"Retrieval input contains unknown cases: {sorted(unknown_case_ids)}")

    results = [
        evaluate_case_retrieval(
            case,
            retrievals.get(case.case_id, []),
            top_k=top_k,
            quote_token_recall_threshold=quote_token_recall_threshold,
        )
        for case in suite.cases
    ]
    eligible = [result for result in results if result.eligible]

    def mean(field: str) -> float | None:
        values = [getattr(result, field) for result in eligible]
        numeric = [value for value in values if value is not None]
        return sum(numeric) / len(numeric) if numeric else None

    missing = sum(case.case_id not in retrievals for case in suite.cases)
    aggregate = RetrievalAggregate(
        suite_id=suite.suite_id,
        dataset_version=suite.dataset_version,
        config_name=config_name,
        top_k=top_k,
        quote_token_recall_threshold=quote_token_recall_threshold,
        case_count=len(results),
        eligible_cases=len(eligible),
        ineligible_cases=len(results) - len(eligible),
        missing_case_predictions=missing,
        recall_at_k=mean("recall_at_k"),
        precision_at_k=mean("precision_at_k"),
        mrr=mean("reciprocal_rank"),
        gold_evidence_coverage=mean("gold_evidence_coverage"),
        required_paper_coverage=mean("required_paper_coverage"),
        macro_paper_recall=mean("macro_paper_recall"),
    )
    return RetrievalReport(aggregate=aggregate, cases=results)


def load_retrieval_jsonl(path: str | Path) -> dict[str, list[RetrievedChunk]]:
    source = Path(path)
    rows: dict[str, list[RetrievedChunk]] = {}
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Could not read retrieval input {source}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = RetrievalInputRow.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid retrieval row at line {line_number}: {exc}") from exc
        if row.case_id in rows:
            raise ValueError(f"Duplicate retrieval case_id at line {line_number}: {row.case_id}")
        rows[row.case_id] = row.retrieved
    return rows
