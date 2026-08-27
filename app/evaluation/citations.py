"""Deterministic claim-to-evidence citation safety metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.evaluation.models import StrictModel

CITATION_SAFETY_CONTRACT_VERSION = "1.0.0"


class CitationClaimRecord(StrictModel):
    claim_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    text: str = Field(min_length=1)
    requires_citation: bool
    cited_evidence_ids: list[str]
    supporting_evidence_ids: list[str]

    @field_validator("cited_evidence_ids", "supporting_evidence_ids")
    @classmethod
    def evidence_ids_are_unique(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Evidence IDs must not be blank.")
        if len(value) != len(set(value)):
            raise ValueError("Evidence IDs within a claim must be unique.")
        return value


class CitationSafetyCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    available_evidence_ids: list[str]
    claims: list[CitationClaimRecord] = Field(min_length=1)

    @field_validator("available_evidence_ids")
    @classmethod
    def available_evidence_is_well_formed(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Available evidence IDs must not be blank.")
        return value

    @model_validator(mode="after")
    def references_are_valid(self) -> CitationSafetyCase:
        if len(self.available_evidence_ids) != len(set(self.available_evidence_ids)):
            raise ValueError("available_evidence_ids must be unique.")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique within a case.")
        available = set(self.available_evidence_ids)
        for claim in self.claims:
            unknown_support = set(claim.supporting_evidence_ids) - available
            if unknown_support:
                raise ValueError(
                    f"Claim {claim.claim_id} has unknown supporting evidence IDs: "
                    f"{sorted(unknown_support)}"
                )
        return self


class CitationSafetySuite(StrictModel):
    contract_version: str
    suite_id: str
    dataset_version: str
    benchmark_status: Literal["fixture", "development"]
    description: str
    cases: list[CitationSafetyCase] = Field(min_length=1)

    @model_validator(mode="after")
    def contract_and_cases_are_valid(self) -> CitationSafetySuite:
        if self.contract_version != CITATION_SAFETY_CONTRACT_VERSION:
            raise ValueError(
                "Unsupported citation contract_version; expected "
                f"{CITATION_SAFETY_CONTRACT_VERSION}."
            )
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Citation safety case_id values must be unique.")
        return self


class CitationMetricCounts(StrictModel):
    claim_count: int
    citation_required_claims: int
    claims_with_valid_citation: int
    claims_with_supporting_citation: int
    citation_assignments: int
    supporting_citation_assignments: int
    invalid_citation_assignments: int

    def add(self, other: CitationMetricCounts) -> CitationMetricCounts:
        return CitationMetricCounts(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in type(self).model_fields
            }
        )


class CitationCaseResult(StrictModel):
    case_id: str
    metrics: dict[str, float | int | None]


class CitationSafetyReport(StrictModel):
    contract_version: str = CITATION_SAFETY_CONTRACT_VERSION
    suite_id: str
    dataset_version: str
    benchmark_status: str
    case_count: int
    metrics: dict[str, float | int | None]
    results: list[CitationCaseResult]


def load_citation_safety_suite(path: str | Path) -> CitationSafetySuite:
    try:
        return CitationSafetySuite.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load citation safety suite {path}: {exc}") from exc


def _counts(case: CitationSafetyCase) -> CitationMetricCounts:
    available = set(case.available_evidence_ids)
    required = [claim for claim in case.claims if claim.requires_citation]
    citation_assignments = 0
    supporting_assignments = 0
    invalid_assignments = 0
    claims_with_valid = 0
    claims_with_support = 0
    for claim in case.claims:
        cited = set(claim.cited_evidence_ids)
        supporting = set(claim.supporting_evidence_ids)
        valid = cited & available
        supported = cited & supporting
        citation_assignments += len(cited)
        supporting_assignments += len(supported)
        invalid_assignments += len(cited - available)
        if claim.requires_citation:
            claims_with_valid += bool(valid)
            claims_with_support += bool(supported)
    return CitationMetricCounts(
        claim_count=len(case.claims),
        citation_required_claims=len(required),
        claims_with_valid_citation=claims_with_valid,
        claims_with_supporting_citation=claims_with_support,
        citation_assignments=citation_assignments,
        supporting_citation_assignments=supporting_assignments,
        invalid_citation_assignments=invalid_assignments,
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(counts: CitationMetricCounts) -> dict[str, float | int | None]:
    return {
        "citation_precision": _ratio(
            counts.supporting_citation_assignments, counts.citation_assignments
        ),
        "citation_completeness": _ratio(
            counts.claims_with_valid_citation, counts.citation_required_claims
        ),
        "unsupported_claim_rate": _ratio(
            counts.citation_required_claims - counts.claims_with_supporting_citation,
            counts.citation_required_claims,
        ),
        "invalid_citation_rate": _ratio(
            counts.invalid_citation_assignments, counts.citation_assignments
        ),
        **counts.model_dump(),
    }


def evaluate_citation_safety(suite: CitationSafetySuite) -> CitationSafetyReport:
    total = CitationMetricCounts(
        claim_count=0,
        citation_required_claims=0,
        claims_with_valid_citation=0,
        claims_with_supporting_citation=0,
        citation_assignments=0,
        supporting_citation_assignments=0,
        invalid_citation_assignments=0,
    )
    results = []
    for case in suite.cases:
        counts = _counts(case)
        total = total.add(counts)
        results.append(CitationCaseResult(case_id=case.case_id, metrics=_metrics(counts)))
    return CitationSafetyReport(
        suite_id=suite.suite_id,
        dataset_version=suite.dataset_version,
        benchmark_status=suite.benchmark_status,
        case_count=len(suite.cases),
        metrics=_metrics(total),
        results=results,
    )


def render_citation_report(report: CitationSafetyReport) -> str:
    def display(name: str) -> str:
        value = report.metrics[name]
        return "not applicable" if value is None else f"{value:.4f}"

    lines = [
        "# Citation safety report",
        "",
        f"- Suite: `{report.suite_id}`",
        f"- Status: `{report.benchmark_status}`",
        f"- Citation precision: `{display('citation_precision')}`",
        f"- Citation completeness: `{display('citation_completeness')}`",
        f"- Unsupported claim rate: `{display('unsupported_claim_rate')}`",
        f"- Invalid citation rate: `{display('invalid_citation_rate')}`",
        "",
        "| Case | Precision | Completeness | Unsupported claims | Invalid citations |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in report.results:
        values = []
        for name in (
            "citation_precision",
            "citation_completeness",
            "unsupported_claim_rate",
            "invalid_citation_rate",
        ):
            value = case.metrics[name]
            values.append("n/a" if value is None else f"{value:.4f}")
        lines.append(f"| {case.case_id} | {' | '.join(values)} |")
    return "\n".join(lines) + "\n"


def write_citation_report(report: CitationSafetyReport, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "citation_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination / "citation_report.md").write_text(
        render_citation_report(report), encoding="utf-8"
    )
