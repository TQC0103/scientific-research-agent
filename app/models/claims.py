"""Structured atomic-claim and claim-to-evidence verification contract."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

CLAIM_VERIFICATION_CONTRACT_VERSION = "1.0.0"
_CITATION_PATTERN = re.compile(r"\[(\d+)]")
CitationLabel = Annotated[int, Field(ge=1)]


def citation_labels_in_text(text: str) -> list[int]:
    """Return unique visible numeric labels in first-appearance order."""
    labels: list[int] = []
    for match in _CITATION_PATTERN.finditer(text):
        label = int(match.group(1))
        if label not in labels:
            labels.append(label)
    return labels


class ClaimModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceRelationship(StrEnum):
    ENTAILS = "entails"
    PARTIAL = "partial"
    DOES_NOT_SUPPORT = "does_not_support"


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    NOT_REQUIRED = "not_required"


class AtomicClaim(ClaimModel):
    claim_id: str = Field(pattern=r"^claim_[1-9][0-9]*$")
    claim_text: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    requires_citation: bool
    citation_labels: list[CitationLabel]

    @model_validator(mode="after")
    def citations_are_unique(self) -> AtomicClaim:
        if len(self.citation_labels) != len(set(self.citation_labels)):
            raise ValueError("citation_labels must be unique within a claim.")
        return self


class ClaimEvidenceLink(ClaimModel):
    citation_label: CitationLabel
    relationship: EvidenceRelationship
    reason: str = Field(min_length=1)


class ClaimAssessment(ClaimModel):
    claim_id: str = Field(pattern=r"^claim_[1-9][0-9]*$")
    verdict: ClaimVerdict
    cited_evidence: list[ClaimEvidenceLink]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_labels_are_unique(self) -> ClaimAssessment:
        labels = [item.citation_label for item in self.cited_evidence]
        if len(labels) != len(set(labels)):
            raise ValueError("Each citation label may be assessed only once per claim.")
        return self


class ClaimVerificationBundle(ClaimModel):
    contract_version: str
    answer: str = Field(min_length=1)
    evidence_count: int = Field(ge=0)
    claims: list[AtomicClaim] = Field(min_length=1)
    assessments: list[ClaimAssessment] = Field(min_length=1)

    @model_validator(mode="after")
    def cross_references_and_verdicts_are_consistent(self) -> ClaimVerificationBundle:
        if self.contract_version != CLAIM_VERIFICATION_CONTRACT_VERSION:
            raise ValueError(
                "Unsupported claim contract_version; expected "
                f"{CLAIM_VERIFICATION_CONTRACT_VERSION}."
            )
        claim_ids = [claim.claim_id for claim in self.claims]
        assessment_ids = [assessment.claim_id for assessment in self.assessments]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique.")
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("Assessment claim_id values must be unique.")
        if set(claim_ids) != set(assessment_ids):
            raise ValueError("Every atomic claim requires exactly one assessment.")
        expected_claim_ids = [f"claim_{number}" for number in range(1, len(self.claims) + 1)]
        if claim_ids != expected_claim_ids:
            raise ValueError("Claims must use sequential IDs in answer order.")
        if assessment_ids != claim_ids:
            raise ValueError("Assessments must follow claim order.")

        assessments = {item.claim_id: item for item in self.assessments}
        source_positions = []
        for claim in self.claims:
            if claim.source_text not in self.answer:
                raise ValueError(f"Claim {claim.claim_id} source_text is not in the answer.")
            source_positions.append(self.answer.find(claim.source_text))
            source_labels = citation_labels_in_text(claim.source_text)
            if source_labels != claim.citation_labels:
                raise ValueError(
                    f"Claim {claim.claim_id} citation_labels must match its source_text."
                )
            if any(label > self.evidence_count for label in claim.citation_labels):
                raise ValueError(
                    f"Claim {claim.claim_id} cites evidence outside the supplied set."
                )

            assessment = assessments[claim.claim_id]
            assessed_labels = [item.citation_label for item in assessment.cited_evidence]
            if assessed_labels != claim.citation_labels:
                raise ValueError(
                    f"Claim {claim.claim_id} must assess every cited label in source order."
                )
            expected = expected_claim_verdict(claim, assessment.cited_evidence)
            if assessment.verdict != expected:
                raise ValueError(
                    f"Claim {claim.claim_id} verdict {assessment.verdict} conflicts with "
                    f"evidence relationships; expected {expected}."
                )
        if source_positions != sorted(source_positions):
            raise ValueError("Claims must follow their source spans in answer order.")
        return self


def expected_claim_verdict(
    claim: AtomicClaim, evidence: list[ClaimEvidenceLink]
) -> ClaimVerdict:
    if not claim.requires_citation:
        if claim.citation_labels or evidence:
            raise ValueError("Claims that do not require citation cannot carry evidence labels.")
        return ClaimVerdict.NOT_REQUIRED
    relationships = {item.relationship for item in evidence}
    if EvidenceRelationship.ENTAILS in relationships:
        return ClaimVerdict.SUPPORTED
    if EvidenceRelationship.PARTIAL in relationships:
        return ClaimVerdict.PARTIAL
    return ClaimVerdict.UNSUPPORTED


def load_claim_verification(path: str | Path) -> ClaimVerificationBundle:
    try:
        return ClaimVerificationBundle.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load claim verification bundle {path}: {exc}") from exc


def claim_verification_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            "https://github.com/TQC0103/scientific-research-agent/"
            "evaluation/schema/claim-verification.schema.json"
        ),
        **ClaimVerificationBundle.model_json_schema(),
    }
