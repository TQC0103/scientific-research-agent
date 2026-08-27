import re
from datetime import datetime
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

ARXIV_ID_PATTERN = re.compile(r"^[0-9]{4}\.[0-9]{4,5}$")
VERSIONED_ARXIV_ID_PATTERN = re.compile(r"^([0-9]{4}\.[0-9]{4,5})v([1-9][0-9]*)$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkStatus(StrEnum):
    SCHEMA_FIXTURE = "schema_fixture"
    DEVELOPMENT = "development"
    FROZEN = "frozen"


class EvaluationSplit(StrEnum):
    FIXTURE = "fixture"
    DEVELOPMENT = "development"
    TEST = "test"


class QuestionType(StrEnum):
    SINGLE_PAPER_FACT = "single_paper_fact"
    METHOD = "method"
    RESULT = "result"
    MULTI_PAPER_COMPARISON = "multi_paper_comparison"
    EVIDENCE_MISSING = "evidence_missing"
    UNSUPPORTED_QUESTION = "unsupported_question"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    PARTIAL_EVIDENCE = "partial_evidence"


class ProvenanceKind(StrEnum):
    EXTERNAL = "external"
    REPO_CURATED = "repo_curated"
    SYNTHETIC = "synthetic"


class Provenance(StrictModel):
    kind: ProvenanceKind
    dataset_name: str = Field(min_length=1)
    dataset_version: str | None = None
    source_url: str | None = None
    source_case_id: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    license: str = Field(min_length=1)
    adaptation_notes: str

    @model_validator(mode="after")
    def require_external_url(self) -> "Provenance":
        if self.kind == ProvenanceKind.EXTERNAL:
            parsed = urlparse(self.source_url or "")
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("External provenance requires an HTTP(S) source_url.")
        return self


class AnnotationSource(StrEnum):
    EXTERNAL_DATASET = "external_dataset"
    REPO_AUTHORED = "repo_authored"
    SYNTHETIC = "synthetic"


class Annotation(StrictModel):
    source: AnnotationSource
    annotator_count: int | None = Field(default=None, ge=0)
    reviewer_count: int | None = Field(default=None, ge=0)
    adjudicated: bool | None = None
    notes: str


class Paper(StrictModel):
    paper_id: str
    versioned_id: str
    revision: int = Field(ge=1)
    title: str = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_agree(self) -> "Paper":
        if not ARXIV_ID_PATTERN.fullmatch(self.paper_id):
            raise ValueError(f"Invalid base arXiv ID: {self.paper_id}")
        match = VERSIONED_ARXIV_ID_PATTERN.fullmatch(self.versioned_id)
        if not match or match.group(1) != self.paper_id or int(match.group(2)) != self.revision:
            raise ValueError("paper_id, versioned_id, and revision must identify one revision.")
        return self


class AnswerCriterion(StrictModel):
    criterion_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    required: bool


class ExpectedDecision(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


class AbstentionReason(StrEnum):
    EVIDENCE_MISSING = "evidence_missing"
    UNSUPPORTED_QUESTION = "unsupported_question"
    UNRESOLVED_CONFLICT = "unresolved_conflict"


class Expected(StrictModel):
    decision: ExpectedDecision
    abstention_reason: AbstentionReason | None = None
    required_paper_ids: list[str]
    reference_answer: str | None = Field(default=None, min_length=1)
    answer_criteria: list[AnswerCriterion]
    forbidden_claims: list[str]

    @model_validator(mode="after")
    def decision_fields_agree(self) -> "Expected":
        if self.decision == ExpectedDecision.ANSWER:
            if self.abstention_reason is not None:
                raise ValueError("Answer cases cannot define abstention_reason.")
            if not self.reference_answer or not self.answer_criteria or not self.required_paper_ids:
                raise ValueError(
                    "Answer cases require reference_answer, answer_criteria, and required papers."
                )
        elif self.abstention_reason is None:
            raise ValueError("Abstention cases require abstention_reason.")
        return self


class SourceType(StrEnum):
    FULL_TEXT = "full_text"
    ABSTRACT = "abstract"


class GoldEvidence(StrictModel):
    evidence_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    evidence_group_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    paper_id: str
    versioned_id: str
    source_type: SourceType
    page: int | None = Field(default=None, ge=1)
    section: str | None
    quote: str = Field(min_length=20)
    chunk_index: int | None = Field(default=None, ge=0)
    supports: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def location_is_valid(self) -> "GoldEvidence":
        if self.source_type == SourceType.FULL_TEXT and self.page is None:
            raise ValueError("Full-text gold evidence requires a page.")
        if self.source_type == SourceType.ABSTRACT and self.page is not None:
            raise ValueError("Abstract gold evidence cannot invent a page.")
        return self


class ChallengeKind(StrEnum):
    NEGATIVE = "negative"
    ADVERSARIAL = "adversarial"
    EVIDENCE_MISSING = "evidence_missing"
    UNSUPPORTED_PREMISE = "unsupported_premise"
    PARTIAL_EVIDENCE = "partial_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    CROSS_PAPER_LEAKAGE = "cross_paper_leakage"


class Challenge(StrictModel):
    kinds: list[ChallengeKind] = Field(min_length=1)
    description: str = Field(min_length=1)


class EvaluationCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=1)
    question_type: QuestionType
    evaluation_split: EvaluationSplit
    provenance: Provenance
    annotation: Annotation
    papers: list[Paper] = Field(min_length=1)
    expected: Expected
    gold_evidence: list[GoldEvidence]
    challenge: Challenge | None = None
    tags: list[str]
    notes: str | None = None

    @model_validator(mode="after")
    def validate_cross_references(self) -> "EvaluationCase":
        paper_ids = [paper.paper_id for paper in self.papers]
        versioned_ids = [paper.versioned_id for paper in self.papers]
        if len(set(paper_ids)) != len(paper_ids) or len(set(versioned_ids)) != len(versioned_ids):
            raise ValueError("Paper identifiers must be unique within a case.")
        missing_required = set(self.expected.required_paper_ids) - set(versioned_ids)
        if missing_required:
            raise ValueError(f"Required papers are not declared: {sorted(missing_required)}")

        criterion_ids = [criterion.criterion_id for criterion in self.expected.answer_criteria]
        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("criterion_id values must be unique within a case.")
        evidence_ids = [item.evidence_id for item in self.gold_evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence_id values must be unique within a case.")
        for item in self.gold_evidence:
            if item.paper_id not in paper_ids or item.versioned_id not in versioned_ids:
                raise ValueError(f"Gold evidence {item.evidence_id} uses an undeclared paper.")
            unsupported = set(item.supports) - set(criterion_ids)
            if unsupported:
                raise ValueError(
                    f"Gold evidence {item.evidence_id} references unknown criteria: "
                    f"{sorted(unsupported)}"
                )
        if self.expected.decision == ExpectedDecision.ANSWER and not self.gold_evidence:
            raise ValueError("Answer cases require at least one gold evidence item.")
        return self


class EvaluationSuite(StrictModel):
    schema_version: str
    suite_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    dataset_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    benchmark_status: BenchmarkStatus
    frozen_at: datetime | None = None
    description: str = Field(min_length=1)
    cases: list[EvaluationCase]

    @model_validator(mode="after")
    def validate_suite(self) -> "EvaluationSuite":
        if self.schema_version != "1.1.0":
            raise ValueError("Unsupported schema_version; expected 1.1.0.")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_id values must be unique within a suite.")
        if self.benchmark_status == BenchmarkStatus.FROZEN and self.frozen_at is None:
            raise ValueError("Frozen suites require frozen_at.")
        if self.benchmark_status == BenchmarkStatus.SCHEMA_FIXTURE and any(
            case.evaluation_split != EvaluationSplit.FIXTURE for case in self.cases
        ):
            raise ValueError("Schema fixture suites may contain only fixture cases.")
        return self

    def assert_publishable(self) -> None:
        """Reject suites that must not be presented as held-out benchmark results."""
        if self.benchmark_status != BenchmarkStatus.FROZEN or self.frozen_at is None:
            raise ValueError("Only a frozen suite with frozen_at is publishable.")
        test_cases = [case for case in self.cases if case.evaluation_split == EvaluationSplit.TEST]
        if not test_cases:
            raise ValueError("A publishable suite requires at least one test case.")
        for case in test_cases:
            if case.provenance.kind == ProvenanceKind.SYNTHETIC:
                raise ValueError(f"Synthetic test case {case.case_id} is not publishable.")
            if case.provenance.kind == ProvenanceKind.REPO_CURATED and (
                not case.annotation.adjudicated or (case.annotation.reviewer_count or 0) < 1
            ):
                raise ValueError(
                    f"Repo-curated test case {case.case_id} requires independent review "
                    "and adjudication."
                )
