"""Advisory LLM review for repo-authored evaluation cases.

Judge output is deliberately separate from the evaluation source artifact. It
can find likely authoring defects, but it is not an independent human review and
cannot make a development suite publishable.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from enum import StrEnum

from pydantic import Field, model_validator

from app.evaluation.models import EvaluationCase, EvaluationSuite, ExpectedDecision, StrictModel

JUDGE_CONTRACT_VERSION = "1.0.0"


class JudgeVerdict(StrEnum):
    PASS = "pass"
    NEEDS_REVISION = "needs_revision"
    FAIL = "fail"


class JudgeScores(StrictModel):
    question_clarity: int = Field(ge=1, le=5)
    evidence_entailment: int = Field(ge=1, le=5)
    answer_alignment: int = Field(ge=1, le=5)
    citation_specificity: int = Field(ge=1, le=5)
    challenge_validity: int = Field(ge=1, le=5)


class JudgeFinding(StrictModel):
    severity: str = Field(pattern=r"^(info|warning|error)$")
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)


class JudgeResult(StrictModel):
    contract_version: str
    case_id: str
    verdict: JudgeVerdict
    scores: JudgeScores
    findings: list[JudgeFinding]
    rationale: str = Field(min_length=1)
    human_review_required: bool

    @model_validator(mode="after")
    def contract_is_supported(self) -> JudgeResult:
        if self.contract_version != JUDGE_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported judge contract_version; expected {JUDGE_CONTRACT_VERSION}."
            )
        return self


class JudgeReport(StrictModel):
    contract_version: str
    suite_id: str
    dataset_version: str
    judge_model: str
    advisory_only: bool = True
    case_count: int
    model_calls: int
    verdict_counts: dict[str, int]
    mean_scores: dict[str, float]
    human_review_required_count: int
    results: list[JudgeResult]


SYSTEM_PROMPT = """You are auditing a small scientific QA evaluation dataset.
Use only the supplied case record. Find annotation defects; do not answer the
research question from memory. An exact quote may establish only claims it
actually states. For abstention cases without gold evidence, you may assess the
coherence of the negative-case design but cannot prove document-wide absence.
Return exactly one JSON object matching the requested schema, with no markdown.
This review is advisory and never counts as independent human adjudication."""


def build_judge_prompt(case: EvaluationCase) -> str:
    payload = {
        "case_id": case.case_id,
        "question": case.question,
        "question_type": case.question_type,
        "papers": [paper.model_dump(mode="json") for paper in case.papers],
        "expected": case.expected.model_dump(mode="json"),
        "gold_evidence": [item.model_dump(mode="json") for item in case.gold_evidence],
        "challenge": case.challenge.model_dump(mode="json") if case.challenge else None,
    }
    schema = {
        "contract_version": JUDGE_CONTRACT_VERSION,
        "case_id": case.case_id,
        "verdict": "pass | needs_revision | fail",
        "scores": {
            "question_clarity": "integer 1..5",
            "evidence_entailment": "integer 1..5",
            "answer_alignment": "integer 1..5",
            "citation_specificity": "integer 1..5",
            "challenge_validity": "integer 1..5",
        },
        "findings": [
            {"severity": "info | warning | error", "category": "string", "message": "string"}
        ],
        "rationale": "short string",
        "human_review_required": "boolean",
    }
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Score whether the question is unambiguous, each cited quote entails its listed "
        "criteria, the reference answer stays within evidence, the citation location is "
        "specific, and any challenge/abstention label is coherent. Penalize invented "
        "causal or numerical claims. Every abstention case must set human_review_required "
        "to true.\n\nCASE:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"OUTPUT SCHEMA:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def _first_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Judge response did not contain a valid JSON object.")


def parse_judge_response(case: EvaluationCase, response: str) -> JudgeResult:
    result = JudgeResult.model_validate(_first_json_object(response))
    if result.case_id != case.case_id:
        raise ValueError(
            f"Judge returned case_id {result.case_id!r}; expected {case.case_id!r}."
        )
    if case.expected.decision == ExpectedDecision.ABSTAIN and not result.human_review_required:
        result = result.model_copy(update={"human_review_required": True})
    return result


def build_judge_report(
    suite: EvaluationSuite,
    judge_model: str,
    results: Iterable[JudgeResult],
) -> JudgeReport:
    materialized = list(results)
    expected_ids = [case.case_id for case in suite.cases]
    result_ids = [result.case_id for result in materialized]
    if Counter(result_ids) != Counter(expected_ids):
        raise ValueError("Judge results must contain every suite case exactly once.")

    score_fields = tuple(JudgeScores.model_fields)
    means = {
        field: round(
            sum(getattr(result.scores, field) for result in materialized) / len(materialized),
            4,
        )
        for field in score_fields
    }
    verdict_counts = Counter(result.verdict.value for result in materialized)
    return JudgeReport(
        contract_version=JUDGE_CONTRACT_VERSION,
        suite_id=suite.suite_id,
        dataset_version=suite.dataset_version,
        judge_model=judge_model,
        case_count=len(materialized),
        model_calls=len(materialized),
        verdict_counts=dict(sorted(verdict_counts.items())),
        mean_scores=means,
        human_review_required_count=sum(item.human_review_required for item in materialized),
        results=materialized,
    )
