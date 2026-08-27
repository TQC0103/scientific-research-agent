import json
from pathlib import Path

import pytest

from app.evaluation.citations import (
    CitationSafetySuite,
    citation_case_from_claim_verification,
    evaluate_citation_safety,
)
from app.models.claims import (
    ClaimVerdict,
    ClaimVerificationBundle,
    claim_verification_json_schema,
    load_claim_verification,
)

FIXTURE = Path("evaluation/suites/v0_5/claim_verification_fixtures.json")


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_locks_all_task_7_verdict_shapes() -> None:
    bundle = load_claim_verification(FIXTURE)

    assert bundle.contract_version == "1.0.0"
    assert len(bundle.claims) == 6
    assert [assessment.verdict for assessment in bundle.assessments] == [
        ClaimVerdict.SUPPORTED,
        ClaimVerdict.PARTIAL,
        ClaimVerdict.UNSUPPORTED,
        ClaimVerdict.UNSUPPORTED,
        ClaimVerdict.NOT_REQUIRED,
        ClaimVerdict.UNSUPPORTED,
    ]


def test_claim_source_must_be_traceable_to_answer() -> None:
    payload = _payload()
    payload["claims"][0]["source_text"] = "Text invented during claim extraction."

    with pytest.raises(ValueError, match="source_text is not in the answer"):
        ClaimVerificationBundle.model_validate(payload)


def test_claim_citation_labels_must_match_visible_source_labels() -> None:
    payload = _payload()
    payload["claims"][0]["citation_labels"] = [2]

    with pytest.raises(ValueError, match="must match its source_text"):
        ClaimVerificationBundle.model_validate(payload)


def test_claim_cannot_reference_evidence_outside_supplied_set() -> None:
    payload = _payload()
    payload["evidence_count"] = 3

    with pytest.raises(ValueError, match="outside the supplied set"):
        ClaimVerificationBundle.model_validate(payload)


def test_every_claim_requires_exactly_one_assessment() -> None:
    payload = _payload()
    payload["assessments"].pop()

    with pytest.raises(ValueError, match="exactly one assessment"):
        ClaimVerificationBundle.model_validate(payload)


def test_claims_must_keep_sequential_ids_and_answer_order() -> None:
    payload = _payload()
    payload["claims"][0], payload["claims"][1] = payload["claims"][1], payload["claims"][0]

    with pytest.raises(ValueError, match="sequential IDs in answer order"):
        ClaimVerificationBundle.model_validate(payload)


def test_assessment_must_cover_every_claim_citation_in_source_order() -> None:
    payload = _payload()
    payload["assessments"][0]["cited_evidence"] = []

    with pytest.raises(ValueError, match="assess every cited label"):
        ClaimVerificationBundle.model_validate(payload)


def test_verdict_is_derived_from_evidence_relationships() -> None:
    payload = _payload()
    payload["assessments"][1]["verdict"] = "supported"

    with pytest.raises(ValueError, match="conflicts with evidence relationships"):
        ClaimVerificationBundle.model_validate(payload)


def test_claim_without_citation_requirement_cannot_attach_evidence() -> None:
    payload = _payload()
    payload["answer"] = payload["answer"].replace(
        "The following sentence is a transition.",
        "The following sentence is a transition [1].",
    )
    payload["claims"][4]["source_text"] = "The following sentence is a transition [1]."
    payload["claims"][4]["citation_labels"] = [1]
    payload["assessments"][4]["cited_evidence"] = [
        {
            "citation_label": 1,
            "relationship": "entails",
            "reason": "Unnecessary evidence.",
        }
    ]

    with pytest.raises(ValueError, match="do not require citation cannot carry"):
        ClaimVerificationBundle.model_validate(payload)


def test_exported_json_schema_is_strict_and_versioned() -> None:
    schema = claim_verification_json_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith("evaluation/schema/claim-verification.schema.json")
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "contract_version",
        "answer",
        "evidence_count",
        "claims",
        "assessments",
    }


def test_committed_json_schema_matches_the_pydantic_contract() -> None:
    committed = json.loads(
        Path("evaluation/schema/claim-verification.schema.json").read_text(encoding="utf-8")
    )

    assert committed == claim_verification_json_schema()


def test_validated_bundle_adapts_directly_to_task_9_metrics() -> None:
    bundle = load_claim_verification(FIXTURE)
    case = citation_case_from_claim_verification(
        bundle,
        case_id="claim_contract",
        evidence_ids=["e1", "e2", "e3", "e4"],
    )
    suite = CitationSafetySuite(
        contract_version="1.0.0",
        suite_id="claim-contract-metrics",
        dataset_version="0.1.0",
        benchmark_status="fixture",
        description="Contract bridge test.",
        cases=[case],
    )

    metrics = evaluate_citation_safety(suite).metrics
    assert metrics["citation_precision"] == 0.25
    assert metrics["citation_completeness"] == 0.8
    assert metrics["unsupported_claim_rate"] == 0.8
    assert metrics["invalid_citation_rate"] == 0.0
