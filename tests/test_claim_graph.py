from app.agent import graph
from app.models.claim_verifier import ClaimVerificationRun
from app.models.claims import (
    CLAIM_VERIFICATION_CONTRACT_VERSION,
    AtomicClaim,
    ClaimAssessment,
    ClaimEvidenceLink,
    ClaimVerificationBundle,
)


def _bundle(*verdicts: str) -> ClaimVerificationBundle:
    source_parts = [f"Fact {number} [{number}]." for number in range(1, len(verdicts) + 1)]
    claims = []
    assessments = []
    for number, (source, verdict) in enumerate(zip(source_parts, verdicts), start=1):
        relationship = {
            "supported": "entails",
            "partial": "partial",
            "unsupported": "does_not_support",
        }[verdict]
        claims.append(
            AtomicClaim(
                claim_id=f"claim_{number}",
                claim_text=f"Fact {number}.",
                source_text=source,
                requires_citation=True,
                citation_labels=[number],
            )
        )
        assessments.append(
            ClaimAssessment(
                claim_id=f"claim_{number}",
                verdict=verdict,
                cited_evidence=[
                    ClaimEvidenceLink(
                        citation_label=number,
                        relationship=relationship,
                        reason=f"Evidence is {relationship}.",
                    )
                ],
                reason=f"Claim is {verdict}.",
            )
        )
    return ClaimVerificationBundle(
        contract_version=CLAIM_VERIFICATION_CONTRACT_VERSION,
        answer=" ".join(source_parts),
        evidence_count=len(verdicts),
        claims=claims,
        assessments=assessments,
    )


def test_synthesis_routing_requires_sufficient_evidence_and_safe_citations() -> None:
    assert graph.route_after_synthesis({"evidence_sufficient": False}) == "end"
    assert (
        graph.route_after_synthesis(
            {"evidence_sufficient": True, "synthesis_citation_valid": True}
        )
        == "verify_claims"
    )
    assert (
        graph.route_after_synthesis(
            {"evidence_sufficient": True, "synthesis_citation_valid": False}
        )
        == "abstain"
    )


def test_claim_status_distinguishes_verified_repairable_and_unsupported() -> None:
    assert graph._claim_verification_status(_bundle("supported")) == "verified"
    assert graph._claim_verification_status(_bundle("partial")) == "repairable"
    assert (
        graph._claim_verification_status(_bundle("supported", "unsupported"))
        == "repairable"
    )
    assert graph._claim_verification_status(_bundle("unsupported")) == "unsupported"


def test_claim_verification_stores_validated_result(monkeypatch) -> None:
    bundle = _bundle("supported")
    monkeypatch.setattr(
        graph,
        "verify_answer_claims_bounded",
        lambda *args: ClaimVerificationRun(
            bundle=bundle, model_calls=1, output_repaired=False
        ),
    )

    result = graph.verify_claims(
        {
            "answer": "Fact 1 [1].\n\nSources:\n[1] trusted",
            "verified_evidence": [{"text": "Fact 1."}],
            "user_query": "Question?",
        }
    )

    assert result["claim_verification_status"] == "verified"
    assert result["claim_verification_attempt_count"] == 1
    assert result["claim_verification"]["claims"][0]["claim_id"] == "claim_1"


def test_claim_verification_failure_is_fail_closed(monkeypatch) -> None:
    def fail(*args):
        raise ValueError("invalid model output")

    monkeypatch.setattr(graph, "verify_answer_claims_bounded", fail)
    result = graph.verify_claims(
        {"answer": "Fact [1].", "verified_evidence": [{}], "user_query": "Question?"}
    )

    assert result["claim_verification_status"] == "invalid"
    assert result["claim_verification_attempt_count"] == 1
    assert "invalid model output" in result["claim_verification_error"]


def test_claim_verification_counts_bounded_output_repair(monkeypatch) -> None:
    bundle = _bundle("supported")
    monkeypatch.setattr(
        graph,
        "verify_answer_claims_bounded",
        lambda *args: ClaimVerificationRun(
            bundle=bundle, model_calls=2, output_repaired=True
        ),
    )
    result = graph.verify_claims(
        {"answer": "Fact 1 [1].", "verified_evidence": [{}], "user_query": "Question?"}
    )
    assert result["claim_verification_status"] == "verified"
    assert result["claim_verification_attempt_count"] == 2


def test_repair_route_allows_exactly_one_revision() -> None:
    assert (
        graph.route_after_claim_verification(
            {"claim_verification_status": "repairable", "claim_revision_count": 0}
        )
        == "revise"
    )
    assert (
        graph.route_after_claim_verification(
            {
                "claim_verification_status": "repairable",
                "claim_revision_count": graph.MAX_CLAIM_REVISIONS,
            }
        )
        == "abstain"
    )
    assert (
        graph.route_after_claim_verification(
            {"claim_verification_status": "unsupported", "claim_revision_count": 0}
        )
        == "abstain"
    )


def test_revision_is_recorded_and_reverified(monkeypatch) -> None:
    bundle = _bundle("partial")
    revised = "Narrowed fact [1].\n\nSources:\n[1] trusted"
    monkeypatch.setattr(graph, "repair_answer_claims", lambda *args: revised)
    state = {
        "user_query": "Question?",
        "answer": "Fact 1 [1].\n\nSources:\n[1] trusted",
        "verified_evidence": [{"arxiv_id": "paper", "text": "Fact 1."}],
        "candidate_papers": [],
        "claim_verification": bundle.model_dump(mode="json"),
        "claim_revision_count": 0,
        "claim_revision_history": ["Fact 1 [1].\n\nSources:\n[1] trusted"],
    }

    result = graph.revise_answer(state)

    assert result["answer"] == revised
    assert result["claim_revision_count"] == 1
    assert result["claim_revision_history"][-1] == revised
    assert graph.route_after_revision(result) == "verify_claims"


def test_failed_revision_abstains_without_another_loop(monkeypatch) -> None:
    bundle = _bundle("partial")

    def fail(*args):
        raise ValueError("repair failed")

    monkeypatch.setattr(graph, "repair_answer_claims", fail)
    result = graph.revise_answer(
        {
            "user_query": "Question?",
            "answer": "Fact 1 [1].",
            "verified_evidence": [{}],
            "candidate_papers": [],
            "claim_verification": bundle.model_dump(mode="json"),
            "claim_revision_count": 0,
            "claim_revision_history": ["Fact 1 [1]."],
        }
    )

    assert result["claim_revision_count"] == 1
    assert result["claim_verification_status"] == "invalid"
    assert graph.route_after_revision(result) == "abstain"
    abstention = graph.abstain_on_claims(result)
    assert abstention["claim_verification_status"] == "abstained"
    assert "One bounded revision was attempted" in abstention["answer"]
