from app.agent import graph
from app.models.claims import (
    CLAIM_VERIFICATION_CONTRACT_VERSION,
    AtomicClaim,
    ClaimAssessment,
    ClaimEvidenceLink,
    ClaimVerificationBundle,
)
from app.models.verifier import EvidenceVerification


def test_full_graph_requires_and_synthesizes_both_explicit_papers(monkeypatch) -> None:
    paper_ids = ["2401.00001", "2401.00002"]
    verifier_calls = []

    monkeypatch.setattr(
        graph,
        "get_arxiv_metadata",
        lambda paper_id: {
            "arxiv_id": paper_id,
            "versioned_id": f"{paper_id}v1",
            "title": f"Paper {paper_id}",
            "abstract": f"Abstract {paper_id}",
        },
    )
    monkeypatch.setattr(graph, "index_is_current", lambda paper: True)
    monkeypatch.setattr(
        graph,
        "retrieve",
        lambda paper_id, query, top_k: [
            {
                "arxiv_id": paper_id,
                "versioned_id": f"{paper_id}v1",
                "page": 2,
                "section": "Method",
                "chunk_index": 0,
                "text": f"Supported method for {paper_id}",
                "score": 0.8,
                "retrieval_score": 0.03,
            }
        ],
    )

    def fake_verify(question, evidence, current_query, scope):
        verifier_calls.append(evidence[0]["arxiv_id"])
        return EvidenceVerification(
            sufficient=True,
            reason="This paper's side is covered.",
            supported_evidence=[1],
        )

    def fake_answer(question, evidence, papers):
        assert {item["arxiv_id"] for item in evidence} == set(paper_ids)
        return (
            "Grounded comparison [1][2].\n\nSources:\n"
            "[1] arXiv:2401.00001v1 — page 2, Method\n"
            "[2] arXiv:2401.00002v1 — page 2, Method"
        )

    def fake_claim_verification(answer, evidence, question):
        assert len(evidence) == 2
        body = answer.split("\n\nSources:\n", 1)[0]
        return ClaimVerificationBundle(
            contract_version=CLAIM_VERIFICATION_CONTRACT_VERSION,
            answer=body,
            evidence_count=2,
            claims=[
                AtomicClaim(
                    claim_id="claim_1",
                    claim_text="Grounded comparison.",
                    source_text=body,
                    requires_citation=True,
                    citation_labels=[1, 2],
                )
            ],
            assessments=[
                ClaimAssessment(
                    claim_id="claim_1",
                    verdict="supported",
                    cited_evidence=[
                        ClaimEvidenceLink(
                            citation_label=1,
                            relationship="entails",
                            reason="The first paper supports its side.",
                        ),
                        ClaimEvidenceLink(
                            citation_label=2,
                            relationship="entails",
                            reason="The second paper supports its side.",
                        ),
                    ],
                    reason="Both comparison sides are supported.",
                )
            ],
        )

    monkeypatch.setattr(graph, "verify_evidence", fake_verify)
    monkeypatch.setattr(graph, "answer_from_evidence", fake_answer)
    monkeypatch.setattr(graph, "verify_answer_claims", fake_claim_verification)

    result = graph.build_graph().invoke(
        {"user_query": "Compare the methods", "paper_ids": paper_ids}
    )

    assert result["coverage_mode"] == "all"
    assert result["selected_papers"] == paper_ids
    assert result["evidence_sufficient"] is True
    assert verifier_calls == paper_ids
    assert result["answer"].startswith("Grounded comparison [1][2].")
    assert result["claim_verification_status"] == "verified"
    assert result["claim_verification_attempt_count"] == 1
    assert result["claim_revision_count"] == 0
