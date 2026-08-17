import pytest

from app.agent import graph
from app.models import verifier
from app.models.verifier import EvidenceVerification, _extract_json


def _chunk(index: int, text: str = "Direct supporting passage.") -> dict:
    return {
        "arxiv_id": "1706.03762",
        "versioned_id": "1706.03762v7",
        "page": 1,
        "section": "Introduction",
        "chunk_index": index,
        "text": text,
        "score": 0.5,
    }


def test_extract_json_accepts_fenced_or_prefixed_model_output() -> None:
    payload = _extract_json('```json\n{"sufficient": false, "reason": "missing"}\n```')
    assert payload == {"sufficient": False, "reason": "missing"}


def test_verifier_repairs_contradictory_false_with_no_missing_information(monkeypatch) -> None:
    class FakeModel:
        def invoke(self, prompt):
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"sufficient": false, "reason": "Fully supported", '
                        '"missing_information": [], "suggested_query": null, '
                        '"supported_evidence": [1]}'
                    )
                },
            )()

    monkeypatch.setattr(verifier, "get_llm", lambda **kwargs: FakeModel())
    result = verifier.verify_evidence("Question", [_chunk(0)])
    assert result.sufficient is True
    assert result.suggested_query is None


def test_check_rewrites_query_when_llm_verifier_finds_a_gap(monkeypatch) -> None:
    monkeypatch.setattr(
        graph,
        "verify_evidence",
        lambda question, evidence, current_query: EvidenceVerification(
            sufficient=False,
            reason="The passages do not explain token order.",
            missing_information=["The positional encoding formula and purpose."],
            suggested_query="positional encoding token order sine cosine",
            supported_evidence=[],
        ),
    )
    result = graph.check_evidence(
        {
            "user_query": "How is token order represented?",
            "retrieval_query": "How is token order represented?",
            "retrieved_chunks": [_chunk(0)],
            "retrieval_attempt_count": 0,
            "tool_errors": [],
        }
    )
    assert result["evidence_sufficient"] is False
    assert result["should_retry_retrieval"] is True
    assert result["retrieval_query"] == "positional encoding token order sine cosine"


def test_synthesis_uses_only_passages_approved_by_verifier(monkeypatch) -> None:
    captured = {}

    def fake_answer(question, evidence, papers):
        captured["evidence"] = evidence
        return "Verified answer."

    monkeypatch.setattr(graph, "answer_from_evidence", fake_answer)
    evidence = [_chunk(0, "unsupported"), _chunk(1, "supported")]
    result = graph.synthesize(
        {
            "user_query": "Question",
            "candidate_papers": [{"arxiv_id": "1706.03762", "title": "Paper"}],
            "retrieved_chunks": evidence,
            "evidence_sufficient": True,
            "evidence_verification": {"supported_evidence": [2]},
            "tool_errors": [],
        }
    )
    assert result["answer"] == "Verified answer."
    assert captured["evidence"] == [evidence[1]]


def test_insufficient_evidence_stops_without_calling_synthesis_model(monkeypatch) -> None:
    def fail_answer(*args, **kwargs):
        raise AssertionError("Synthesis model must not run")

    monkeypatch.setattr(graph, "answer_from_evidence", fail_answer)
    result = graph.synthesize(
        {
            "user_query": "Question",
            "candidate_papers": [],
            "retrieved_chunks": [_chunk(0)],
            "evidence_sufficient": False,
            "evidence_verification": {
                "reason": "No direct result was retrieved.",
                "missing_information": ["The measured result."],
                "supported_evidence": [],
            },
            "tool_errors": [],
        }
    )
    assert "Insufficient evidence" in result["answer"]
    assert "The measured result" in result["answer"]


def test_invalid_verifier_response_fails_closed(monkeypatch) -> None:
    def fail_verifier(*args, **kwargs):
        raise ValueError("bad JSON")

    monkeypatch.setattr(graph, "verify_evidence", fail_verifier)
    result = graph.check_evidence(
        {
            "user_query": "Question",
            "retrieval_query": "Question",
            "retrieved_chunks": [_chunk(0)],
            "retrieval_attempt_count": 0,
            "tool_errors": [],
        }
    )
    assert result["evidence_sufficient"] is False
    assert result["should_retry_retrieval"] is False
    assert result["tool_errors"] == ["verifier: bad JSON"]


@pytest.mark.parametrize(
    ("state", "route"),
    [
        ({"evidence_sufficient": True}, "synthesize"),
        ({"evidence_sufficient": False, "should_retry_retrieval": True}, "retrieve"),
        (
            {
                "evidence_sufficient": False,
                "should_retry_retrieval": False,
                "candidate_papers": [],
            },
            "synthesize",
        ),
    ],
)
def test_route_after_check(state, route) -> None:
    assert graph.route_after_check(state) == route
