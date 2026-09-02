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


def _paper_chunk(paper_id: str, index: int, text: str) -> dict:
    item = _chunk(index, text)
    item["arxiv_id"] = paper_id
    item["versioned_id"] = f"{paper_id}v1"
    return item


def test_accumulated_passages_are_bounded_with_new_results_first(monkeypatch) -> None:
    monkeypatch.setattr(graph.settings, "max_accumulated_passages_per_paper", 8)
    previous = [_chunk(index) for index in range(8)]
    new = [_chunk(index) for index in range(8, 13)]

    merged = graph._merge_passages(new, previous)

    assert len(merged) == 8
    assert [item["chunk_index"] for item in merged] == [8, 9, 10, 11, 12, 0, 1, 2]


def test_extract_json_accepts_fenced_or_prefixed_model_output() -> None:
    payload = _extract_json('```json\n{"sufficient": false, "reason": "missing"}\n```')
    assert payload == {"sufficient": False, "reason": "missing"}


def test_verifier_preserves_false_decision_with_partial_support(monkeypatch) -> None:
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
    assert result.sufficient is False
    assert result.missing_information == ["At least one requested element remains unsupported."]
    assert result.suggested_query == "Question"


def test_verifier_rejects_true_decision_with_explicit_missing_requirement(
    monkeypatch,
) -> None:
    class FakeModel:
        def invoke(self, prompt):
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"sufficient": true, "reason": "Two of three factors are present", '
                        '"missing_information": ["inference latency reduction factor"], '
                        '"suggested_query": "inference latency numerical factor", '
                        '"supported_evidence": [1]}'
                    )
                },
            )()

    monkeypatch.setattr(verifier, "get_llm", lambda **kwargs: FakeModel())
    result = verifier.verify_evidence(
        "By what factors were parameters, memory, and latency reduced?", [_chunk(0)]
    )

    assert result.sufficient is False
    assert result.missing_information == ["inference latency reduction factor"]
    assert result.supported_evidence == [1]
    assert result.suggested_query == "inference latency numerical factor"


def test_synthesis_fails_closed_on_internally_inconsistent_verification(
    monkeypatch,
) -> None:
    def fail_answer(*args, **kwargs):
        raise AssertionError("Synthesis model must not run")

    monkeypatch.setattr(graph, "answer_from_evidence", fail_answer)
    result = graph.synthesize(
        {
            "user_query": "By what factors were parameters, memory, and latency reduced?",
            "candidate_papers": [{"arxiv_id": "1706.03762", "title": "Paper"}],
            "selected_papers": ["1706.03762"],
            "retrieved_chunks_by_paper": {"1706.03762": [_chunk(0)]},
            "evidence_sufficient": True,
            "evidence_verifications": {
                "1706.03762": {
                    "sufficient": True,
                    "reason": "Only two of three requested factors are present.",
                    "missing_information": ["inference latency reduction factor"],
                    "supported_evidence": [1],
                }
            },
            "tool_errors": [],
        }
    )

    assert result["evidence_sufficient"] is False
    assert result["verified_evidence"] == []
    assert "inference latency reduction factor" in result["answer"]


@pytest.mark.parametrize(
    ("question", "passage", "missing"),
    [
        (
            "How much electrical energy did training consume?",
            "Training took 3.5 days on eight GPUs and used 2.3e19 FLOPs.",
            "electrical-energy",
        ),
        (
            "What ImageNet top-1 accuracy is reported?",
            "The model achieved 28.4 BLEU on WMT English-to-German.",
            "ImageNet",
        ),
    ],
)
def test_semantic_anchor_guard_rejects_wrong_metric_substitution(
    monkeypatch, question, passage, missing
) -> None:
    class FakeModel:
        def invoke(self, prompt):
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"sufficient": true, "reason": "Related result", '
                        '"missing_information": [], "suggested_query": null, '
                        '"supported_evidence": [1]}'
                    )
                },
            )()

    monkeypatch.setattr(verifier, "get_llm", lambda **kwargs: FakeModel())
    result = verifier.verify_evidence(question, [_chunk(0, passage)])
    assert result.sufficient is False
    assert result.supported_evidence == []
    assert missing in result.reason
    assert result.suggested_query != question


def test_semantic_anchor_guard_accepts_explicit_requested_metric(monkeypatch) -> None:
    class FakeModel:
        def invoke(self, prompt):
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"sufficient": true, "reason": "Direct result", '
                        '"missing_information": [], "suggested_query": null, '
                        '"supported_evidence": [1]}'
                    )
                },
            )()

    monkeypatch.setattr(verifier, "get_llm", lambda **kwargs: FakeModel())
    result = verifier.verify_evidence(
        "What ImageNet top-1 accuracy is reported?",
        [_chunk(0, "On ImageNet, the model reports top-1 accuracy of 81.2%.")],
    )
    assert result.sufficient is True
    assert result.supported_evidence == [1]


def test_check_rewrites_query_when_llm_verifier_finds_a_gap(monkeypatch) -> None:
    monkeypatch.setattr(
        graph,
        "verify_evidence",
        lambda question, evidence, current_query, scope: EvidenceVerification(
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
            "selected_papers": ["1706.03762"],
            "coverage_mode": "any",
            "papers_to_retrieve": ["1706.03762"],
            "retrieval_queries": {"1706.03762": "How is token order represented?"},
            "retrieved_chunks": [_chunk(0)],
            "retrieved_chunks_by_paper": {"1706.03762": [_chunk(0)]},
            "retrieval_attempt_counts": {},
            "tool_errors": [],
        }
    )
    assert result["evidence_sufficient"] is False
    assert result["should_retry_retrieval"] is True
    assert result["retrieval_queries"]["1706.03762"] == (
        "positional encoding token order sine cosine"
    )
    assert result["papers_to_retrieve"] == ["1706.03762"]


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
            "selected_papers": ["1706.03762"],
            "retrieved_chunks": evidence,
            "retrieved_chunks_by_paper": {"1706.03762": evidence},
            "evidence_sufficient": True,
            "evidence_verifications": {
                "1706.03762": {"sufficient": True, "supported_evidence": [2]}
            },
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
            "selected_papers": ["1706.03762"],
            "retrieved_chunks": [_chunk(0)],
            "retrieved_chunks_by_paper": {"1706.03762": [_chunk(0)]},
            "evidence_sufficient": False,
            "evidence_verifications": {
                "1706.03762": {
                    "sufficient": False,
                    "reason": "No direct result was retrieved.",
                    "missing_information": ["The measured result."],
                    "supported_evidence": [],
                }
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
            "selected_papers": ["1706.03762"],
            "coverage_mode": "any",
            "papers_to_retrieve": ["1706.03762"],
            "retrieval_queries": {"1706.03762": "Question"},
            "retrieved_chunks": [_chunk(0)],
            "retrieved_chunks_by_paper": {"1706.03762": [_chunk(0)]},
            "retrieval_attempt_counts": {},
            "tool_errors": [],
        }
    )
    assert result["evidence_sufficient"] is False
    assert result["should_retry_retrieval"] is False
    assert result["tool_errors"] == ["verifier 1706.03762: bad JSON"]


def test_multi_paper_verification_retries_only_the_missing_paper(monkeypatch) -> None:
    paper_a = "2401.00001"
    paper_b = "2401.00002"
    scoped_questions = []

    def fake_verify(question, evidence, current_query, scope):
        scoped_questions.append((question, scope))
        if evidence[0]["arxiv_id"] == paper_a:
            return EvidenceVerification(
                sufficient=True,
                reason="Paper A is covered.",
                supported_evidence=[1],
            )
        return EvidenceVerification(
            sufficient=False,
            reason="Paper B lacks its measured result.",
            missing_information=["Paper B measured result"],
            suggested_query="paper B experiment measured result table",
        )

    monkeypatch.setattr(graph, "verify_evidence", fake_verify)
    chunks = {
        paper_a: [_paper_chunk(paper_a, 0, "Paper A evidence")],
        paper_b: [_paper_chunk(paper_b, 0, "Paper B background")],
    }
    result = graph.check_evidence(
        {
            "user_query": "Compare paper A and paper B.",
            "candidate_papers": [
                {"arxiv_id": paper_a, "title": "A"},
                {"arxiv_id": paper_b, "title": "B"},
            ],
            "selected_papers": [paper_a, paper_b],
            "required_paper_ids": [paper_a, paper_b],
            "required_paper_count": 2,
            "coverage_mode": "all",
            "papers_to_retrieve": [paper_a, paper_b],
            "retrieved_chunks_by_paper": chunks,
            "retrieval_queries": {},
            "retrieval_attempt_counts": {},
            "tool_errors": [],
        }
    )
    assert result["evidence_sufficient"] is False
    assert result["papers_to_retrieve"] == [paper_b]
    assert result["evidence_verifications"][paper_a]["sufficient"] is True
    assert result["evidence_verifications"][paper_b]["sufficient"] is False
    assert all(
        "Ignore all missing information about other papers" in item[0] for item in scoped_questions
    )
    assert all(
        "Do not require passages about the other papers" in item[1] for item in scoped_questions
    )


def test_multi_paper_synthesis_keeps_approved_evidence_separate(monkeypatch) -> None:
    paper_a = "2401.00001"
    paper_b = "2401.00002"
    chunks = {
        paper_a: [
            _paper_chunk(paper_a, 0, "A unsupported"),
            _paper_chunk(paper_a, 1, "A supported"),
        ],
        paper_b: [_paper_chunk(paper_b, 0, "B supported")],
    }
    captured = {}

    def fake_answer(question, evidence, papers):
        captured["evidence"] = evidence
        return "Comparison."

    monkeypatch.setattr(graph, "answer_from_evidence", fake_answer)
    result = graph.synthesize(
        {
            "user_query": "Compare A and B",
            "candidate_papers": [
                {"arxiv_id": paper_a, "title": "A"},
                {"arxiv_id": paper_b, "title": "B"},
            ],
            "selected_papers": [paper_a, paper_b],
            "retrieved_chunks_by_paper": chunks,
            "evidence_sufficient": True,
            "evidence_verifications": {
                paper_a: {"sufficient": True, "supported_evidence": [2]},
                paper_b: {"sufficient": True, "supported_evidence": [1]},
            },
            "tool_errors": [],
        }
    )
    assert result["answer"] == "Comparison."
    assert [item["text"] for item in captured["evidence"]] == [
        "A supported",
        "B supported",
    ]


@pytest.mark.parametrize(
    ("state", "route"),
    [
        ({"evidence_sufficient": True}, "synthesize"),
        ({"evidence_sufficient": False, "papers_to_retrieve": ["1706.03762"]}, "retrieve"),
        (
            {
                "evidence_sufficient": False,
                "should_retry_retrieval": False,
                "candidate_papers": [],
            },
            "synthesize",
        ),
        (
            {
                "evidence_sufficient": False,
                "papers_to_retrieve": [],
                "selected_papers": ["2401.00001"],
                "required_paper_ids": ["2401.00001", "2401.00002"],
                "candidate_papers": [
                    {"arxiv_id": "2401.00001"},
                    {"arxiv_id": "2401.00002"},
                ],
                "iteration_count": 1,
            },
            "index_next",
        ),
    ],
)
def test_route_after_check(state, route) -> None:
    assert graph.route_after_check(state) == route
