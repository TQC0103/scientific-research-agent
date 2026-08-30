import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models import claim_verifier
from app.models.claims import ClaimVerdict, EvidenceRelationship

FIXTURE = Path("evaluation/suites/v0_5/claim_verification_fixtures.json")


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _evidence() -> list[dict]:
    return [
        {
            "arxiv_id": "1706.03762",
            "versioned_id": "1706.03762v7",
            "page": number,
            "section": "Results",
            "text": f"Synthetic verifier-approved passage {number}.",
        }
        for number in range(1, 5)
    ]


def _span_payload(payload: dict) -> dict:
    spans = claim_verifier.answer_source_spans(payload["answer"])
    span_ids = {span.text: span.span_id for span in spans}
    assessments = {item["claim_id"]: item for item in payload["assessments"]}
    return {
        "claims": [
            {
                "claim_text": claim["claim_text"],
                "source_span_id": span_ids[claim["source_text"]],
                "requires_citation": claim["requires_citation"],
                "evidence_judgments": [
                    {
                        "relationship": item["relationship"],
                        "reason": item["reason"],
                    }
                    for item in assessments[claim["claim_id"]]["cited_evidence"]
                ],
                "assessment_reason": assessments[claim["claim_id"]]["reason"],
            }
            for claim in payload["claims"]
        ],
    }


def _fake_model(monkeypatch, payload: dict, captured: dict | None = None) -> None:
    class FakeModel:
        def invoke(self, prompt: str) -> SimpleNamespace:
            if captured is not None:
                captured["prompt"] = prompt
            return SimpleNamespace(content=json.dumps(payload))

    def factory(**kwargs):
        if captured is not None:
            captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(claim_verifier, "get_llm", factory)


def test_verifier_returns_supported_partial_unsupported_and_not_required(monkeypatch) -> None:
    payload = _payload()
    _fake_model(monkeypatch, payload)

    bundle = claim_verifier.verify_answer_claims(payload["answer"], _evidence())

    assert [item.verdict for item in bundle.assessments] == [
        ClaimVerdict.SUPPORTED,
        ClaimVerdict.PARTIAL,
        ClaimVerdict.UNSUPPORTED,
        ClaimVerdict.UNSUPPORTED,
        ClaimVerdict.NOT_REQUIRED,
        ClaimVerdict.UNSUPPORTED,
    ]
    assert bundle.assessments[0].cited_evidence[0].relationship == EvidenceRelationship.ENTAILS
    assert bundle.assessments[1].cited_evidence[0].relationship == EvidenceRelationship.PARTIAL


def test_wrong_citation_and_missing_citation_remain_unsupported(monkeypatch) -> None:
    payload = _payload()
    _fake_model(monkeypatch, payload)

    bundle = claim_verifier.verify_answer_claims(payload["answer"], _evidence())

    wrong = bundle.assessments[3]
    missing = bundle.assessments[5]
    assert wrong.cited_evidence[0].citation_label == 4
    assert wrong.cited_evidence[0].relationship == EvidenceRelationship.DOES_NOT_SUPPORT
    assert wrong.verdict == ClaimVerdict.UNSUPPORTED
    assert missing.cited_evidence == []
    assert missing.verdict == ClaimVerdict.UNSUPPORTED


def test_prompt_uses_only_numbered_approved_evidence_and_exact_contract(monkeypatch) -> None:
    payload = _payload()
    captured = {}
    _fake_model(monkeypatch, payload, captured)

    claim_verifier.verify_answer_claims(
        payload["answer"], _evidence(), question="What did the paper report?"
    )

    prompt = captured["prompt"]
    assert "verifier-approved evidence" in prompt
    assert "Do not repair, rewrite, or answer the question" in prompt
    assert "What did the paper report?" in prompt
    assert "[4] 1706.03762v7 — page 4, Results" in prompt
    assert payload["answer"] in prompt
    assert captured["kwargs"] == {"temperature": 0, "num_predict": 1800}


def test_deterministic_sources_block_is_excluded_before_extraction(monkeypatch) -> None:
    payload = _payload()
    captured = {}
    _fake_model(monkeypatch, payload, captured)
    rendered = payload["answer"] + "\n\nSources:\n[1] trusted metadata"

    bundle = claim_verifier.verify_answer_claims(rendered, _evidence())

    assert bundle.answer == payload["answer"]
    assert "Sources:" not in captured["prompt"]


def test_parser_accepts_fenced_json_but_rejects_changed_answer() -> None:
    payload = _payload()
    fenced = f"```json\n{json.dumps(payload)}\n```"

    bundle = claim_verifier.parse_claim_verifier_response(
        fenced,
        expected_answer=payload["answer"],
        evidence_count=4,
    )
    assert len(bundle.claims) == 6

    payload["answer"] = "A changed answer."
    with pytest.raises(ValueError, match="changed the answer body"):
        claim_verifier.parse_claim_verifier_response(
            json.dumps(payload),
            expected_answer=bundle.answer,
            evidence_count=4,
        )


def test_parser_rejects_changed_evidence_count() -> None:
    payload = _payload()
    payload["evidence_count"] = 5

    with pytest.raises(ValueError, match="changed the supplied evidence count"):
        claim_verifier.parse_claim_verifier_response(
            json.dumps(payload),
            expected_answer=payload["answer"],
            evidence_count=4,
        )


def test_invalid_model_bundle_fails_closed(monkeypatch) -> None:
    payload = _payload()
    payload["assessments"][1]["verdict"] = "supported"
    _fake_model(monkeypatch, payload)

    with pytest.raises(ValueError, match="Invalid claim-verifier response"):
        claim_verifier.verify_answer_claims(payload["answer"], _evidence())


def test_bounded_verifier_repairs_invalid_source_and_citation_structure(monkeypatch) -> None:
    valid = _payload()
    span_valid = _span_payload(valid)
    invalid = json.loads(json.dumps(span_valid))
    invalid["claims"][0]["source_span_id"] = "span_99"
    responses = iter([json.dumps(invalid), json.dumps(span_valid)])
    prompts = []

    class FakeModel:
        def invoke(self, prompt: str) -> SimpleNamespace:
            prompts.append(prompt)
            return SimpleNamespace(content=next(responses))

    monkeypatch.setattr(claim_verifier, "get_llm", lambda **kwargs: FakeModel())
    run = claim_verifier.verify_answer_claims_bounded(valid["answer"], _evidence())

    assert run.bundle.answer == valid["answer"]
    assert run.model_calls == 2
    assert run.output_repaired is True
    assert "Repair the structure" in prompts[1]
    assert "claim IDs, source text" in prompts[1]
    assert "Verifier-approved evidence" not in prompts[1]


def test_bounded_verifier_fails_after_exactly_one_invalid_output_repair(monkeypatch) -> None:
    payload = _payload()
    invalid = _span_payload(payload)
    invalid["claims"][0]["source_span_id"] = "span_99"
    calls = 0

    class FakeModel:
        def invoke(self, prompt: str) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            return SimpleNamespace(content=json.dumps(invalid))

    monkeypatch.setattr(claim_verifier, "get_llm", lambda **kwargs: FakeModel())
    with pytest.raises(claim_verifier.ClaimVerificationRunError) as captured:
        claim_verifier.verify_answer_claims_bounded(payload["answer"], _evidence())

    assert calls == 2
    assert captured.value.model_calls == 2
    assert "after one bounded output repair" in str(captured.value)


def test_span_bound_parser_derives_exact_source_text_and_visible_labels() -> None:
    payload = _payload()
    span_payload = _span_payload(payload)

    bundle = claim_verifier.parse_span_bound_claim_response(
        json.dumps(span_payload),
        expected_answer=payload["answer"],
        evidence_count=4,
    )

    assert [claim.source_text for claim in bundle.claims] == [
        claim["source_text"] for claim in payload["claims"]
    ]
    assert [claim.citation_labels for claim in bundle.claims] == [
        claim["citation_labels"] for claim in payload["claims"]
    ]
    assert "source_text" not in span_payload["claims"][0]
    assert "citation_labels" not in span_payload["claims"][0]
    assert "claim_id" not in span_payload["claims"][0]
    assert "citation_label" not in span_payload["claims"][0]["evidence_judgments"][0]
    assert "verdict" not in span_payload["claims"][0]
    assert "answer" not in span_payload
    assert "evidence_count" not in span_payload
    assert "contract_version" not in span_payload


def test_answer_spans_keep_uncited_lead_in_the_following_citation_scope() -> None:
    answer = (
        "The architecture uses self-attention. This enables parallel processing [1]. "
        "A second result is reported [2]. A trailing uncited claim."
    )

    spans = claim_verifier.answer_source_spans(answer)

    assert [span.text for span in spans] == [
        "The architecture uses self-attention. This enables parallel processing [1].",
        "A second result is reported [2].",
        "A trailing uncited claim.",
    ]
    assert all(span.text in answer for span in spans)


def test_span_bound_parser_rejects_unknown_or_reordered_spans() -> None:
    payload = _payload()
    unknown = _span_payload(payload)
    unknown["claims"][0]["source_span_id"] = "span_99"
    with pytest.raises(ValueError, match="unknown source_span_id"):
        claim_verifier.parse_span_bound_claim_response(
            json.dumps(unknown),
            expected_answer=payload["answer"],
            evidence_count=4,
        )

    reordered = _span_payload(payload)
    reordered["claims"][0]["source_span_id"] = "span_2"
    reordered["claims"][1]["source_span_id"] = "span_1"
    with pytest.raises(ValueError, match="source_span_id order"):
        claim_verifier.parse_span_bound_claim_response(
            json.dumps(reordered),
            expected_answer=payload["answer"],
            evidence_count=4,
        )


def test_span_bound_parser_binds_judgments_to_visible_labels_by_position() -> None:
    payload = _payload()
    span_payload = _span_payload(payload)
    span_payload["claims"][0]["evidence_judgments"] = []

    with pytest.raises(ValueError, match="one evidence judgment per visible"):
        claim_verifier.parse_span_bound_claim_response(
            json.dumps(span_payload),
            expected_answer=payload["answer"],
            evidence_count=4,
        )


def test_empty_answer_or_evidence_never_calls_model(monkeypatch) -> None:
    monkeypatch.setattr(
        claim_verifier,
        "get_llm",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model must not load")),
    )

    with pytest.raises(ValueError, match="non-empty answer body"):
        claim_verifier.verify_answer_claims("  ", _evidence())
    with pytest.raises(ValueError, match="verifier-approved evidence"):
        claim_verifier.verify_answer_claims("Claim [1].", [])


def test_claim_repair_uses_only_failed_claims_and_restores_trusted_sources(
    monkeypatch,
) -> None:
    payload = _payload()
    bundle = claim_verifier.ClaimVerificationBundle.model_validate(payload)
    captured = {}

    class FakeModel:
        def invoke(self, prompt: str) -> SimpleNamespace:
            captured["prompt"] = prompt
            return SimpleNamespace(content="The measured result was 91% [1].")

    def factory(**kwargs):
        captured["kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(claim_verifier, "get_llm", factory)
    papers = {"1706.03762": {"title": "Attention Is All You Need"}}

    answer = claim_verifier.repair_answer_claims(
        "What did the paper report?",
        payload["answer"],
        _evidence(),
        papers,
        bundle,
    )

    assert answer.startswith("The measured result was 91% [1].\n\nSources:\n")
    assert "arXiv:1706.03762v7" in answer
    assert "claim_1 (supported)" not in captured["prompt"]
    assert "claim_2 (partial)" in captured["prompt"]
    assert "claim_3 (unsupported)" in captured["prompt"]
    assert captured["kwargs"] == {"temperature": 0, "num_predict": 1200}


def test_claim_repair_without_citation_fails_citation_safety(monkeypatch) -> None:
    payload = _payload()
    bundle = claim_verifier.ClaimVerificationBundle.model_validate(payload)

    class FakeModel:
        def invoke(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(content="A revised but uncited factual answer.")

    monkeypatch.setattr(claim_verifier, "get_llm", lambda **kwargs: FakeModel())

    answer = claim_verifier.repair_answer_claims(
        "Question?",
        payload["answer"],
        _evidence(),
        {"1706.03762": {"title": "Paper"}},
        bundle,
    )

    assert "citation" in answer.casefold()
    assert "\n\nSources:\n" not in answer
