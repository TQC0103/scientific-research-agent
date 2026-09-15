from types import SimpleNamespace

from app.models import llm
from app.models.llm import (
    INVALID_CITATION_MESSAGE,
    MISSING_CITATION_MESSAGE,
    AnswerSynthesisRun,
    answer_from_evidence_bounded,
    format_verified_sources,
    parse_citation_labels,
)


def test_sources_are_derived_from_verified_metadata() -> None:
    evidence = [
        {
            "arxiv_id": "1706.03762",
            "page": 5,
            "section": "Attention",
            "text": "Evidence",
        }
    ]
    papers = {"1706.03762": {"title": "Attention Is All You Need"}}
    result = format_verified_sources("Multi-head attention uses subspaces. [1]", evidence, papers)
    assert "p.5, Attention" in result
    assert "Attention Is All You Need" in result


def test_abstract_source_uses_version_and_does_not_invent_page() -> None:
    evidence = [
        {
            "arxiv_id": "1706.03762",
            "versioned_id": "1706.03762v7",
            "page": None,
            "section": "Abstract",
            "text": "Evidence",
        }
    ]
    papers = {"1706.03762": {"title": "Attention Is All You Need"}}
    result = format_verified_sources("Abstract-level claim. [1]", evidence, papers)
    assert "arXiv:1706.03762v7" in result
    assert "— Abstract" in result
    assert "p.None" not in result


def test_missing_citation_fails_closed_without_attaching_first_source() -> None:
    evidence = [{"arxiv_id": "1706.03762", "page": 4, "section": "Method", "text": "x"}]

    result = format_verified_sources("A scientific claim without a label.", evidence, {})

    assert result == MISSING_CITATION_MESSAGE
    assert "[1]" not in result
    assert "Sources:" not in result


def test_invalid_citation_fails_closed_even_when_another_label_is_valid() -> None:
    evidence = [{"arxiv_id": "1706.03762", "page": 4, "section": "Method", "text": "x"}]

    result = format_verified_sources("Claim [1], fabricated source [9].", evidence, {})

    assert result == INVALID_CITATION_MESSAGE
    assert "Sources:" not in result


def test_citation_parser_deduplicates_and_separates_invalid_labels() -> None:
    labels = parse_citation_labels("First [2], repeated [2], invalid [0] and [7].", 2)

    assert labels.valid == (2,)
    assert labels.invalid == (0, 7)


def test_bounded_synthesis_repairs_only_citations_and_hides_paper_references(
    monkeypatch,
) -> None:
    responses = iter(
        [
            "Transformer (big) achieved 28.4 BLEU [18].",
            "Transformer (big) achieved 28.4 BLEU [1].",
        ]
    )
    prompts = []

    class FakeModel:
        def invoke(self, prompt: str) -> SimpleNamespace:
            prompts.append(prompt)
            return SimpleNamespace(content=next(responses))

    monkeypatch.setattr(llm, "get_llm", lambda **kwargs: FakeModel())
    evidence = [
        {
            "arxiv_id": "1706.03762",
            "versioned_id": "1706.03762v7",
            "page": 8,
            "section": "Results",
            "text": "Prior system [18] scored lower; Transformer (big) scored 28.4 BLEU.",
        }
    ]

    run = answer_from_evidence_bounded(
        "What BLEU score was reported?",
        evidence,
        {"1706.03762": {"title": "Attention Is All You Need"}},
    )

    assert isinstance(run, AnswerSynthesisRun)
    assert run.model_calls == 2
    assert run.citation_repair_count == 1
    assert run.citation_repair_error is None
    assert run.answer.startswith("Transformer (big) achieved 28.4 BLEU [1].")
    assert "[18]" not in prompts[0]
    assert "(paper reference omitted)" in prompts[0]


def test_bounded_synthesis_can_add_a_missing_citation_without_changing_text(
    monkeypatch,
) -> None:
    responses = iter(["Transformer scored 28.4 BLEU.", "Transformer scored 28.4 BLEU [1]."])

    class FakeModel:
        def invoke(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(content=next(responses))

    monkeypatch.setattr(llm, "get_llm", lambda **kwargs: FakeModel())
    evidence = [
        {
            "arxiv_id": "1706.03762",
            "page": 1,
            "section": "Abstract",
            "text": "Transformer scored 28.4 BLEU.",
        }
    ]

    run = answer_from_evidence_bounded("What was the score?", evidence, {})

    assert run.answer.startswith("Transformer scored 28.4 BLEU [1].")
    assert run.model_calls == 2
    assert run.citation_repair_count == 1
    assert run.citation_repair_error is None


def test_bounded_citation_repair_fails_closed_if_answer_text_changes(monkeypatch) -> None:
    responses = iter(["Fact [9].", "Different fact [1]."])

    class FakeModel:
        def invoke(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(content=next(responses))

    monkeypatch.setattr(llm, "get_llm", lambda **kwargs: FakeModel())
    evidence = [
        {
            "arxiv_id": "1706.03762",
            "page": 1,
            "section": "Results",
            "text": "Fact.",
        }
    ]

    run = answer_from_evidence_bounded("Question?", evidence, {})

    assert run.answer == INVALID_CITATION_MESSAGE
    assert run.model_calls == 2
    assert run.citation_repair_count == 1
    assert run.citation_repair_error == "Citation repair changed non-citation answer text."
