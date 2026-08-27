from app.models.llm import (
    INVALID_CITATION_MESSAGE,
    MISSING_CITATION_MESSAGE,
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
