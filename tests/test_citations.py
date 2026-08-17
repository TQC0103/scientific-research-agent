from app.models.llm import format_verified_sources


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
