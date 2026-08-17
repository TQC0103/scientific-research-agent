from app.tools.arxiv_search import split_arxiv_id


def test_split_modern_versioned_id() -> None:
    assert split_arxiv_id("1706.03762v7") == ("1706.03762v7", "1706.03762", 7)


def test_split_legacy_versioned_id() -> None:
    assert split_arxiv_id("cs/9901001v2") == ("cs/9901001v2", "cs/9901001", 2)
