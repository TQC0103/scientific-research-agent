from app.agent import graph
from app.tools.paper_download import PdfUnavailableError


def test_discover_uses_local_fts_without_remote_when_catalog_is_sufficient(monkeypatch) -> None:
    local = [
        {"arxiv_id": f"2501.0000{number}", "title": f"Paper {number}", "abstract": "x"}
        for number in range(1, 4)
    ]
    monkeypatch.setattr(graph, "search_local", lambda query, limit: local)

    def fail_remote(*args, **kwargs):
        raise AssertionError("Remote arXiv search should not run")

    monkeypatch.setattr(graph, "search_arxiv", fail_remote)
    result = graph.discover({"user_query": "scientific retrieval", "paper_ids": []})
    assert result["discovery_source"] == "sqlite_fts5"
    assert result["candidate_papers"] == local


def test_pdf_failure_falls_back_to_abstract(monkeypatch) -> None:
    paper = {
        "arxiv_id": "2501.00001",
        "versioned_id": "2501.00001v2",
        "title": "Paper",
        "abstract": "Abstract-only evidence.",
    }
    monkeypatch.setattr(graph, "get_arxiv_metadata", lambda paper_id: paper)
    monkeypatch.setattr(graph, "index_is_current", lambda candidate: False)

    def fail_index(*args, **kwargs):
        raise PdfUnavailableError("PDF unavailable")

    monkeypatch.setattr(graph, "index_paper", fail_index)
    state = {
        "user_query": "What does the paper claim?",
        "candidate_papers": [paper],
        "selected_papers": [],
        "failed_papers": [],
        "tool_errors": [],
        "iteration_count": 0,
    }
    updated = graph.index_next(state)
    evidence = graph.retrieve_evidence({**state, **updated})["retrieved_chunks"]
    assert updated["failed_papers"] == ["2501.00001"]
    assert evidence[0]["section"] == "Abstract"
    assert evidence[0]["page"] is None
