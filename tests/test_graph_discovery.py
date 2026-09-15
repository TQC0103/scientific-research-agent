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
    assert result["coverage_mode"] == "any"


def test_explicit_multi_paper_discovery_requires_every_paper(monkeypatch) -> None:
    monkeypatch.setattr(
        graph,
        "get_arxiv_metadata",
        lambda paper_id: {"arxiv_id": paper_id, "title": f"Paper {paper_id}"},
    )
    result = graph.discover(
        {
            "user_query": "Compare the two methods",
            "paper_ids": ["2401.00001", "2401.00002"],
        }
    )
    assert result["coverage_mode"] == "all"
    assert result["required_paper_ids"] == ["2401.00001", "2401.00002"]
    assert result["required_paper_count"] == 2


def test_automatic_comparison_requires_two_candidates(monkeypatch) -> None:
    local = [
        {"arxiv_id": f"2401.0000{number}", "title": f"Paper {number}", "abstract": "x"}
        for number in range(1, 4)
    ]
    monkeypatch.setattr(graph, "search_local", lambda query, limit: local)
    result = graph.discover({"user_query": "Compare these approaches", "paper_ids": []})
    assert result["coverage_mode"] == "all"
    assert result["required_paper_ids"] == ["2401.00001", "2401.00002"]
    assert result["required_paper_count"] == 2


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


def test_new_index_refreshes_artifact_metadata_before_retrieval(monkeypatch) -> None:
    stale = {
        "arxiv_id": "2501.00001",
        "versioned_id": "2501.00001v2",
        "title": "Paper",
        "abstract": "",
        "pdf_sha256": None,
    }
    refreshed = {**stale, "pdf_sha256": "verified-pdf-hash"}
    monkeypatch.setattr(graph, "get_arxiv_metadata", lambda paper_id: stale)
    monkeypatch.setattr(graph, "get_paper", lambda paper_id: refreshed)
    monkeypatch.setattr(graph, "index_paper", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph,
        "index_is_current",
        lambda paper: paper.get("pdf_sha256") == "verified-pdf-hash",
    )
    monkeypatch.setattr(
        graph,
        "retrieve",
        lambda paper_id, query, top_k: [
            {
                "arxiv_id": paper_id,
                "versioned_id": "2501.00001v2",
                "page": 1,
                "section": "Method",
                "chunk_index": 0,
                "text": "Freshly indexed evidence.",
                "score": 0.9,
                "retrieval_score": 0.03,
            }
        ],
    )
    state = {
        "user_query": "What does the paper claim?",
        "candidate_papers": [stale],
        "selected_papers": [],
        "failed_papers": [],
        "tool_errors": [],
        "iteration_count": 0,
    }

    updated = graph.index_next(state)
    evidence = graph.retrieve_evidence({**state, **updated})["retrieved_chunks"]

    assert updated["candidate_papers"][0]["pdf_sha256"] == "verified-pdf-hash"
    assert evidence[0]["text"] == "Freshly indexed evidence."
