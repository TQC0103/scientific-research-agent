from app.agent import graph
from app.models.planner import QueryPlan
from app.tools.paper_download import PdfUnavailableError


def _plan(mode: str = "single_paper") -> QueryPlan:
    return QueryPlan(
        mode=mode,
        search_query="planned scientific query",
        required_paper_count=2 if mode == "multi_paper" else 1,
        comparison_dimensions=["architecture"] if mode == "multi_paper" else [],
        rationale="Test plan.",
    )


def test_discover_uses_local_fts_without_remote_when_catalog_is_sufficient(monkeypatch) -> None:
    local = [
        {"arxiv_id": f"2501.0000{number}", "title": f"Paper {number}", "abstract": "x"}
        for number in range(1, 4)
    ]
    monkeypatch.setattr(graph, "search_local", lambda query, limit: local)
    monkeypatch.setattr(graph, "plan_query", lambda *args, **kwargs: _plan())

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
        "plan_query",
        lambda *args, **kwargs: QueryPlan(
            mode="multi_paper",
            search_query="planned query",
            required_paper_count=2,
            comparison_dimensions=["method"],
            rationale="Test plan.",
        ),
    )
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
    monkeypatch.setattr(
        graph, "plan_query", lambda *args, **kwargs: _plan("multi_paper")
    )
    result = graph.discover({"user_query": "Compare these approaches", "paper_ids": []})
    assert result["coverage_mode"] == "all"
    assert result["required_paper_ids"] == ["2401.00001", "2401.00002"]
    assert result["required_paper_count"] == 2


def test_discovery_uses_planned_query_for_local_and_remote_search(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(graph, "plan_query", lambda *args, **kwargs: _plan())
    monkeypatch.setattr(
        graph, "search_local", lambda query, limit: calls.append(("local", query)) or []
    )
    monkeypatch.setattr(
        graph, "search_arxiv", lambda query, max_results: calls.append(("arxiv", query)) or []
    )

    result = graph.discover({"user_query": "verbose original request", "paper_ids": []})

    assert calls == [
        ("local", "planned scientific query"),
        ("arxiv", "planned scientific query"),
    ]
    assert result["query_plan"]["search_query"] == "planned scientific query"


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
