from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.config import settings
from app.db.database import search_local
from app.ingestion.indexing import index_paper
from app.models.llm import answer_from_evidence
from app.models.verifier import verify_evidence
from app.retrieval.vector_store import index_is_current, retrieve
from app.tools.arxiv_search import get_arxiv_metadata, search_arxiv
from app.tools.paper_download import PaperDownloadError

AUTO_INDEX_LIMIT = 2
MIN_LOCAL_CANDIDATES = 3


def _merge_candidates(*groups: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for group in groups:
        for paper in group:
            merged[paper["arxiv_id"]] = paper
    return list(merged.values())


def discover(state: AgentState) -> dict:
    explicit = state.get("paper_ids", [])
    if explicit:
        candidates = [get_arxiv_metadata(pid) for pid in explicit]
        source = "explicit_arxiv_ids"
    else:
        local = search_local(state["user_query"], limit=5)
        if len(local) >= MIN_LOCAL_CANDIDATES:
            candidates = local
            source = "sqlite_fts5"
        else:
            remote = search_arxiv(state["user_query"], max_results=5)
            candidates = _merge_candidates(local, remote)
            source = "sqlite_fts5+arxiv" if local else "arxiv"
    return {
        "candidate_papers": candidates,
        "selected_papers": [],
        "failed_papers": [],
        "tool_errors": [],
        "iteration_count": 0,
        "retrieval_query": state["user_query"],
        "retrieval_attempt_count": 0,
        "evidence_sufficient": False,
        "discovery_source": source,
    }


def index_next(state: AgentState) -> dict:
    selected = list(state.get("selected_papers", []))
    failed = list(state.get("failed_papers", []))
    errors = list(state.get("tool_errors", []))
    remaining = [p["arxiv_id"] for p in state["candidate_papers"] if p["arxiv_id"] not in selected]
    if not remaining:
        return {}

    paper_id = remaining[0]
    selected.append(paper_id)
    try:
        paper = get_arxiv_metadata(paper_id)
        candidates = [
            paper if item["arxiv_id"] == paper_id else item for item in state["candidate_papers"]
        ]
        if not index_is_current(paper):
            index_paper(paper_id, paper=paper)
    except (PaperDownloadError, ValueError, OSError) as exc:
        failed.append(paper_id)
        errors.append(f"{paper_id}: {exc}")
        candidates = state["candidate_papers"]
    return {
        "candidate_papers": candidates,
        "selected_papers": selected,
        "failed_papers": failed,
        "tool_errors": errors,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def retrieve_evidence(state: AgentState) -> dict:
    papers = {paper["arxiv_id"]: paper for paper in state.get("candidate_papers", [])}
    failed = set(state.get("failed_papers", []))
    evidence = []
    retrieval_query = state.get("retrieval_query") or state["user_query"]
    for paper_id in state.get("selected_papers", []):
        paper = papers[paper_id]
        if paper_id not in failed and index_is_current(paper):
            evidence.extend(retrieve(paper_id, retrieval_query, top_k=6))
        elif paper.get("abstract"):
            evidence.append(
                {
                    "arxiv_id": paper_id,
                    "versioned_id": paper.get("versioned_id") or paper_id,
                    "page": None,
                    "section": "Abstract",
                    "chunk_index": 0,
                    "text": paper["abstract"],
                    "score": 0.15,
                    "retrieval_score": 0.0,
                }
            )
    evidence.sort(key=lambda item: item.get("retrieval_score", item["score"]), reverse=True)
    # Keep the newest retrieval first, then retain distinct passages from earlier
    # attempts so the verifier can combine complementary evidence.
    combined = evidence[:8] + state.get("retrieved_chunks", [])
    unique = []
    seen = set()
    for item in combined:
        key = (item["arxiv_id"], item.get("chunk_index"), item.get("page"), item["text"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return {"retrieved_chunks": unique[:12]}


def check_evidence(state: AgentState) -> dict:
    evidence = state.get("retrieved_chunks", [])
    attempts = state.get("retrieval_attempt_count", 0) + 1
    errors = list(state.get("tool_errors", []))
    verifier_failed = False
    try:
        result = verify_evidence(
            state["user_query"], evidence, state.get("retrieval_query") or state["user_query"]
        )
        verification = result.model_dump()
    except (ValueError, OSError) as exc:
        verifier_failed = True
        verification = {
            "sufficient": False,
            "reason": "The local evidence verifier failed, so the system failed closed.",
            "missing_information": ["A successful evidence verification."],
            "suggested_query": state["user_query"],
            "supported_evidence": [],
        }
        errors.append(f"verifier: {exc}")

    proposed = (verification.get("suggested_query") or "").strip()
    current = (state.get("retrieval_query") or state["user_query"]).strip()
    if not verification["sufficient"] and (not proposed or proposed.casefold() == current.casefold()):
        missing = " ".join(verification.get("missing_information") or []).strip()
        proposed = (
            f"{missing} specific mechanism terminology section table".strip()
            if missing
            else f"{current} specific evidence section table"
        )
    can_rewrite = (
        not verification["sufficient"]
        and not verifier_failed
        and attempts <= settings.max_retrieval_rewrites
        and bool(proposed)
        and proposed.casefold() != current.casefold()
    )
    return {
        "evidence_sufficient": verification["sufficient"],
        "evidence_verification": verification,
        "retrieval_query": proposed if can_rewrite else current,
        "retrieval_attempt_count": attempts,
        "should_retry_retrieval": can_rewrite,
        "tool_errors": errors,
    }


def route_after_check(state: AgentState) -> str:
    if state.get("evidence_sufficient"):
        return "synthesize"
    if state.get("should_retry_retrieval"):
        return "retrieve"
    remaining = len(state.get("candidate_papers", [])) - len(state.get("selected_papers", []))
    can_index = (
        remaining > 0
        and len(state.get("selected_papers", [])) < AUTO_INDEX_LIMIT
        and state.get("iteration_count", 0) < settings.max_tool_loops
    )
    return "index_next" if can_index else "synthesize"


def synthesize(state: AgentState) -> dict:
    papers = {paper["arxiv_id"]: paper for paper in state.get("candidate_papers", [])}
    verification = state.get("evidence_verification", {})
    if not state.get("evidence_sufficient"):
        missing = verification.get("missing_information") or ["Direct supporting evidence."]
        answer = "Insufficient evidence to answer without guessing."
        answer += f"\n\nReason: {verification.get('reason', 'The retrieved passages were insufficient.')}"
        answer += "\n\nMissing:\n- " + "\n- ".join(missing)
    else:
        evidence = state.get("retrieved_chunks", [])
        supported = verification.get("supported_evidence", [])
        verified_evidence = [evidence[number - 1] for number in supported]
        answer = answer_from_evidence(state["user_query"], verified_evidence, papers)
    if state.get("tool_errors"):
        answer += "\n\nRetrieval notes:\n- " + "\n- ".join(state["tool_errors"])
    return {"answer": answer}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("discover", discover)
    graph.add_node("index_next", index_next)
    graph.add_node("retrieve", retrieve_evidence)
    graph.add_node("check", check_evidence)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "discover")
    graph.add_edge("discover", "index_next")
    graph.add_edge("index_next", "retrieve")
    graph.add_edge("retrieve", "check")
    graph.add_conditional_edges(
        "check",
        route_after_check,
        {"retrieve": "retrieve", "index_next": "index_next", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


research_graph = build_graph()
