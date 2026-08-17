from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.config import settings
from app.db.database import search_local
from app.ingestion.indexing import index_paper
from app.models.llm import answer_from_evidence
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
    for paper_id in state.get("selected_papers", []):
        paper = papers[paper_id]
        if paper_id not in failed and index_is_current(paper):
            evidence.extend(retrieve(paper_id, state["user_query"], top_k=4))
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
                }
            )
    evidence.sort(key=lambda item: item["score"], reverse=True)
    return {"retrieved_chunks": evidence[:8]}


def check_evidence(state: AgentState) -> dict:
    evidence = state.get("retrieved_chunks", [])
    full_text = [item for item in evidence if item.get("page") is not None]
    enough = len(full_text) >= 3 and max((item["score"] for item in full_text), default=0) >= 0.30
    if len(state.get("selected_papers", [])) >= AUTO_INDEX_LIMIT:
        enough = True
    if state.get("iteration_count", 0) >= settings.max_tool_loops:
        enough = True
    return {"evidence_sufficient": enough}


def route_after_check(state: AgentState) -> str:
    remaining = len(state.get("candidate_papers", [])) - len(state.get("selected_papers", []))
    return "synthesize" if state.get("evidence_sufficient") or remaining <= 0 else "index_next"


def synthesize(state: AgentState) -> dict:
    papers = {paper["arxiv_id"]: paper for paper in state.get("candidate_papers", [])}
    answer = answer_from_evidence(state["user_query"], state.get("retrieved_chunks", []), papers)
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
        "check", route_after_check, {"index_next": "index_next", "synthesize": "synthesize"}
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


research_graph = build_graph()
