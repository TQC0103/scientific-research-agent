from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.config import settings
from app.db.database import get_paper
from app.ingestion.indexing import index_paper
from app.models.llm import answer_from_evidence
from app.retrieval.vector_store import index_directory, retrieve
from app.tools.arxiv_search import get_arxiv_metadata, search_arxiv

AUTO_INDEX_LIMIT = 2


def discover(state: AgentState) -> dict:
    explicit = state.get("paper_ids", [])
    if explicit:
        candidates = [get_paper(pid) or get_arxiv_metadata(pid) for pid in explicit]
    else:
        candidates = search_arxiv(state["user_query"], max_results=5)
    return {"candidate_papers": candidates, "selected_papers": [], "iteration_count": 0}


def index_next(state: AgentState) -> dict:
    selected = list(state.get("selected_papers", []))
    remaining = [p["arxiv_id"] for p in state["candidate_papers"] if p["arxiv_id"] not in selected]
    if not remaining:
        return {}
    paper_id = remaining[0]
    if not (index_directory(paper_id) / "index.faiss").exists():
        index_paper(paper_id)
    selected.append(paper_id)
    return {"selected_papers": selected, "iteration_count": state.get("iteration_count", 0) + 1}


def retrieve_evidence(state: AgentState) -> dict:
    evidence = []
    for paper_id in state.get("selected_papers", []):
        evidence.extend(retrieve(paper_id, state["user_query"], top_k=4))
    evidence.sort(key=lambda item: item["score"], reverse=True)
    return {"retrieved_chunks": evidence[:8]}


def check_evidence(state: AgentState) -> dict:
    evidence = state.get("retrieved_chunks", [])
    enough = len(evidence) >= 3 and max((e["score"] for e in evidence), default=0) >= 0.30
    if len(state.get("selected_papers", [])) >= AUTO_INDEX_LIMIT:
        enough = True
    if state.get("iteration_count", 0) >= settings.max_tool_loops:
        enough = True
    return {"evidence_sufficient": enough}


def route_after_check(state: AgentState) -> str:
    remaining = len(state.get("candidate_papers", [])) - len(state.get("selected_papers", []))
    return "synthesize" if state.get("evidence_sufficient") or remaining <= 0 else "index_next"


def synthesize(state: AgentState) -> dict:
    papers = {p["arxiv_id"]: p for p in state.get("candidate_papers", [])}
    answer = answer_from_evidence(state["user_query"], state.get("retrieved_chunks", []), papers)
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

