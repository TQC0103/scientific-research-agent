from typing import TypedDict


class AgentState(TypedDict, total=False):
    user_query: str
    paper_ids: list[str]
    candidate_papers: list[dict]
    selected_papers: list[str]
    retrieved_chunks: list[dict]
    evidence_sufficient: bool
    answer: str
    iteration_count: int
    discovery_source: str
    failed_papers: list[str]
    tool_errors: list[str]
