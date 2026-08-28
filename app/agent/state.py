from typing import TypedDict


class AgentState(TypedDict, total=False):
    user_query: str
    paper_ids: list[str]
    candidate_papers: list[dict]
    selected_papers: list[str]
    retrieved_chunks: list[dict]
    retrieved_chunks_by_paper: dict[str, list[dict]]
    evidence_sufficient: bool
    evidence_verification: dict
    evidence_verifications: dict[str, dict]
    retrieval_query: str
    retrieval_queries: dict[str, str]
    retrieval_attempt_count: int
    retrieval_attempt_counts: dict[str, int]
    should_retry_retrieval: bool
    papers_to_retrieve: list[str]
    required_paper_ids: list[str]
    required_paper_count: int
    coverage_mode: str
    answer: str
    verified_evidence: list[dict]
    synthesis_citation_valid: bool
    claim_verification: dict
    claim_verification_status: str
    claim_verification_error: str | None
    claim_verification_attempt_count: int
    claim_revision_count: int
    claim_revision_history: list[str]
    iteration_count: int
    discovery_source: str
    failed_papers: list[str]
    tool_errors: list[str]
