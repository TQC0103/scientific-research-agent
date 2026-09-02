import re

from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.config import settings
from app.db.database import search_local
from app.ingestion.indexing import index_paper
from app.models.claim_verifier import (
    ClaimVerificationRunError,
    repair_answer_claims,
    verify_answer_claims_bounded,
)
from app.models.claims import ClaimVerdict, ClaimVerificationBundle
from app.models.llm import answer_from_evidence
from app.models.verifier import verify_evidence
from app.retrieval.vector_store import index_is_current, retrieve
from app.tools.arxiv_search import get_arxiv_metadata, search_arxiv
from app.tools.paper_download import PaperDownloadError

AUTO_INDEX_LIMIT = 2
MIN_LOCAL_CANDIDATES = 3
MAX_CLAIM_REVISIONS = 1
MULTI_PAPER_PATTERN = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differences?|similarities?|two papers?)\b"
    r"|so sánh|khác nhau|giống nhau|hai (bài|paper)",
    re.IGNORECASE,
)


def _merge_candidates(*groups: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for group in groups:
        for paper in group:
            merged[paper["arxiv_id"]] = paper
    return list(merged.values())


def _requires_multi_paper(query: str) -> bool:
    return bool(MULTI_PAPER_PATTERN.search(query))


def _group_evidence(evidence: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for item in evidence:
        grouped.setdefault(item["arxiv_id"], []).append(item)
    return grouped


def _merge_passages(
    new: list[dict], previous: list[dict], *, limit: int | None = None
) -> list[dict]:
    effective_limit = settings.max_accumulated_passages_per_paper if limit is None else limit
    unique = []
    seen = set()
    for item in new[:8] + previous:
        key = (item["arxiv_id"], item.get("chunk_index"), item.get("page"), item["text"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:effective_limit]


def _coverage_sufficient(state: AgentState, verifications: dict[str, dict]) -> bool:
    def complete(paper_id: str) -> bool:
        verification = verifications.get(paper_id, {})
        return bool(
            verification.get("sufficient")
            and verification.get("supported_evidence")
            and not verification.get("missing_information")
        )

    if state.get("coverage_mode") == "all":
        required = state.get("required_paper_ids", [])
        required_count = state.get("required_paper_count", len(required))
        return bool(
            len(required) >= required_count and all(complete(paper_id) for paper_id in required)
        )
    return any(complete(paper_id) for paper_id in state.get("selected_papers", []))


def discover(state: AgentState) -> dict:
    explicit = state.get("paper_ids", [])
    if explicit:
        candidates = _merge_candidates([get_arxiv_metadata(pid) for pid in explicit])
        source = "explicit_arxiv_ids"
        coverage_mode = "all"
        required_paper_ids = [paper["arxiv_id"] for paper in candidates]
        required_paper_count = len(required_paper_ids)
    else:
        local = search_local(state["user_query"], limit=5)
        if len(local) >= MIN_LOCAL_CANDIDATES:
            candidates = local
            source = "sqlite_fts5"
        else:
            remote = search_arxiv(state["user_query"], max_results=5)
            candidates = _merge_candidates(local, remote)
            source = "sqlite_fts5+arxiv" if local else "arxiv"
        if _requires_multi_paper(state["user_query"]):
            coverage_mode = "all"
            required_paper_ids = [paper["arxiv_id"] for paper in candidates[:AUTO_INDEX_LIMIT]]
            required_paper_count = AUTO_INDEX_LIMIT
        else:
            coverage_mode = "any"
            required_paper_ids = []
            required_paper_count = 1
    return {
        "candidate_papers": candidates,
        "selected_papers": [],
        "required_paper_ids": required_paper_ids,
        "required_paper_count": required_paper_count,
        "coverage_mode": coverage_mode,
        "failed_papers": [],
        "tool_errors": [],
        "iteration_count": 0,
        "retrieval_query": state["user_query"],
        "retrieval_queries": {},
        "retrieval_attempt_count": 0,
        "retrieval_attempt_counts": {},
        "retrieved_chunks": [],
        "retrieved_chunks_by_paper": {},
        "evidence_sufficient": False,
        "evidence_verifications": {},
        "verified_evidence": [],
        "synthesis_citation_valid": False,
        "claim_verification": {},
        "claim_verification_status": "not_run",
        "claim_verification_error": None,
        "claim_verification_attempt_count": 0,
        "claim_revision_count": 0,
        "claim_revision_history": [],
        "papers_to_retrieve": [],
        "discovery_source": source,
    }


def index_next(state: AgentState) -> dict:
    selected = list(state.get("selected_papers", []))
    failed = list(state.get("failed_papers", []))
    errors = list(state.get("tool_errors", []))
    required_remaining = [
        paper_id for paper_id in state.get("required_paper_ids", []) if paper_id not in selected
    ]
    all_remaining = [
        paper["arxiv_id"]
        for paper in state.get("candidate_papers", [])
        if paper["arxiv_id"] not in selected
    ]
    if not required_remaining and not all_remaining:
        return {"papers_to_retrieve": []}

    paper_id = (required_remaining or all_remaining)[0]
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
        "papers_to_retrieve": [paper_id],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def retrieve_evidence(state: AgentState) -> dict:
    papers = {paper["arxiv_id"]: paper for paper in state.get("candidate_papers", [])}
    failed = set(state.get("failed_papers", []))
    queries = dict(state.get("retrieval_queries", {}))
    by_paper = {
        paper_id: list(items)
        for paper_id, items in state.get("retrieved_chunks_by_paper", {}).items()
    }
    if not by_paper and state.get("retrieved_chunks"):
        by_paper = _group_evidence(state["retrieved_chunks"])
    targets = state.get("papers_to_retrieve") or state.get("selected_papers", [])[-1:]

    for paper_id in targets:
        paper = papers[paper_id]
        query = queries.get(paper_id, state["user_query"])
        evidence = []
        if paper_id not in failed and index_is_current(paper):
            evidence = retrieve(paper_id, query, top_k=6)
        elif paper.get("abstract"):
            evidence = [
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
            ]
        evidence.sort(key=lambda item: item.get("retrieval_score", item["score"]), reverse=True)
        by_paper[paper_id] = _merge_passages(evidence, by_paper.get(paper_id, []))
        queries.setdefault(paper_id, query)

    flattened = [
        item for paper_id in state.get("selected_papers", []) for item in by_paper.get(paper_id, [])
    ]
    last_query = queries.get(targets[-1], state["user_query"]) if targets else state["user_query"]
    return {
        "retrieved_chunks": flattened,
        "retrieved_chunks_by_paper": by_paper,
        "retrieval_queries": queries,
        "retrieval_query": last_query,
    }


def check_evidence(state: AgentState) -> dict:
    by_paper = state.get("retrieved_chunks_by_paper") or _group_evidence(
        state.get("retrieved_chunks", [])
    )
    targets = state.get("papers_to_retrieve") or list(by_paper)
    queries = dict(state.get("retrieval_queries", {}))
    attempts_by_paper = dict(state.get("retrieval_attempt_counts", {}))
    verifications = dict(state.get("evidence_verifications", {}))
    errors = list(state.get("tool_errors", []))
    retry_papers = []
    papers = {paper["arxiv_id"]: paper for paper in state.get("candidate_papers", [])}
    multi_paper = state.get("coverage_mode") == "all" and state.get("required_paper_count", 1) > 1

    for paper_id in targets:
        evidence = by_paper.get(paper_id, [])
        current = queries.get(paper_id, state["user_query"]).strip()
        attempts = attempts_by_paper.get(paper_id, 0) + 1
        verifier_failed = False
        paper = papers.get(paper_id, {})
        scope = None
        if multi_paper:
            scope = (
                f"Assess coverage only for arXiv:{paper_id} "
                f"({paper.get('title', 'Unknown title')}). Decide whether this paper supplies "
                "enough evidence for its own side of the multi-paper question. Every requested "
                "comparison dimension about this paper must have direct passage support. Do not "
                "require passages about the other papers."
            )
            verification_question = (
                f"For arXiv:{paper_id} only, do these passages provide this paper's own "
                "information for every requested comparison dimension? If even one dimension "
                "is absent, mark insufficient and target it in the new query. Ignore all missing "
                f"information about other papers. Original comparison: {state['user_query']}"
            )
        else:
            verification_question = state["user_query"]
        try:
            result = verify_evidence(verification_question, evidence, current, scope)
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
            errors.append(f"verifier {paper_id}: {exc}")

        if verification.get("sufficient") and (
            verification.get("missing_information") or not verification.get("supported_evidence")
        ):
            verification["sufficient"] = False
            verification["missing_information"] = verification.get("missing_information") or [
                "A directly supporting passage."
            ]
            verification["suggested_query"] = (
                verification.get("suggested_query") or state["user_query"]
            )

        proposed = (verification.get("suggested_query") or "").strip()
        if not verification["sufficient"] and (
            not proposed or proposed.casefold() == current.casefold()
        ):
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
        queries[paper_id] = proposed if can_rewrite else current
        attempts_by_paper[paper_id] = attempts
        verification["retrieval_attempts"] = attempts
        verification["final_retrieval_query"] = queries[paper_id]
        verifications[paper_id] = verification
        if can_rewrite:
            retry_papers.append(paper_id)

    enough = _coverage_sufficient(state, verifications)
    aggregate = {
        "sufficient": enough,
        "coverage_mode": state.get("coverage_mode", "any"),
        "required_paper_ids": state.get("required_paper_ids", []),
        "required_paper_count": state.get("required_paper_count", 1),
        "by_paper": verifications,
    }
    last_target = targets[-1] if targets else None
    return {
        "evidence_sufficient": enough,
        "evidence_verification": aggregate,
        "evidence_verifications": verifications,
        "retrieval_queries": queries,
        "retrieval_query": queries.get(last_target, state["user_query"]),
        "retrieval_attempt_counts": attempts_by_paper,
        "retrieval_attempt_count": max(attempts_by_paper.values(), default=0),
        "papers_to_retrieve": retry_papers,
        "should_retry_retrieval": bool(retry_papers),
        "tool_errors": errors,
    }


def route_after_check(state: AgentState) -> str:
    if state.get("evidence_sufficient"):
        return "synthesize"
    if state.get("papers_to_retrieve"):
        return "retrieve"

    selected = set(state.get("selected_papers", []))
    required_remaining = [
        paper_id for paper_id in state.get("required_paper_ids", []) if paper_id not in selected
    ]
    within_tool_limit = state.get("iteration_count", 0) < settings.max_tool_loops
    if required_remaining and within_tool_limit:
        return "index_next"

    all_remaining = [
        paper["arxiv_id"]
        for paper in state.get("candidate_papers", [])
        if paper["arxiv_id"] not in selected
    ]
    can_try_optional = (
        state.get("coverage_mode") == "any"
        and bool(all_remaining)
        and len(selected) < AUTO_INDEX_LIMIT
        and within_tool_limit
    )
    return "index_next" if can_try_optional else "synthesize"


def synthesize(state: AgentState) -> dict:
    papers = {paper["arxiv_id"]: paper for paper in state.get("candidate_papers", [])}
    verifications = state.get("evidence_verifications", {})
    by_paper = state.get("retrieved_chunks_by_paper") or _group_evidence(
        state.get("retrieved_chunks", [])
    )
    verified_evidence = []
    evidence_sufficient = bool(
        state.get("evidence_sufficient") and _coverage_sufficient(state, verifications)
    )
    if not evidence_sufficient:
        answer = "Insufficient evidence to answer without guessing."
        details = []
        relevant = state.get("required_paper_ids") or state.get("selected_papers", [])
        required_count = state.get("required_paper_count", len(relevant))
        if state.get("coverage_mode") == "all" and len(relevant) < required_count:
            details.append(
                f"Required coverage from {required_count} papers, but only "
                f"{len(relevant)} candidate paper(s) were available."
            )
        for paper_id in relevant:
            verification = verifications.get(paper_id)
            if not verification:
                details.append(f"arXiv:{paper_id}: evidence was not verified.")
                continue
            if (
                verification.get("sufficient")
                and verification.get("supported_evidence")
                and not verification.get("missing_information")
            ):
                continue
            missing = verification.get("missing_information") or ["Direct supporting evidence."]
            details.append(
                f"arXiv:{paper_id}: {verification.get('reason', 'Evidence was insufficient.')} "
                f"Missing: {'; '.join(missing)}"
            )
        answer += "\n\nCoverage gaps:\n- " + "\n- ".join(
            details or ["No paper had sufficient verified evidence."]
        )
    else:
        for paper_id in state.get("selected_papers", []):
            verification = verifications.get(paper_id, {})
            if not (
                verification.get("sufficient")
                and verification.get("supported_evidence")
                and not verification.get("missing_information")
            ):
                continue
            evidence = by_paper.get(paper_id, [])
            verified_evidence.extend(
                evidence[number - 1]
                for number in verification.get("supported_evidence", [])
                if 1 <= number <= len(evidence)
            )
        answer = answer_from_evidence(state["user_query"], verified_evidence, papers)
    if state.get("tool_errors"):
        answer += "\n\nRetrieval notes:\n- " + "\n- ".join(state["tool_errors"])
    citation_valid = bool(evidence_sufficient and verified_evidence and "\n\nSources:\n" in answer)
    return {
        "evidence_sufficient": evidence_sufficient,
        "answer": answer,
        "verified_evidence": verified_evidence,
        "synthesis_citation_valid": citation_valid,
        "claim_verification": {},
        "claim_verification_status": "not_run",
        "claim_verification_error": None,
        "claim_verification_attempt_count": 0,
        "claim_revision_count": 0,
        "claim_revision_history": [answer] if evidence_sufficient else [],
    }


def route_after_synthesis(state: AgentState) -> str:
    if not state.get("evidence_sufficient"):
        return "end"
    return "verify_claims" if state.get("synthesis_citation_valid") else "abstain"


def _claim_verification_status(bundle: ClaimVerificationBundle) -> str:
    verdicts = [
        assessment.verdict
        for assessment in bundle.assessments
        if assessment.verdict != ClaimVerdict.NOT_REQUIRED
    ]
    if not verdicts or all(verdict == ClaimVerdict.SUPPORTED for verdict in verdicts):
        return "verified"
    if ClaimVerdict.PARTIAL in verdicts:
        return "repairable"
    if ClaimVerdict.SUPPORTED in verdicts and ClaimVerdict.UNSUPPORTED in verdicts:
        return "repairable"
    return "unsupported"


def verify_claims(state: AgentState) -> dict:
    attempts = state.get("claim_verification_attempt_count", 0)
    try:
        run = verify_answer_claims_bounded(
            state["answer"],
            state.get("verified_evidence", []),
            state["user_query"],
        )
        attempts += run.model_calls
        bundle = run.bundle
    except (ValueError, OSError) as exc:
        attempts += exc.model_calls if isinstance(exc, ClaimVerificationRunError) else 1
        return {
            "claim_verification": {},
            "claim_verification_status": "invalid",
            "claim_verification_error": str(exc),
            "claim_verification_attempt_count": attempts,
        }
    return {
        "claim_verification": bundle.model_dump(mode="json"),
        "claim_verification_status": _claim_verification_status(bundle),
        "claim_verification_error": None,
        "claim_verification_attempt_count": attempts,
    }


def route_after_claim_verification(state: AgentState) -> str:
    status = state.get("claim_verification_status")
    if status == "verified":
        return "end"
    if status == "repairable" and state.get("claim_revision_count", 0) < MAX_CLAIM_REVISIONS:
        return "revise"
    return "abstain"


def revise_answer(state: AgentState) -> dict:
    revisions = state.get("claim_revision_count", 0) + 1
    history = list(state.get("claim_revision_history", []))
    papers = {paper["arxiv_id"]: paper for paper in state.get("candidate_papers", [])}
    try:
        verification = ClaimVerificationBundle.model_validate(state.get("claim_verification", {}))
        answer = repair_answer_claims(
            state["user_query"],
            state["answer"],
            state.get("verified_evidence", []),
            papers,
            verification,
        )
        history.append(answer)
        return {
            "answer": answer,
            "synthesis_citation_valid": "\n\nSources:\n" in answer,
            "claim_revision_count": revisions,
            "claim_revision_history": history,
            "claim_verification_status": "not_run",
            "claim_verification_error": None,
        }
    except (ValueError, OSError) as exc:
        return {
            "claim_revision_count": revisions,
            "claim_revision_history": history,
            "claim_verification_status": "invalid",
            "claim_verification_error": str(exc),
        }


def route_after_revision(state: AgentState) -> str:
    if state.get("claim_verification_status") == "invalid" or not state.get(
        "synthesis_citation_valid"
    ):
        return "abstain"
    return "verify_claims"


def abstain_on_claims(state: AgentState) -> dict:
    answer = (
        "Unable to provide a fully citation-grounded answer: claim-level verification "
        "could not establish every substantive claim from the approved evidence."
    )
    if state.get("claim_revision_count", 0):
        answer += " One bounded revision was attempted."
    return {
        "answer": answer,
        "claim_verification_status": "abstained",
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("discover", discover)
    graph.add_node("index_next", index_next)
    graph.add_node("retrieve", retrieve_evidence)
    graph.add_node("check", check_evidence)
    graph.add_node("synthesize", synthesize)
    graph.add_node("verify_claims", verify_claims)
    graph.add_node("revise", revise_answer)
    graph.add_node("abstain", abstain_on_claims)
    graph.add_edge(START, "discover")
    graph.add_edge("discover", "index_next")
    graph.add_edge("index_next", "retrieve")
    graph.add_edge("retrieve", "check")
    graph.add_conditional_edges(
        "check",
        route_after_check,
        {"retrieve": "retrieve", "index_next": "index_next", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "synthesize",
        route_after_synthesis,
        {"verify_claims": "verify_claims", "abstain": "abstain", "end": END},
    )
    graph.add_conditional_edges(
        "verify_claims",
        route_after_claim_verification,
        {"revise": "revise", "abstain": "abstain", "end": END},
    )
    graph.add_conditional_edges(
        "revise",
        route_after_revision,
        {"verify_claims": "verify_claims", "abstain": "abstain"},
    )
    graph.add_edge("abstain", END)
    return graph.compile()


research_graph = build_graph()
