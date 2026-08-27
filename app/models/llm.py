import re
from dataclasses import dataclass

from langchain_ollama import ChatOllama

from app.config import settings

MISSING_CITATION_MESSAGE = (
    "Unable to provide a citation-grounded answer: synthesis did not include a valid "
    "verified citation."
)
INVALID_CITATION_MESSAGE = (
    "Unable to provide a citation-grounded answer: synthesis referenced an invalid "
    "citation label."
)


@dataclass(frozen=True)
class CitationLabels:
    valid: tuple[int, ...]
    invalid: tuple[int, ...]


def parse_citation_labels(answer: str, evidence_count: int) -> CitationLabels:
    """Resolve numeric labels without inventing or silently discarding citations."""
    valid = []
    invalid = []
    for match in re.finditer(r"\[(\d+)]", answer):
        number = int(match.group(1))
        target = valid if 1 <= number <= evidence_count else invalid
        if number not in target:
            target.append(number)
    return CitationLabels(valid=tuple(valid), invalid=tuple(invalid))


def get_llm(*, temperature: float = 0.1, num_predict: int = 1000, seed: int = 42) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=16384,
        num_predict=num_predict,
        seed=seed,
    )


def answer_from_evidence(question: str, evidence: list[dict], papers: dict[str, dict]) -> str:
    if not evidence:
        return "Insufficient evidence: no relevant indexed passages were retrieved."
    excerpts = []
    for number, item in enumerate(evidence, start=1):
        paper = papers.get(item["arxiv_id"], {})
        source_id = item.get("versioned_id") or paper.get("versioned_id") or item["arxiv_id"]
        location = f"p.{item['page']}, {item['section']}" if item.get("page") else "Abstract"
        label = f"[{number}] arXiv:{source_id}, {location}"
        excerpts.append(f"{label}\nTitle: {paper.get('title', 'Unknown')}\n{item['text']}")
    prompt = f"""You are a careful scientific research assistant.
Answer the question using ONLY the evidence below. Every substantive scientific claim
must end with one or more matching citation labels such as [1] or [2]. Keep papers
separate when comparing them. If evidence is incomplete or conflicting, say so plainly.
Do not invent bibliographic details, results, page numbers, or citations.
For multi-part comparisons, address only dimensions explicitly supported for each paper.
Do not infer a training objective, loss, directionality, or model capability merely from
an architecture description; state it only when a cited passage directly supports it.

Question:
{question}

Evidence:
{chr(10).join(excerpts)}

Return only the concise cited answer. Do not write a bibliography or Sources section;
the application adds it from verified metadata.
"""
    raw = str(get_llm().invoke(prompt).content).split("\nSources:", 1)[0].strip()
    return format_verified_sources(raw, evidence, papers)


def format_verified_sources(answer: str, evidence: list[dict], papers: dict[str, dict]) -> str:
    """Append source details from trusted chunk metadata, never model-generated text."""
    citations = parse_citation_labels(answer, len(evidence))
    if citations.invalid:
        return INVALID_CITATION_MESSAGE
    if not citations.valid:
        return MISSING_CITATION_MESSAGE
    lines = []
    for number in citations.valid:
        item = evidence[number - 1]
        paper = papers.get(item["arxiv_id"], {})
        source_id = item.get("versioned_id") or paper.get("versioned_id") or item["arxiv_id"]
        location = f"p.{item['page']}, {item['section']}" if item.get("page") else "Abstract"
        lines.append(f"[{number}] arXiv:{source_id} — {paper.get('title', 'Unknown')} — {location}")
    return f"{answer}\n\nSources:\n" + "\n".join(lines)
