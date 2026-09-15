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


@dataclass(frozen=True)
class AnswerSynthesisRun:
    answer: str
    raw_answer: str
    model_calls: int
    citation_repair_count: int = 0
    citation_repair_error: str | None = None


PAPER_REFERENCE_PATTERN = re.compile(r"\[(?:\d+(?:\s*,\s*\d+)*)\]")


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


def _evidence_excerpts(evidence: list[dict], papers: dict[str, dict]) -> list[str]:
    excerpts = []
    for number, item in enumerate(evidence, start=1):
        paper = papers.get(item["arxiv_id"], {})
        source_id = item.get("versioned_id") or paper.get("versioned_id") or item["arxiv_id"]
        location = f"p.{item['page']}, {item['section']}" if item.get("page") else "Abstract"
        label = f"[{number}] arXiv:{source_id}, {location}"
        text = PAPER_REFERENCE_PATTERN.sub("(paper reference omitted)", item["text"])
        excerpts.append(f"{label}\nTitle: {paper.get('title', 'Unknown')}\n{text}")
    return excerpts


def _citation_free_text(answer: str) -> str:
    normalized = " ".join(PAPER_REFERENCE_PATTERN.sub("", answer).split())
    return re.sub(r"\s+([.,;:!?])", r"\1", normalized)


def _answer_prompt(question: str, excerpts: list[str]) -> str:
    labels = ", ".join(f"[{number}]" for number in range(1, len(excerpts) + 1))
    return f"""You are a careful scientific research assistant.
Answer the question using ONLY the evidence below. Every substantive scientific claim
must end with one or more matching citation labels such as [1] or [2]. Keep papers
separate when comparing them. For a multi-paper comparison, write one citation-complete
sentence for each paper. Do not add an uncited opening or closing comparison summary. A
single claim that names or contrasts multiple papers must carry supporting labels from every
paper it mentions. If evidence is incomplete or conflicting, say so plainly.
Do not invent bibliographic details, results, page numbers, or citations.
For multi-part comparisons, address only dimensions explicitly supported for each paper.
Do not infer a training objective, loss, directionality, or model capability merely from
an architecture description; state it only when a cited passage directly supports it.
The only allowed citation labels are: {labels}. Numeric references from the original paper
have been replaced with "(paper reference omitted)" and are never answer citation labels.

Question:
{question}

Evidence:
{chr(10).join(excerpts)}

Return only the concise cited answer. Do not write a bibliography or Sources section;
the application adds it from verified metadata.
"""


def _citation_repair_prompt(raw_answer: str, excerpts: list[str]) -> str:
    labels = ", ".join(f"[{number}]" for number in range(1, len(excerpts) + 1))
    return f"""Repair only the numeric citation labels in the answer below.

Keep every non-citation word, number, equation, and punctuation mark unchanged. Add, remove,
or replace only citation tokens such as [1]. Every substantive scientific claim must end in
the matching evidence label. The only allowed labels are: {labels}. Never copy a paper's
internal bibliography references. Return only the same answer with corrected citation labels,
without a Sources section, bibliography, Markdown fence, or commentary.

Answer:
{raw_answer}

Evidence:
{chr(10).join(excerpts)}
"""


def answer_from_evidence_bounded(
    question: str, evidence: list[dict], papers: dict[str, dict]
) -> AnswerSynthesisRun:
    """Synthesize once and permit one text-invariant citation-only repair."""
    if not evidence:
        answer = "Insufficient evidence: no relevant indexed passages were retrieved."
        return AnswerSynthesisRun(answer=answer, raw_answer=answer, model_calls=0)
    excerpts = _evidence_excerpts(evidence, papers)
    raw = str(get_llm().invoke(_answer_prompt(question, excerpts)).content).split(
        "\nSources:", 1
    )[0].strip()
    formatted = format_verified_sources(raw, evidence, papers)
    if formatted not in {MISSING_CITATION_MESSAGE, INVALID_CITATION_MESSAGE}:
        return AnswerSynthesisRun(answer=formatted, raw_answer=raw, model_calls=1)

    repaired = str(
        get_llm(temperature=0, num_predict=1000).invoke(
            _citation_repair_prompt(raw, excerpts)
        ).content
    ).split("\nSources:", 1)[0].strip()
    if _citation_free_text(repaired) != _citation_free_text(raw):
        return AnswerSynthesisRun(
            answer=formatted,
            raw_answer=raw,
            model_calls=2,
            citation_repair_count=1,
            citation_repair_error="Citation repair changed non-citation answer text.",
        )
    repaired_formatted = format_verified_sources(repaired, evidence, papers)
    repair_error = None
    if repaired_formatted in {MISSING_CITATION_MESSAGE, INVALID_CITATION_MESSAGE}:
        repair_error = repaired_formatted
    return AnswerSynthesisRun(
        answer=repaired_formatted,
        raw_answer=raw,
        model_calls=2,
        citation_repair_count=1,
        citation_repair_error=repair_error,
    )


def answer_from_evidence(question: str, evidence: list[dict], papers: dict[str, dict]) -> str:
    """Compatibility wrapper returning only the bounded synthesis answer."""
    return answer_from_evidence_bounded(question, evidence, papers).answer


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
