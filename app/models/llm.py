import re

from langchain_ollama import ChatOllama

from app.config import settings


def get_llm(*, temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        num_ctx=16384,
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
    cited = []
    for match in re.finditer(r"\[(\d+)]", answer):
        number = int(match.group(1))
        if 1 <= number <= len(evidence) and number not in cited:
            cited.append(number)
    if not cited and evidence:
        cited = [1]
        answer = f"{answer} [1]"
    lines = []
    for number in cited:
        item = evidence[number - 1]
        paper = papers.get(item["arxiv_id"], {})
        source_id = item.get("versioned_id") or paper.get("versioned_id") or item["arxiv_id"]
        location = f"p.{item['page']}, {item['section']}" if item.get("page") else "Abstract"
        lines.append(f"[{number}] arXiv:{source_id} — {paper.get('title', 'Unknown')} — {location}")
    return f"{answer}\n\nSources:\n" + "\n".join(lines)
