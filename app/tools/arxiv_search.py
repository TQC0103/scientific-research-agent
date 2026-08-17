from datetime import UTC

import arxiv

from app.config import settings
from app.db.database import upsert_paper


def _clean_id(entry_id: str) -> str:
    return entry_id.rsplit("/", 1)[-1].split("v", 1)[0]


def search_arxiv(
    query: str,
    *,
    year: int | None = None,
    category: str | None = None,
    max_results: int | None = None,
) -> list[dict]:
    limit = max_results or settings.arxiv_max_results
    query_terms = [part for part in query.replace('"', " ").split() if part]
    terms = [" AND ".join(f'all:"{part}"' for part in query_terms)]
    if category:
        terms.append(f"cat:{category}")
    if year:
        terms.append(f"submittedDate:[{year}01010000 TO {year}12312359]")
    search = arxiv.Search(
        query=" AND ".join(terms),
        # Oversample because date/category are also enforced client-side. The
        # public API occasionally returns out-of-range results for compound queries.
        max_results=limit * 3,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    papers: list[dict] = []
    for item in client.results(search):
        if year and item.published.year != year:
            continue
        if category and category not in item.categories:
            continue
        paper = {
            "arxiv_id": _clean_id(item.entry_id),
            "title": " ".join(item.title.split()),
            "abstract": " ".join(item.summary.split()),
            "authors": [author.name for author in item.authors],
            "categories": item.categories,
            "published": item.published.astimezone(UTC).isoformat(),
            "updated": item.updated.astimezone(UTC).isoformat(),
            "doi": item.doi,
            "pdf_url": item.pdf_url,
        }
        upsert_paper(paper)
        papers.append(paper)
        if len(papers) >= limit:
            break
    return papers


def get_arxiv_metadata(arxiv_id: str) -> dict:
    search = arxiv.Search(id_list=[arxiv_id], max_results=1)
    client = arxiv.Client(page_size=1, delay_seconds=3, num_retries=3)
    try:
        item = next(client.results(search))
    except StopIteration as exc:
        raise ValueError(f"arXiv paper not found: {arxiv_id}") from exc
    paper = {
        "arxiv_id": _clean_id(item.entry_id),
        "title": " ".join(item.title.split()),
        "abstract": " ".join(item.summary.split()),
        "authors": [author.name for author in item.authors],
        "categories": item.categories,
        "published": item.published.astimezone(UTC).isoformat(),
        "updated": item.updated.astimezone(UTC).isoformat(),
        "doi": item.doi,
        "pdf_url": item.pdf_url,
    }
    upsert_paper(paper)
    return paper
