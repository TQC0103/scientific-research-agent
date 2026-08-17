import re
from datetime import UTC
from typing import Literal

import arxiv

from app.config import settings
from app.db.database import get_paper, upsert_paper

DateMode = Literal["first_submitted", "last_revised", "active_in_period"]


def split_arxiv_id(value: str) -> tuple[str, str, int | None]:
    short_id = value.rstrip("/")
    if "/abs/" in short_id:
        short_id = short_id.split("/abs/", 1)[1]
    match = re.match(r"^(?P<base>.+?)v(?P<version>\d+)$", short_id)
    if not match:
        return short_id, short_id, None
    return short_id, match.group("base"), int(match.group("version"))


def normalize_result(item: arxiv.Result) -> dict:
    versioned_id, base_id, version = split_arxiv_id(item.get_short_id())
    published = item.published.astimezone(UTC).isoformat()
    updated = item.updated.astimezone(UTC).isoformat()
    return {
        "arxiv_id": base_id,
        "versioned_id": versioned_id,
        "version": version,
        "title": " ".join(item.title.split()),
        "abstract": " ".join(item.summary.split()),
        "authors": [author.name for author in item.authors],
        "categories": item.categories,
        "primary_category": item.primary_category,
        # Kept for backwards compatibility; the explicit names below are canonical.
        "published": published,
        "updated": updated,
        "first_submitted_at": published,
        "last_revised_at": updated,
        "doi": item.doi,
        "journal_ref": item.journal_ref,
        "comment": item.comment,
        "pdf_url": item.pdf_url,
    }


def _matches_year(item: arxiv.Result, year: int, date_mode: DateMode) -> bool:
    first_year = item.published.astimezone(UTC).year
    revised_year = item.updated.astimezone(UTC).year
    if date_mode == "first_submitted":
        return first_year == year
    if date_mode == "last_revised":
        return revised_year == year
    return first_year == year or revised_year == year


def search_arxiv(
    query: str,
    *,
    year: int | None = None,
    category: str | None = None,
    max_results: int | None = None,
    date_mode: DateMode = "first_submitted",
) -> list[dict]:
    limit = max_results or settings.arxiv_max_results
    query_terms = [part for part in query.replace('"', " ").split() if part]
    if not query_terms:
        return []
    terms = [" AND ".join(f'all:"{part}"' for part in query_terms)]
    if category:
        terms.append(f"cat:{category}")
    if year and date_mode == "first_submitted":
        terms.append(f"submittedDate:[{year}01010000 TO {year}12312359]")

    sort_by = arxiv.SortCriterion.Relevance
    scan_results = limit * 3
    if year and date_mode in {"last_revised", "active_in_period"}:
        # The Atom API has no lastUpdatedDate filter. Scan a bounded relevance
        # window and validate client-side. The bound avoids unbounded API crawling.
        scan_results = min(max(limit * 20, 200), 1000)

    search = arxiv.Search(
        query=" AND ".join(terms),
        max_results=scan_results,
        sort_by=sort_by,
        sort_order=arxiv.SortOrder.Descending,
    )
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    papers: list[dict] = []
    for item in client.results(search):
        if year and not _matches_year(item, year, date_mode):
            continue
        if category and category not in item.categories:
            continue
        paper = normalize_result(item)
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
    paper = normalize_result(item)
    upsert_paper(paper)
    return get_paper(paper["arxiv_id"]) or paper
