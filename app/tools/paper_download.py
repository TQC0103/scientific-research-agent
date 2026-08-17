from pathlib import Path

import httpx

from app.config import settings
from app.db.database import get_paper, update_artifact
from app.tools.arxiv_search import get_arxiv_metadata


def safe_paper_id(arxiv_id: str) -> str:
    return arxiv_id.strip().replace("/", "_").replace("\\", "_")


def download_paper(arxiv_id: str, *, force: bool = False) -> Path:
    settings.ensure_directories()
    paper = get_paper(arxiv_id) or get_arxiv_metadata(arxiv_id)
    target = settings.papers_dir / f"{safe_paper_id(arxiv_id)}.pdf"
    if target.exists() and target.stat().st_size > 1024 and not force:
        return target
    with httpx.stream(
        "GET",
        paper["pdf_url"],
        follow_redirects=True,
        timeout=settings.request_timeout_seconds,
        headers={"User-Agent": "scientific-research-agent/0.1 (personal research project)"},
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            raise RuntimeError(f"Expected PDF, received {content_type!r}")
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    update_artifact(arxiv_id, pdf_path=str(target))
    return target

