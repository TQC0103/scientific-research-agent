from pathlib import Path

from app.db.database import set_pdf_status, update_index_artifact
from app.ingestion.chunking import chunk_pages
from app.ingestion.pdf_parser import parse_pdf
from app.retrieval.vector_store import build_index
from app.tools.arxiv_search import get_arxiv_metadata
from app.tools.paper_download import download_paper, file_sha256


def index_paper(
    arxiv_id: str,
    *,
    force_download: bool = False,
    paper: dict | None = None,
) -> tuple[Path, int]:
    paper = paper or get_arxiv_metadata(arxiv_id)
    pdf = download_paper(arxiv_id, force=force_download, paper=paper)
    try:
        chunks = chunk_pages(parse_pdf(pdf))
    except ValueError:
        set_pdf_status(paper["arxiv_id"], "no_text_layer")
        raise
    pdf_hash = file_sha256(pdf)
    destination = build_index(paper["arxiv_id"], chunks, paper=paper, pdf_sha256=pdf_hash)
    update_index_artifact(paper["arxiv_id"], version=paper.get("version"), pdf_sha256=pdf_hash)
    return destination, len(chunks)
