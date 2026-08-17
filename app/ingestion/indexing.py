from pathlib import Path

from app.db.database import update_artifact
from app.ingestion.chunking import chunk_pages
from app.ingestion.pdf_parser import parse_pdf
from app.retrieval.vector_store import build_index
from app.tools.paper_download import download_paper


def index_paper(arxiv_id: str, *, force_download: bool = False) -> tuple[Path, int]:
    pdf = download_paper(arxiv_id, force=force_download)
    chunks = chunk_pages(parse_pdf(pdf))
    destination = build_index(arxiv_id, chunks)
    update_artifact(arxiv_id, indexed=True)
    return destination, len(chunks)

