import hashlib
import tempfile
from pathlib import Path

import httpx
import pymupdf

from app.config import settings
from app.db.database import update_pdf_artifact
from app.tools.arxiv_search import get_arxiv_metadata


class PaperDownloadError(RuntimeError):
    """Base class for recoverable paper download failures."""


class PdfUnavailableError(PaperDownloadError):
    """The arXiv record has no usable PDF response."""


class InvalidPdfError(PaperDownloadError):
    """The downloaded bytes are not a valid readable PDF."""


def safe_paper_id(arxiv_id: str) -> str:
    return arxiv_id.strip().replace("/", "_").replace("\\", "_")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pdf(path: Path) -> None:
    if not path.exists() or path.stat().st_size < 5:
        raise InvalidPdfError(f"PDF is missing or too small: {path}")
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise InvalidPdfError(f"Response does not start with a PDF signature: {path}")
    try:
        with pymupdf.open(path) as document:
            if document.page_count < 1:
                raise InvalidPdfError(f"PDF has no pages: {path}")
    except InvalidPdfError:
        raise
    except Exception as exc:
        raise InvalidPdfError(f"PyMuPDF cannot open {path.name}: {exc}") from exc


def download_paper(
    arxiv_id: str,
    *,
    force: bool = False,
    paper: dict | None = None,
) -> Path:
    settings.ensure_directories()
    paper = paper or get_arxiv_metadata(arxiv_id)
    base_id = paper["arxiv_id"]
    versioned_id = paper.get("versioned_id") or base_id
    target = settings.papers_dir / f"{safe_paper_id(versioned_id)}.pdf"

    if target.exists() and not force:
        try:
            validate_pdf(target)
            update_pdf_artifact(
                base_id,
                status="available",
                path=str(target),
                sha256=file_sha256(target),
                size=target.stat().st_size,
            )
            return target
        except InvalidPdfError:
            target.unlink(missing_ok=True)

    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        update_pdf_artifact(base_id, status="unavailable")
        raise PdfUnavailableError(f"arXiv returned no PDF URL for {versioned_id}")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{safe_paper_id(versioned_id)}-",
            suffix=".part",
            dir=settings.papers_dir,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            with httpx.stream(
                "GET",
                pdf_url,
                follow_redirects=True,
                timeout=settings.request_timeout_seconds,
                headers={"User-Agent": "scientific-research-agent/0.2 (personal research project)"},
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    temp_file.write(chunk)
        validate_pdf(temp_path)
        temp_path.replace(target)
        digest = file_sha256(target)
        update_pdf_artifact(
            base_id,
            status="available",
            path=str(target),
            sha256=digest,
            size=target.stat().st_size,
        )
        return target
    except httpx.HTTPStatusError as exc:
        status = "unavailable" if exc.response.status_code in {404, 410} else "download_failed"
        update_pdf_artifact(base_id, status=status)
        raise PdfUnavailableError(
            f"PDF request failed for {versioned_id}: HTTP {exc.response.status_code}"
        ) from exc
    except (httpx.HTTPError, InvalidPdfError) as exc:
        update_pdf_artifact(base_id, status="download_failed")
        raise PaperDownloadError(f"Could not download {versioned_id}: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
