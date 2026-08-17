import pymupdf
import pytest

from app.tools.paper_download import InvalidPdfError, validate_pdf


def test_validate_pdf_accepts_readable_pdf(tmp_path) -> None:
    path = tmp_path / "valid.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(path)
    validate_pdf(path)


def test_validate_pdf_rejects_html_response(tmp_path) -> None:
    path = tmp_path / "not-a-pdf.pdf"
    path.write_bytes(b"<html>error</html>" * 100)
    with pytest.raises(InvalidPdfError):
        validate_pdf(path)
