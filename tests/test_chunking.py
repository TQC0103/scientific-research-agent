from app.ingestion.chunking import chunk_pages
from app.ingestion.pdf_parser import detect_section


def test_detect_section() -> None:
    assert detect_section("3.2 Experiments") == "Experiments"
    assert detect_section("ordinary sentence") is None


def test_chunk_metadata_is_preserved() -> None:
    pages = [{"page": 4, "text": "Methods\n" + ("Evidence sentence. " * 150)}]
    chunks = chunk_pages(pages, chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(chunk["page"] == 4 for chunk in chunks)
    assert all(chunk["section"] == "Method" for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))

