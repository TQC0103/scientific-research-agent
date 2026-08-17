from app.config import settings
from app.db.database import (
    get_paper,
    search_local,
    update_index_artifact,
    update_pdf_artifact,
    upsert_paper,
)


def _paper(version: int) -> dict:
    return {
        "arxiv_id": "2501.00001",
        "versioned_id": f"2501.00001v{version}",
        "version": version,
        "title": "Agentic Retrieval for Scientific QA",
        "abstract": "An agent retrieves scientific evidence.",
        "authors": ["A. Researcher"],
        "categories": ["cs.AI"],
        "primary_category": "cs.AI",
        "published": "2025-01-01T00:00:00+00:00",
        "updated": f"2025-01-0{version}T00:00:00+00:00",
        "first_submitted_at": "2025-01-01T00:00:00+00:00",
        "last_revised_at": f"2025-01-0{version}T00:00:00+00:00",
        "doi": None,
        "journal_ref": None,
        "comment": None,
        "pdf_url": f"https://arxiv.org/pdf/2501.00001v{version}",
    }


def test_fts_search_and_version_change_invalidates_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    upsert_paper(_paper(1))
    update_pdf_artifact("2501.00001", status="available", path="paper.pdf", sha256="abc", size=1234)
    update_index_artifact("2501.00001", version=1, pdf_sha256="abc")

    assert search_local("What is scientific retrieval?")[0]["versioned_id"] == "2501.00001v1"

    upsert_paper(_paper(2))
    refreshed = get_paper("2501.00001")
    assert refreshed["pdf_status"] == "stale"
    assert refreshed["pdf_path"] is None
    assert refreshed["indexed_version"] is None
