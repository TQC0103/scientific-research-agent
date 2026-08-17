from app.tools.paper_download import safe_paper_id


def test_safe_paper_id_handles_legacy_ids() -> None:
    assert safe_paper_id("cs/9901001") == "cs_9901001"

