import json
from pathlib import Path

import pytest

from app.evaluation.loader import DatasetValidationError, load_suite

FIXTURE_SUITE = Path("evaluation/suites/v0_5/schema_fixtures.json")


def test_loads_committed_schema_fixtures_but_blocks_publishing() -> None:
    suite = load_suite(FIXTURE_SUITE)
    assert suite.schema_version == "1.1.0"
    assert len(suite.cases) == 4
    with pytest.raises(ValueError, match="Only a frozen suite"):
        suite.assert_publishable()


def test_rejects_malformed_json(tmp_path: Path) -> None:
    source = tmp_path / "broken.json"
    source.write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Malformed JSON"):
        load_suite(source)


def test_rejects_unknown_required_paper(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE_SUITE.read_text(encoding="utf-8"))
    payload["cases"][0]["expected"]["required_paper_ids"] = ["9999.99999v1"]
    source = tmp_path / "bad-reference.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Required papers are not declared"):
        load_suite(source)


def test_rejects_arxiv_revision_mismatch(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE_SUITE.read_text(encoding="utf-8"))
    payload["cases"][0]["papers"][0]["revision"] = 6
    source = tmp_path / "bad-revision.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="must identify one revision"):
        load_suite(source)


def test_rejects_unreviewed_repo_case_as_publishable(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE_SUITE.read_text(encoding="utf-8"))
    payload["benchmark_status"] = "frozen"
    payload["frozen_at"] = "2026-08-21T00:00:00Z"
    for case in payload["cases"]:
        case["evaluation_split"] = "test"
    source = tmp_path / "unreviewed.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    suite = load_suite(source)
    with pytest.raises(ValueError, match="independent review"):
        suite.assert_publishable()
