import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from app.evaluation.internal_retrieval_runner import (
    load_internal_corpus,
    render_ablation_markdown,
    run_internal_retrieval_ablation,
    write_ablation_outputs,
)
from app.evaluation.loader import load_suite

SUITE_PATH = Path("evaluation/suites/v0_5/development_10.json")


class FakeDenseEncoder:
    marker = "dot products grow large"

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[0.0, 1.0] if self.marker in text else [1.0, 0.0] for text in texts],
            dtype="float32",
        )

    def encode_query(self, text: str) -> np.ndarray:
        del text
        return np.asarray([0.0, 1.0], dtype="float32")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pdf(path: Path, pages: list[str]) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), text, fontsize=9)
    document.save(path)
    document.close()


def _single_case_inputs(tmp_path: Path):
    full_suite = load_suite(SUITE_PATH)
    case = full_suite.cases[0]
    suite = full_suite.model_copy(update={"cases": [case]})
    papers = tmp_path / "papers"
    papers.mkdir()
    pdf = papers / "1706.03762v7.pdf"
    unrelated = (
        "This appendix discusses optimization schedules and batching details. "
        "It intentionally contains no explanation of scaled attention. " * 3
    )
    evidence = case.gold_evidence[0].quote
    _write_pdf(pdf, [unrelated, evidence])
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "versioned_id": "1706.03762v7",
                        "pdf_sha256": _sha256(pdf),
                        "page_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return suite, sources, papers


def test_lexical_runner_uses_pinned_pdf_and_scores_gold_hidden_ranking(tmp_path: Path) -> None:
    suite, sources, papers = _single_case_inputs(tmp_path)
    result = run_internal_retrieval_ablation(
        suite,
        sources_path=sources,
        papers_dir=papers,
        modes=("lexical",),
        top_k=1,
    )
    aggregate = result.reports["lexical"].aggregate
    assert aggregate.eligible_cases == 1
    assert aggregate.recall_at_k == 1.0
    assert aggregate.precision_at_k == 1.0
    assert result.chunk_counts == {"1706.03762v7": 2}
    assert result.dense_model is None


def test_dense_and_hybrid_share_chunks_and_emit_reports(tmp_path: Path) -> None:
    suite, sources, papers = _single_case_inputs(tmp_path)
    result = run_internal_retrieval_ablation(
        suite,
        sources_path=sources,
        papers_dir=papers,
        modes=("lexical", "dense", "hybrid"),
        top_k=1,
        dense_encoder=FakeDenseEncoder(),
        dense_model="fake/dense",
    )
    assert set(result.reports) == {"lexical", "dense", "hybrid"}
    assert result.reports["dense"].aggregate.recall_at_k == 1.0
    assert result.reports["hybrid"].aggregate.recall_at_k == 1.0
    assert result.rankings["hybrid"][suite.cases[0].case_id][0]["dense_rank"] == 1

    output = tmp_path / "output"
    write_ablation_outputs(result, output)
    assert (output / "ablation_summary.json").is_file()
    assert (output / "ablation_report.md").is_file()
    assert (output / "hybrid" / "retrieved.jsonl").is_file()
    assert "annotation-relative" in render_ablation_markdown(result)


def test_corpus_rejects_pdf_hash_mismatch(tmp_path: Path) -> None:
    suite, sources, papers = _single_case_inputs(tmp_path)
    payload = json.loads(sources.read_text(encoding="utf-8"))
    payload["sources"][0]["pdf_sha256"] = "0" * 64
    sources.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_internal_corpus(suite, sources_path=sources, papers_dir=papers)


def test_runner_rejects_duplicate_modes(tmp_path: Path) -> None:
    suite, sources, papers = _single_case_inputs(tmp_path)
    with pytest.raises(ValueError, match="duplicates"):
        run_internal_retrieval_ablation(
            suite,
            sources_path=sources,
            papers_dir=papers,
            modes=("lexical", "lexical"),
        )
