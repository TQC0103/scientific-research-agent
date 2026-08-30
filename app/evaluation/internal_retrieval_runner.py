"""Portable lexical/dense/hybrid ablations for the internal evaluation suite."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pymupdf

from app.evaluation.models import EvaluationSuite
from app.evaluation.qasper_runner import BM25Retriever
from app.evaluation.retrieval import RetrievalReport, evaluate_retrieval
from app.ingestion.chunking import chunk_pages
from app.ingestion.pdf_parser import parse_pdf

SUPPORTED_MODES = (
    "lexical",
    "dense",
    "hybrid",
    "hybrid_score",
    "hybrid_per_paper",
    "hybrid_score_per_paper",
    "hybrid_rerank",
    "hybrid_rerank_per_paper",
)
HYBRID_MODES = frozenset(SUPPORTED_MODES[2:])
DEFAULT_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_DENSE_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
RERANKER_WINDOW_CHARS = 900
RERANKER_WINDOW_OVERLAP = 150


@dataclass(frozen=True)
class CorpusChunk:
    corpus_index: int
    paper_id: str
    versioned_id: str
    page: int
    section: str
    chunk_index: int
    text: str

    def retrieval_payload(self, **scores: Any) -> dict[str, Any]:
        return {
            "arxiv_id": self.paper_id,
            "versioned_id": self.versioned_id,
            "page": self.page,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "text": self.text,
            **scores,
        }


class DenseEncoder(Protocol):
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, question: str, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerDenseEncoder:
    def __init__(
        self, model_name: str, *, revision: str | None = None, batch_size: int = 32
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense internal retrieval requires sentence-transformers."
            ) from exc
        self.model = SentenceTransformer(model_name, revision=revision)
        self.batch_size = batch_size

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )

    def encode_query(self, text: str) -> np.ndarray:
        prompt_name = "query" if "query" in getattr(self.model, "prompts", {}) else None
        return np.asarray(
            self.model.encode(
                [text],
                prompt_name=prompt_name,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0],
            dtype="float32",
        )


class CrossEncoderReranker:
    def __init__(
        self, model_name: str, *, revision: str | None = None, batch_size: int = 32
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Internal retrieval reranking requires sentence-transformers."
            ) from exc
        self.model = CrossEncoder(model_name, revision=revision)
        self.batch_size = batch_size

    def score(self, question: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype="float32")
        return np.asarray(
            self.model.predict(
                [(question, text) for text in texts],
                batch_size=self.batch_size,
                show_progress_bar=False,
            ),
            dtype="float32",
        ).reshape(-1)

@dataclass(frozen=True)
class InternalAblationResult:
    reports: dict[str, RetrievalReport]
    rankings: dict[str, dict[str, list[dict[str, Any]]]]
    latency_seconds: dict[str, float]
    chunk_counts: dict[str, int]
    source_hashes: dict[str, str]
    dense_model: str | None
    dense_revision: str | None
    reranker_model: str | None
    reranker_revision: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load internal source manifest {source}: {exc}") from exc
    entries = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("Internal source manifest requires a non-empty sources list.")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Each internal source entry must be an object.")
        versioned_id = entry.get("versioned_id")
        if not isinstance(versioned_id, str) or not versioned_id:
            raise ValueError("Each internal source requires versioned_id.")
        if versioned_id in result:
            raise ValueError(f"Duplicate internal source revision: {versioned_id}")
        result[versioned_id] = entry
    return result


def load_internal_corpus(
    suite: EvaluationSuite,
    *,
    sources_path: str | Path,
    papers_dir: str | Path,
    chunk_size: int = 1800,
    overlap: int = 250,
) -> tuple[list[CorpusChunk], dict[str, str]]:
    sources = _load_source_manifest(sources_path)
    required_revisions = {
        paper.versioned_id for case in suite.cases for paper in case.papers
    }
    missing_manifest = required_revisions - set(sources)
    if missing_manifest:
        raise ValueError(f"Source manifest is missing revisions: {sorted(missing_manifest)}")

    corpus: list[CorpusChunk] = []
    hashes = {}
    source_directory = Path(papers_dir)
    for versioned_id in sorted(required_revisions):
        entry = sources[versioned_id]
        pdf = source_directory / f"{versioned_id}.pdf"
        if not pdf.is_file():
            raise FileNotFoundError(f"Pinned source PDF is missing: {pdf}")
        actual_hash = _sha256(pdf)
        expected_hash = entry.get("pdf_sha256")
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {versioned_id}: expected {expected_hash}, got {actual_hash}"
            )
        with pymupdf.open(pdf) as document:
            page_count = document.page_count
        if page_count != entry.get("page_count"):
            raise ValueError(
                f"Page-count mismatch for {versioned_id}: expected "
                f"{entry.get('page_count')}, got {page_count}"
            )
        hashes[versioned_id] = actual_hash
        paper_id = versioned_id.rsplit("v", 1)[0]
        for chunk in chunk_pages(parse_pdf(pdf), chunk_size=chunk_size, overlap=overlap):
            corpus.append(
                CorpusChunk(
                    corpus_index=len(corpus),
                    paper_id=paper_id,
                    versioned_id=versioned_id,
                    page=int(chunk["page"]),
                    section=str(chunk["section"]),
                    chunk_index=int(chunk["chunk_index"]),
                    text=str(chunk["text"]),
                )
            )
    return corpus, hashes


def _case_corpus(case: Any, corpus: list[CorpusChunk]) -> list[CorpusChunk]:
    revisions = {paper.versioned_id for paper in case.papers}
    return [chunk for chunk in corpus if chunk.versioned_id in revisions]


def _lexical_ranking(
    question: str, corpus: list[CorpusChunk], *, candidate_k: int
) -> list[tuple[CorpusChunk, float]]:
    retriever = BM25Retriever(tuple(chunk.text for chunk in corpus))
    return [
        (corpus[item.index], item.score)
        for item in retriever.retrieve(question, min(candidate_k, len(corpus)))
    ]


def _dense_ranking(
    question: str,
    corpus: list[CorpusChunk],
    *,
    candidate_k: int,
    document_vectors: np.ndarray,
    encoder: DenseEncoder,
) -> list[tuple[CorpusChunk, float]]:
    query = encoder.encode_query(question)
    positions = np.asarray([chunk.corpus_index for chunk in corpus], dtype="int64")
    scores = document_vectors[positions] @ query
    order = sorted(range(len(corpus)), key=lambda index: (-float(scores[index]), index))
    return [(corpus[index], float(scores[index])) for index in order[:candidate_k]]


def _normalized_scores(
    ranking: list[tuple[CorpusChunk, float]],
    *,
    by_paper: bool = False,
) -> dict[int, float]:
    if not ranking:
        return {}
    if by_paper:
        paper_ids = dict.fromkeys(chunk.paper_id for chunk, _score in ranking)
        result = {}
        for paper_id in paper_ids:
            result.update(
                _normalized_scores(
                    [item for item in ranking if item[0].paper_id == paper_id]
                )
            )
        return result
    values = [score for _chunk, score in ranking]
    minimum = min(values)
    span = max(values) - minimum
    if span <= 1e-12:
        return {chunk.corpus_index: 1.0 for chunk, _score in ranking}
    return {
        chunk.corpus_index: (score - minimum) / span for chunk, score in ranking
    }


def _hybrid_payload(
    lexical: list[tuple[CorpusChunk, float]],
    dense: list[tuple[CorpusChunk, float]],
    *,
    top_k: int,
    method: str,
    paper_order: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    fused: dict[int, float] = {}
    chunks: dict[int, CorpusChunk] = {}
    lexical_ranks = {}
    dense_ranks = {}
    lexical_scores = {chunk.corpus_index: score for chunk, score in lexical}
    dense_scores = {chunk.corpus_index: score for chunk, score in dense}
    by_paper = bool(paper_order)
    lexical_normalized = (
        _normalized_scores(lexical, by_paper=by_paper) if method == "score" else {}
    )
    dense_normalized = (
        _normalized_scores(dense, by_paper=by_paper) if method == "score" else {}
    )
    lexical_paper_ranks: dict[str, int] = {}
    for global_rank, (chunk, _score) in enumerate(lexical, 1):
        lexical_paper_ranks[chunk.paper_id] = lexical_paper_ranks.get(chunk.paper_id, 0) + 1
        rank = lexical_paper_ranks[chunk.paper_id] if by_paper else global_rank
        contribution = (
            lexical_normalized[chunk.corpus_index]
            if method == "score"
            else 1 / (60 + rank)
        )
        fused[chunk.corpus_index] = fused.get(chunk.corpus_index, 0.0) + contribution
        chunks[chunk.corpus_index] = chunk
        lexical_ranks[chunk.corpus_index] = rank
    dense_paper_ranks: dict[str, int] = {}
    for global_rank, (chunk, _score) in enumerate(dense, 1):
        dense_paper_ranks[chunk.paper_id] = dense_paper_ranks.get(chunk.paper_id, 0) + 1
        rank = dense_paper_ranks[chunk.paper_id] if by_paper else global_rank
        contribution = (
            dense_normalized[chunk.corpus_index]
            if method == "score"
            else 1 / (60 + rank)
        )
        fused[chunk.corpus_index] = fused.get(chunk.corpus_index, 0.0) + contribution
        chunks[chunk.corpus_index] = chunk
        dense_ranks[chunk.corpus_index] = rank
    order = sorted(fused, key=lambda index: (-fused[index], index))
    if len(paper_order) > 1 and top_k >= len(paper_order):
        quota = top_k // len(paper_order)
        selected = []
        selected_set = set()
        for paper_id in paper_order:
            for index in order:
                if chunks[index].paper_id == paper_id and index not in selected_set:
                    selected.append(index)
                    selected_set.add(index)
                    if sum(chunks[item].paper_id == paper_id for item in selected) >= quota:
                        break
        for index in order:
            if len(selected) >= top_k:
                break
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
        order = sorted(selected, key=lambda index: (-fused[index], index))
    order = order[:top_k]
    return [
        chunks[index].retrieval_payload(
            score=fused[index],
            retrieval_score=fused[index],
            fusion_method=method,
            lexical_score=lexical_scores.get(index),
            dense_score=dense_scores.get(index),
            lexical_rank=lexical_ranks.get(index),
            dense_rank=dense_ranks.get(index),
        )
        for index in order
    ]


def _apply_paper_quota(
    order: list[int],
    chunks: dict[int, CorpusChunk],
    scores: dict[int, float],
    *,
    top_k: int,
    paper_order: tuple[str, ...],
) -> list[int]:
    if len(paper_order) <= 1 or top_k < len(paper_order):
        return order[:top_k]
    quota = top_k // len(paper_order)
    selected: list[int] = []
    selected_set: set[int] = set()
    for paper_id in paper_order:
        paper_candidates = [
            index
            for index in order
            if chunks[index].paper_id == paper_id and index not in selected_set
        ]
        for index in paper_candidates[:quota]:
            selected.append(index)
            selected_set.add(index)
    for index in order:
        if len(selected) >= top_k:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    return sorted(selected, key=lambda index: (-scores[index], index))[:top_k]


def _reranker_windows(text: str) -> list[str]:
    if len(text) <= RERANKER_WINDOW_CHARS:
        return [text]
    step = RERANKER_WINDOW_CHARS - RERANKER_WINDOW_OVERLAP
    return [
        text[start : start + RERANKER_WINDOW_CHARS]
        for start in range(0, len(text), step)
        if text[start : start + RERANKER_WINDOW_CHARS].strip()
    ]


def _rerank_payload(
    question: str,
    lexical: list[tuple[CorpusChunk, float]],
    dense: list[tuple[CorpusChunk, float]],
    *,
    reranker: Reranker,
    top_k: int,
    paper_order: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    chunks: dict[int, CorpusChunk] = {}
    lexical_scores: dict[int, float] = {}
    dense_scores: dict[int, float] = {}
    lexical_ranks: dict[int, int] = {}
    dense_ranks: dict[int, int] = {}
    lexical_paper_ranks: dict[str, int] = {}
    dense_paper_ranks: dict[str, int] = {}
    by_paper = bool(paper_order)
    for global_rank, (chunk, score) in enumerate(lexical, 1):
        lexical_paper_ranks[chunk.paper_id] = lexical_paper_ranks.get(chunk.paper_id, 0) + 1
        chunks[chunk.corpus_index] = chunk
        lexical_scores[chunk.corpus_index] = score
        lexical_ranks[chunk.corpus_index] = (
            lexical_paper_ranks[chunk.paper_id] if by_paper else global_rank
        )
    for global_rank, (chunk, score) in enumerate(dense, 1):
        dense_paper_ranks[chunk.paper_id] = dense_paper_ranks.get(chunk.paper_id, 0) + 1
        chunks[chunk.corpus_index] = chunk
        dense_scores[chunk.corpus_index] = score
        dense_ranks[chunk.corpus_index] = (
            dense_paper_ranks[chunk.paper_id] if by_paper else global_rank
        )
    candidate_ids = list(chunks)
    windows: list[str] = []
    window_owners: list[int] = []
    for index in candidate_ids:
        chunk_windows = _reranker_windows(chunks[index].text)
        windows.extend(chunk_windows)
        window_owners.extend([index] * len(chunk_windows))
    reranker_values = reranker.score(question, windows)
    if reranker_values.shape != (len(windows),):
        raise ValueError("Reranker returned the wrong score count.")
    reranker_scores = {index: float("-inf") for index in candidate_ids}
    for index, value in zip(window_owners, reranker_values, strict=True):
        reranker_scores[index] = max(reranker_scores[index], float(value))
    order = sorted(candidate_ids, key=lambda index: (-reranker_scores[index], index))
    order = _apply_paper_quota(
        order,
        chunks,
        reranker_scores,
        top_k=top_k,
        paper_order=paper_order,
    )
    return [
        chunks[index].retrieval_payload(
            score=reranker_scores[index],
            retrieval_score=reranker_scores[index],
            fusion_method="cross_encoder_rerank",
            lexical_score=lexical_scores.get(index),
            dense_score=dense_scores.get(index),
            lexical_rank=lexical_ranks.get(index),
            dense_rank=dense_ranks.get(index),
            reranker_score=reranker_scores[index],
        )
        for index in order
    ]


def _ranking_payload(
    mode: str,
    lexical: list[tuple[CorpusChunk, float]],
    dense: list[tuple[CorpusChunk, float]],
    *,
    top_k: int,
    paper_order: tuple[str, ...] = (),
    question: str = "",
    reranker: Reranker | None = None,
) -> list[dict[str, Any]]:
    if mode == "lexical":
        return [
            chunk.retrieval_payload(score=score, lexical_rank=rank)
            for rank, (chunk, score) in enumerate(lexical[:top_k], 1)
        ]
    if mode == "dense":
        return [
            chunk.retrieval_payload(score=score, dense_rank=rank)
            for rank, (chunk, score) in enumerate(dense[:top_k], 1)
        ]
    if "rerank" in mode:
        if reranker is None:
            raise ValueError("A reranker is required for rerank modes.")
        return _rerank_payload(
            question,
            lexical,
            dense,
            reranker=reranker,
            top_k=top_k,
            paper_order=paper_order if "per_paper" in mode else (),
        )
    return _hybrid_payload(
        lexical,
        dense,
        top_k=top_k,
        method="score" if "score" in mode else "rrf",
        paper_order=paper_order if "per_paper" in mode else (),
    )


def run_internal_retrieval_ablation(
    suite: EvaluationSuite,
    *,
    sources_path: str | Path,
    papers_dir: str | Path,
    modes: tuple[str, ...] = SUPPORTED_MODES,
    top_k: int = 5,
    dense_model: str = DEFAULT_DENSE_MODEL,
    dense_revision: str | None = DEFAULT_DENSE_REVISION,
    dense_batch_size: int = 32,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    reranker_revision: str | None = DEFAULT_RERANKER_REVISION,
    reranker_batch_size: int = 32,
    chunk_size: int = 1800,
    overlap: int = 250,
    dense_encoder: DenseEncoder | None = None,
    reranker: Reranker | None = None,
) -> InternalAblationResult:
    if not modes or any(mode not in SUPPORTED_MODES for mode in modes):
        raise ValueError(f"modes must be a non-empty subset of {SUPPORTED_MODES}")
    if len(set(modes)) != len(modes):
        raise ValueError("modes must not contain duplicates.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    corpus, source_hashes = load_internal_corpus(
        suite,
        sources_path=sources_path,
        papers_dir=papers_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    chunk_counts = {
        versioned_id: sum(chunk.versioned_id == versioned_id for chunk in corpus)
        for versioned_id in sorted(source_hashes)
    }
    uses_dense = any(mode == "dense" or mode in HYBRID_MODES for mode in modes)
    encoder = dense_encoder
    document_vectors = None
    if uses_dense:
        encoder = encoder or SentenceTransformerDenseEncoder(
            dense_model, revision=dense_revision, batch_size=dense_batch_size
        )
        document_vectors = encoder.encode_documents([chunk.text for chunk in corpus])
        if document_vectors.shape[0] != len(corpus):
            raise ValueError("Dense encoder returned the wrong document count.")
    uses_reranker = any("rerank" in mode for mode in modes)
    active_reranker = reranker
    if uses_reranker:
        active_reranker = active_reranker or CrossEncoderReranker(
            reranker_model,
            revision=reranker_revision,
            batch_size=reranker_batch_size,
        )

    rankings = {mode: {} for mode in modes}
    latency = {mode: 0.0 for mode in modes}
    candidate_k = max(top_k * 4, 20)
    for case in suite.cases:
        case_corpus = _case_corpus(case, corpus)
        lexical_started = time.perf_counter()
        lexical = _lexical_ranking(case.question, case_corpus, candidate_k=candidate_k)
        lexical_seconds = time.perf_counter() - lexical_started
        dense = []
        dense_seconds = 0.0
        if uses_dense:
            assert encoder is not None and document_vectors is not None
            dense_started = time.perf_counter()
            dense = _dense_ranking(
                case.question,
                case_corpus,
                candidate_k=min(candidate_k, len(case_corpus)),
                document_vectors=document_vectors,
                encoder=encoder,
            )
            dense_seconds = time.perf_counter() - dense_started
        per_paper_lexical = []
        per_paper_dense = []
        per_paper_seconds = 0.0
        if any("per_paper" in mode for mode in modes):
            per_paper_started = time.perf_counter()
            for paper in case.papers:
                paper_corpus = [
                    chunk
                    for chunk in case_corpus
                    if chunk.versioned_id == paper.versioned_id
                ]
                per_paper_lexical.extend(
                    _lexical_ranking(
                        case.question,
                        paper_corpus,
                        candidate_k=min(candidate_k, len(paper_corpus)),
                    )
                )
                assert encoder is not None and document_vectors is not None
                per_paper_dense.extend(
                    _dense_ranking(
                        case.question,
                        paper_corpus,
                        candidate_k=min(candidate_k, len(paper_corpus)),
                        document_vectors=document_vectors,
                        encoder=encoder,
                    )
                )
            per_paper_seconds = time.perf_counter() - per_paper_started
        for mode in modes:
            mode_started = time.perf_counter()
            mode_lexical = per_paper_lexical if "per_paper" in mode else lexical
            mode_dense = per_paper_dense if "per_paper" in mode else dense
            rankings[mode][case.case_id] = _ranking_payload(
                mode,
                mode_lexical,
                mode_dense,
                top_k=top_k,
                paper_order=tuple(paper.paper_id for paper in case.papers),
                question=case.question,
                reranker=active_reranker,
            )
            fusion_seconds = time.perf_counter() - mode_started
            if mode == "lexical":
                latency[mode] += lexical_seconds + fusion_seconds
            elif mode == "dense":
                latency[mode] += dense_seconds + fusion_seconds
            elif "per_paper" in mode:
                latency[mode] += per_paper_seconds + fusion_seconds
            else:
                latency[mode] += lexical_seconds + dense_seconds + fusion_seconds

    reports = {
        mode: evaluate_retrieval(
            suite,
            mode_rankings,
            config_name=f"internal-{mode}",
            top_k=top_k,
        )
        for mode, mode_rankings in rankings.items()
    }
    return InternalAblationResult(
        reports=reports,
        rankings=rankings,
        latency_seconds=latency,
        chunk_counts=chunk_counts,
        source_hashes=source_hashes,
        dense_model=dense_model if uses_dense else None,
        dense_revision=dense_revision if uses_dense else None,
        reranker_model=reranker_model if uses_reranker else None,
        reranker_revision=reranker_revision if uses_reranker else None,
    )


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def render_ablation_markdown(result: InternalAblationResult) -> str:
    lines = [
        "# Internal retrieval ablation",
        "",
        "Metrics are annotation-relative development signals, not held-out accuracy.",
        "",
        "| Mode | Recall@K | Precision@K | MRR | Gold evidence | Required papers | Macro paper recall | Eligible | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, report in result.reports.items():
        aggregate = report.aggregate
        lines.append(
            f"| {mode} | {_format_metric(aggregate.recall_at_k)} | "
            f"{_format_metric(aggregate.precision_at_k)} | {_format_metric(aggregate.mrr)} | "
            f"{_format_metric(aggregate.gold_evidence_coverage)} | "
            f"{_format_metric(aggregate.required_paper_coverage)} | "
            f"{_format_metric(aggregate.macro_paper_recall)} | "
            f"{aggregate.eligible_cases} | {result.latency_seconds[mode]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-case diagnostics",
            "",
            "| Case | Mode | Recall@K | Reciprocal rank | Diagnostics |",
            "|---|---|---:|---:|---|",
        ]
    )
    for mode, report in result.reports.items():
        for case in report.cases:
            diagnostics = ", ".join(case.diagnostics) or "—"
            lines.append(
                f"| {case.case_id} | {mode} | {_format_metric(case.recall_at_k)} | "
                f"{_format_metric(case.reciprocal_rank)} | {diagnostics} |"
            )
    return "\n".join(lines) + "\n"


def write_ablation_outputs(result: InternalAblationResult, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    modes = {}
    for mode, report in result.reports.items():
        mode_dir = destination / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / "retrieved.jsonl").write_text(
            "".join(
                json.dumps(
                    {"case_id": case_id, "retrieved": chunks}, ensure_ascii=False
                )
                + "\n"
                for case_id, chunks in result.rankings[mode].items()
            ),
            encoding="utf-8",
        )
        (mode_dir / "per_case.jsonl").write_text(
            "".join(
                json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n"
                for case in report.cases
            ),
            encoding="utf-8",
        )
        metrics = report.aggregate.model_dump(mode="json")
        metrics["latency_seconds"] = result.latency_seconds[mode]
        (mode_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        modes[mode] = metrics
    summary = {
        "suite_id": next(iter(result.reports.values())).aggregate.suite_id,
        "dataset_version": next(iter(result.reports.values())).aggregate.dataset_version,
        "dense_model": result.dense_model,
        "dense_revision": result.dense_revision,
        "reranker_model": result.reranker_model,
        "reranker_revision": result.reranker_revision,
        "chunk_counts": result.chunk_counts,
        "source_hashes": result.source_hashes,
        "modes": modes,
    }
    (destination / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "ablation_report.md").write_text(
        render_ablation_markdown(result), encoding="utf-8"
    )
