import json
import math
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings

from app.config import settings
from app.tools.paper_download import safe_paper_id

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "how",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "which",
    "why",
    "with",
    "without",
}
RRF_K = 60
RERANKER_WINDOW_CHARS = 900
RERANKER_WINDOW_OVERLAP = 150


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)


def index_directory(arxiv_id: str) -> Path:
    return settings.indexes_dir / safe_paper_id(arxiv_id)


def build_index(arxiv_id: str, chunks: list[dict], *, paper: dict, pdf_sha256: str) -> Path:
    if not chunks:
        raise ValueError("Cannot index a paper with no chunks.")
    destination = index_directory(arxiv_id)
    destination.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(
        _embeddings().embed_documents([c["text"] for c in chunks]), dtype="float32"
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(destination / "index.faiss"))
    (destination / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "index_meta.json").write_text(
        json.dumps(
            {
                "arxiv_id": arxiv_id,
                "versioned_id": paper.get("versioned_id"),
                "version": paper.get("version"),
                "pdf_sha256": pdf_sha256,
                "embedding_model": settings.ollama_embed_model,
                "built_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def index_is_current(paper: dict) -> bool:
    source = index_directory(paper["arxiv_id"])
    required = (source / "index.faiss", source / "chunks.json", source / "index_meta.json")
    if not all(path.exists() for path in required):
        return False
    try:
        metadata = json.loads((source / "index_meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        metadata.get("version") == paper.get("version")
        and metadata.get("pdf_sha256")
        and metadata.get("pdf_sha256") == paper.get("pdf_sha256")
        and metadata.get("embedding_model") == settings.ollama_embed_model
    )


def _lexical_scores(query: str, chunks: list[dict]) -> list[float]:
    """Small BM25-like scorer used alongside dense retrieval."""
    terms = {
        token
        for token in TOKEN_PATTERN.findall(query.casefold())
        if len(token) > 2 and token not in LEXICAL_STOPWORDS
    }
    if not terms:
        return [0.0] * len(chunks)
    documents = [TOKEN_PATTERN.findall(chunk["text"].casefold()) for chunk in chunks]
    document_frequency = {
        term: sum(term in set(document) for document in documents) for term in terms
    }
    count = len(documents)
    scores = []
    for document in documents:
        frequencies = Counter(document)
        score = sum(
            (1 + math.log(frequencies[term]))
            * math.log((count + 1) / (document_frequency[term] + 1))
            for term in terms
            if frequencies[term]
        )
        scores.append(score)
    return scores


def _reranker_windows(text: str) -> list[str]:
    if len(text) <= RERANKER_WINDOW_CHARS:
        return [text]
    step = RERANKER_WINDOW_CHARS - RERANKER_WINDOW_OVERLAP
    return [
        text[start : start + RERANKER_WINDOW_CHARS]
        for start in range(0, len(text), step)
        if text[start : start + RERANKER_WINDOW_CHARS].strip()
    ]


def retrieve(
    arxiv_id: str,
    query: str,
    *,
    top_k: int = 5,
    reranker: Callable[[str, list[str]], list[float]] | None = None,
) -> list[dict]:
    source = index_directory(arxiv_id)
    index_path, chunks_path = source / "index.faiss", source / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"Paper {arxiv_id} is not indexed.")
    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    metadata_path = source / "index_meta.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    query_vector = np.asarray([_embeddings().embed_query(query)], dtype="float32")
    faiss.normalize_L2(query_vector)
    candidate_count = min(max(top_k * 3, 20), len(chunks))
    scores, positions = index.search(query_vector, candidate_count)
    dense_by_position = {
        int(position): (rank, float(score))
        for rank, (score, position) in enumerate(zip(scores[0], positions[0], strict=True), 1)
        if position >= 0
    }
    lexical_scores = _lexical_scores(query, chunks)
    lexical_order = sorted(range(len(chunks)), key=lambda pos: lexical_scores[pos], reverse=True)
    lexical_rank = {
        position: rank
        for rank, position in enumerate(lexical_order, 1)
        if lexical_scores[position] > 0
    }
    lexical_candidate_count = candidate_count if reranker else top_k
    candidates = set(dense_by_position) | set(lexical_order[:lexical_candidate_count])
    fused = []
    for position in candidates:
        dense_rank, dense_score = dense_by_position.get(position, (None, 0.0))
        word_rank = lexical_rank.get(position)
        retrieval_score = (1 / (RRF_K + dense_rank) if dense_rank else 0.0) + (
            1 / (RRF_K + word_rank) if word_rank else 0.0
        )
        fused.append((retrieval_score, dense_score, dense_rank, word_rank, position))
    fused.sort(reverse=True)
    reranker_by_position: dict[int, float] = {}
    if reranker:
        windows: list[str] = []
        window_owners: list[int] = []
        for position in sorted(candidates):
            chunk_windows = _reranker_windows(chunks[position]["text"])
            windows.extend(chunk_windows)
            window_owners.extend([position] * len(chunk_windows))
        values = reranker(query, windows)
        if len(values) != len(windows):
            raise ValueError("Reranker returned the wrong score count.")
        reranker_by_position = {position: float("-inf") for position in candidates}
        for position, value in zip(window_owners, values, strict=True):
            reranker_by_position[position] = max(
                reranker_by_position[position], float(value)
            )
        fused.sort(
            key=lambda item: (reranker_by_position[item[-1]], -item[-1]),
            reverse=True,
        )
    results = []
    for retrieval_score, dense_score, dense_rank, word_rank, position in fused[:top_k]:
        item = dict(chunks[position])
        item.update(
            {
                "arxiv_id": arxiv_id,
                "versioned_id": metadata.get("versioned_id", arxiv_id),
                "score": dense_score,
                "retrieval_score": reranker_by_position.get(
                    position, retrieval_score
                ),
                "dense_rank": dense_rank,
                "lexical_rank": word_rank,
                "fusion_method": (
                    "cross_encoder_rerank" if reranker else "reciprocal_rank_fusion"
                ),
                "reranker_score": reranker_by_position.get(position),
            }
        )
        results.append(item)
    return results
