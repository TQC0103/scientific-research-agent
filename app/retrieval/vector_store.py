import json
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings

from app.config import settings
from app.tools.paper_download import safe_paper_id


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


def retrieve(arxiv_id: str, query: str, *, top_k: int = 5) -> list[dict]:
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
    scores, positions = index.search(query_vector, min(top_k, len(chunks)))
    results = []
    for score, position in zip(scores[0], positions[0], strict=True):
        if position >= 0:
            item = dict(chunks[int(position)])
            item.update(
                {
                    "arxiv_id": arxiv_id,
                    "versioned_id": metadata.get("versioned_id", arxiv_id),
                    "score": float(score),
                }
            )
            results.append(item)
    return results
