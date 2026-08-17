import json
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


def build_index(arxiv_id: str, chunks: list[dict]) -> Path:
    if not chunks:
        raise ValueError("Cannot index a paper with no chunks.")
    destination = index_directory(arxiv_id)
    destination.mkdir(parents=True, exist_ok=True)
    vectors = np.asarray(_embeddings().embed_documents([c["text"] for c in chunks]), dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(destination / "index.faiss"))
    (destination / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def retrieve(arxiv_id: str, query: str, *, top_k: int = 5) -> list[dict]:
    source = index_directory(arxiv_id)
    index_path, chunks_path = source / "index.faiss", source / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"Paper {arxiv_id} is not indexed.")
    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    query_vector = np.asarray([_embeddings().embed_query(query)], dtype="float32")
    faiss.normalize_L2(query_vector)
    scores, positions = index.search(query_vector, min(top_k, len(chunks)))
    results = []
    for score, position in zip(scores[0], positions[0], strict=True):
        if position >= 0:
            item = dict(chunks[int(position)])
            item.update({"arxiv_id": arxiv_id, "score": float(score)})
            results.append(item)
    return results

