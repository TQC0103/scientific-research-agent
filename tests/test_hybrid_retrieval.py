from app.config import settings
from app.retrieval import vector_store
from app.retrieval.vector_store import _lexical_scores, build_index, retrieve


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def test_lexical_score_surfaces_exact_mechanism_terms() -> None:
    chunks = [
        {"text": "The model uses self-attention to process a sequence."},
        {
            "text": (
                "Since there is no recurrence or convolution, token order is injected "
                "with positional encodings."
            )
        },
    ]
    scores = _lexical_scores(
        "How is token order represented without recurrence or convolution?", chunks
    )
    assert scores[1] > scores[0]


def test_opt_in_reranker_scores_windows_and_reorders_candidates(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(vector_store, "_embeddings", lambda: FakeEmbeddings())
    long_evidence = "x" * 1200 + "late evidence marker" + "y" * 400
    build_index(
        "1234.56789",
        [
            {"text": "irrelevant candidate", "page": 1, "chunk_index": 0},
            {"text": long_evidence, "page": 2, "chunk_index": 0},
        ],
        paper={"versioned_id": "1234.56789v1", "version": 1},
        pdf_sha256="fixture-hash",
    )

    def score(_query: str, windows: list[str]) -> list[float]:
        return [1.0 if "late evidence marker" in window else 0.0 for window in windows]

    result = retrieve("1234.56789", "question", top_k=1, reranker=score)

    assert result[0]["page"] == 2
    assert result[0]["fusion_method"] == "cross_encoder_rerank"
    assert result[0]["reranker_score"] == 1.0
