from app.retrieval.vector_store import _lexical_scores


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
