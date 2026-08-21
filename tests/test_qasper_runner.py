import json
from types import SimpleNamespace

from app.evaluation.qasper_runner import (
    AbstainingGenerator,
    BM25Retriever,
    RetrievedParagraph,
    TransformersGenerator,
    load_qasper_papers,
    reciprocal_rank_fusion,
    run_qasper,
)


def _dataset(tmp_path):
    evidence = "The method uses contrastive learning to align image and text representations."
    payload = {
        "paper-a": {
            "title": "A paper",
            "abstract": "A short abstract.",
            "full_text": [
                {
                    "section_name": "Method",
                    "paragraphs": [
                        "Unrelated background about optimization.",
                        evidence,
                    ],
                }
            ],
            "figures_and_tables": [],
            "qas": [
                {
                    "question_id": "q1",
                    "question": "What aligns image and text representations?",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["contrastive learning"],
                                "free_form_answer": "",
                                "yes_no": None,
                                "evidence": [evidence],
                            }
                        }
                    ],
                }
            ],
        }
    }
    path = tmp_path / "qasper.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, evidence


def test_load_qasper_papers_excludes_annotations(tmp_path):
    path, evidence = _dataset(tmp_path)
    paper = load_qasper_papers(path)["paper-a"]
    assert evidence in paper.paragraphs
    assert not hasattr(paper, "qas")


def test_bm25_finds_exact_evidence(tmp_path):
    path, evidence = _dataset(tmp_path)
    paper = load_qasper_papers(path)["paper-a"]
    result = BM25Retriever(paper.paragraphs).retrieve("align image text representations", 1)
    assert result[0].text == evidence


def test_rrf_is_deterministic(tmp_path):
    path, _ = _dataset(tmp_path)
    paper = load_qasper_papers(path)["paper-a"]
    ranking = BM25Retriever(paper.paragraphs).retrieve("image text", 3)
    assert reciprocal_rank_fusion([ranking, ranking], top_k=2)[0].index == ranking[0].index


def test_lexical_smoke_reports_retrieval_metrics(tmp_path):
    path, _ = _dataset(tmp_path)
    rows, aggregate = run_qasper(
        path,
        source_split="dev",
        retrieval_mode="lexical",
        generator=AbstainingGenerator(),
        top_k=1,
    )
    assert rows[0]["retrieval_recall_at_k"] == 1.0
    assert aggregate["retrieval_mrr"] == 1.0
    assert aggregate["model_calls"] == 0
    assert aggregate["generation_batch_calls"] == 0


def test_runner_batches_all_generation_requests(tmp_path):
    path, _ = _dataset(tmp_path)

    class RecordingGenerator:
        model_calls = 0
        batch_calls = 0

        def generate_batch(self, requests):
            self.batch_calls += 1
            self.model_calls += len(requests)
            return [("contrastive learning", [0]) for _request in requests]

    generator = RecordingGenerator()
    rows, aggregate = run_qasper(
        path,
        source_split="dev",
        retrieval_mode="lexical",
        generator=generator,
        top_k=1,
    )
    assert rows[0]["predicted_answer"] == "contrastive learning"
    assert aggregate["model_calls"] == 1
    assert aggregate["generation_batch_calls"] == 1


def test_transformers_generator_sends_prompts_as_one_batched_pipeline_call():
    calls = []

    def fake_pipeline(prompts, **kwargs):
        calls.append((prompts, kwargs))
        return [
            [{"generated_text": '{"answer":"first","evidence_indices":[0]}'}],
            [{"generated_text": '{"answer":"second","evidence_indices":[0,99]}'}],
        ]

    generator = TransformersGenerator.__new__(TransformersGenerator)
    generator.pipeline = fake_pipeline
    generator.max_new_tokens = 64
    generator.batch_size = 8
    generator.model_calls = 0
    generator.batch_calls = 0
    context = [RetrievedParagraph(index=0, text="evidence", score=1.0)]
    results = generator.generate_batch([("question one", context), ("question two", context)])

    assert results == [("first", [0]), ("second", [0])]
    assert len(calls) == 1
    assert calls[0][1]["batch_size"] == 8
    assert generator.model_calls == 2
    assert generator.batch_calls == 1


def test_transformers_generator_configures_left_padding_and_deterministic_flags():
    tokenizer = SimpleNamespace(
        padding_side="right",
        pad_token_id=None,
        pad_token=None,
        eos_token_id=151645,
        eos_token="<|im_end|>",
    )
    generation_config = SimpleNamespace(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )
    text_pipeline = SimpleNamespace(
        tokenizer=tokenizer,
        model=SimpleNamespace(generation_config=generation_config),
    )

    TransformersGenerator._configure_pipeline(text_pipeline)

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<|im_end|>"
    assert generation_config.do_sample is False
    assert generation_config.temperature is None
    assert generation_config.top_p is None
    assert generation_config.top_k is None
