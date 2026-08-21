"""Portable QASPER retrieval and answer runner.

The runner consumes native QASPER JSON directly. Gold annotations are used only
after prediction, so retrieval and generation cannot inspect references.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from app.evaluation.external import QasperCase, load_qasper
from app.evaluation.metrics import qasper_metrics

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class QasperPaper:
    paper_id: str
    title: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedParagraph:
    index: int
    text: str
    score: float


class Generator(Protocol):
    model_calls: int
    batch_calls: int

    def generate(
        self, question: str, contexts: list[RetrievedParagraph]
    ) -> tuple[str, list[int]]: ...

    def generate_batch(
        self, requests: list[tuple[str, list[RetrievedParagraph]]]
    ) -> list[tuple[str, list[int]]]: ...


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_qasper_papers(path: str | Path) -> dict[str, QasperPaper]:
    """Load only paper text, intentionally excluding QA annotations."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load QASPER papers from {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("QASPER root must be an object keyed by paper ID.")
    papers: dict[str, QasperPaper] = {}
    for paper_id, value in payload.items():
        if not isinstance(value, dict) or not isinstance(value.get("full_text"), list):
            raise TypeError(f"Invalid QASPER paper entry: {paper_id}")
        paragraphs = []
        abstract = value.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            paragraphs.append(abstract.strip())
        elif isinstance(abstract, list):
            paragraphs.extend(str(item).strip() for item in abstract if str(item).strip())
        for section in value["full_text"]:
            if not isinstance(section, dict) or not isinstance(section.get("paragraphs"), list):
                raise TypeError(f"Invalid QASPER section in paper {paper_id}")
            paragraphs.extend(
                paragraph.strip()
                for paragraph in section["paragraphs"]
                if isinstance(paragraph, str) and paragraph.strip()
            )
        papers[str(paper_id)] = QasperPaper(
            paper_id=str(paper_id),
            title=str(value.get("title") or paper_id),
            paragraphs=tuple(dict.fromkeys(paragraphs)),
        )
    return papers


class BM25Retriever:
    def __init__(self, paragraphs: tuple[str, ...], *, k1: float = 1.5, b: float = 0.75):
        self.paragraphs = paragraphs
        self.k1 = k1
        self.b = b
        self.tokens = [_tokens(paragraph) for paragraph in paragraphs]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        count = len(self.tokens)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def retrieve(self, query: str, top_k: int) -> list[RetrievedParagraph]:
        query_tokens = _tokens(query)
        scored = []
        for index, tokens in enumerate(self.tokens):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                normalizer = frequency + self.k1 * (
                    1 - self.b
                    + self.b * self.lengths[index] / max(self.average_length, 1.0)
                )
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / normalizer
            scored.append(RetrievedParagraph(index=index, text=self.paragraphs[index], score=score))
        return sorted(scored, key=lambda item: (-item.score, item.index))[:top_k]


class SentenceTransformerRetriever:
    def __init__(self, paragraphs: tuple[str, ...], model_name: str, *, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense QASPER retrieval requires the optional sentence-transformers package."
            ) from exc
        self.paragraphs = paragraphs
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embeddings = self.model.encode(
            list(paragraphs),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedParagraph]:
        query_vector = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ query_vector
        ranked = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
        return [
            RetrievedParagraph(index=index, text=self.paragraphs[index], score=float(scores[index]))
            for index in ranked[:top_k]
        ]


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedParagraph]], *, top_k: int, constant: int = 60
) -> list[RetrievedParagraph]:
    scores: dict[int, float] = {}
    text: dict[int, str] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item.index] = scores.get(item.index, 0.0) + 1 / (constant + rank)
            text[item.index] = item.text
    ranked = sorted(scores, key=lambda index: (-scores[index], index))[:top_k]
    return [RetrievedParagraph(index=index, text=text[index], score=scores[index]) for index in ranked]


class AbstainingGenerator:
    """No-model mode for retrieval smoke tests; answer scores are not model results."""

    model_calls = 0
    batch_calls = 0

    def generate(
        self, question: str, contexts: list[RetrievedParagraph]
    ) -> tuple[str, list[int]]:
        del question
        return "Unanswerable", list(range(len(contexts)))

    def generate_batch(
        self, requests: list[tuple[str, list[RetrievedParagraph]]]
    ) -> list[tuple[str, list[int]]]:
        return [self.generate(question, contexts) for question, contexts in requests]


class TransformersGenerator:
    def __init__(
        self, model_name: str, *, max_new_tokens: int = 256, batch_size: int = 8
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Model generation requires the optional transformers package.") from exc
        self.pipeline = pipeline(
            "text-generation", model=model_name, device_map="auto", trust_remote_code=False
        )
        self._configure_pipeline(self.pipeline)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.model_calls = 0
        self.batch_calls = 0

    @staticmethod
    def _configure_pipeline(text_pipeline: Any) -> None:
        """Configure deterministic decoder-only generation for padded batches."""
        tokenizer = text_pipeline.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        generation_config = getattr(text_pipeline.model, "generation_config", None)
        if generation_config is not None:
            generation_config.do_sample = False
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None

    @staticmethod
    def _prompt(question: str, contexts: list[RetrievedParagraph]) -> str:
        numbered = "\n\n".join(
            f"[{index}] {item.text}" for index, item in enumerate(contexts)
        )
        return (
            "Answer the scientific-paper question using only the supplied paragraphs. "
            "If they do not establish an answer, answer Unanswerable. Return JSON only as "
            '{"answer":"...","evidence_indices":[0]}. Evidence indices must identify only '
            f"paragraphs that directly support the answer.\n\nQuestion: {question}\n\n{numbered}"
        )

    @staticmethod
    def _parse_output(output: Any, context_count: int) -> tuple[str, list[int]]:
        if isinstance(output, list):
            output = output[0] if output else {}
        generated = output.get("generated_text", "") if isinstance(output, dict) else output
        match = re.search(r"\{.*\}", str(generated), flags=re.DOTALL)
        if not match:
            return "Unanswerable", []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return "Unanswerable", []
        answer = str(payload.get("answer") or "Unanswerable").strip()
        indices = payload.get("evidence_indices") or []
        valid_indices = sorted(
            {
                int(index)
                for index in indices
                if isinstance(index, int) and 0 <= index < context_count
            }
        )
        return answer, valid_indices

    def _run_pipeline(self, prompts: list[str], batch_size: int) -> list[Any]:
        self.batch_calls += 1
        try:
            return self.pipeline(
                prompts,
                batch_size=batch_size,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).casefold() or batch_size == 1:
                raise
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, AttributeError):
                pass
            return self._run_pipeline(prompts, max(1, batch_size // 2))

    def generate(
        self, question: str, contexts: list[RetrievedParagraph]
    ) -> tuple[str, list[int]]:
        return self.generate_batch([(question, contexts)])[0]

    def generate_batch(
        self, requests: list[tuple[str, list[RetrievedParagraph]]]
    ) -> list[tuple[str, list[int]]]:
        if not requests:
            return []
        prompts = [self._prompt(question, contexts) for question, contexts in requests]
        self.model_calls += len(requests)
        outputs = self._run_pipeline(prompts, self.batch_size)
        if len(outputs) != len(requests):
            raise RuntimeError("Transformers pipeline returned an unexpected batch size")
        return [
            self._parse_output(output, len(contexts))
            for output, (_, contexts) in zip(outputs, requests, strict=True)
        ]


def _retrieval_scores(case: QasperCase, contexts: list[RetrievedParagraph]) -> tuple[float | None, float | None]:
    references = [set(reference.evidence) for reference in case.references if reference.evidence]
    if not references:
        return None, None
    predicted = [item.text for item in contexts]
    recall = max(len(set(predicted) & expected) / len(expected) for expected in references)
    union = set().union(*references)
    first_rank = next((rank for rank, paragraph in enumerate(predicted, 1) if paragraph in union), None)
    return recall, (1 / first_rank if first_rank else 0.0)


def run_qasper(
    dataset_path: str | Path,
    *,
    source_split: str,
    retrieval_mode: str,
    generator: Generator,
    top_k: int = 5,
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if retrieval_mode not in {"lexical", "dense", "hybrid"}:
        raise ValueError("retrieval_mode must be lexical, dense, or hybrid")
    cases = load_qasper(dataset_path, source_split=source_split, limit=limit)
    papers = load_qasper_papers(dataset_path)
    predictions: dict[str, dict[str, Any]] = {}
    rows = []
    retrieval_recalls = []
    reciprocal_ranks = []
    started = time.perf_counter()
    active_paper_id: str | None = None
    lexical: BM25Retriever | None = None
    dense: SentenceTransformerRetriever | None = None
    prepared = []
    for case in cases:
        case_started = time.perf_counter()
        paper = papers[case.paper_id]
        if case.paper_id != active_paper_id:
            active_paper_id = case.paper_id
            lexical = BM25Retriever(paper.paragraphs)
            dense = (
                SentenceTransformerRetriever(paper.paragraphs, dense_model)
                if retrieval_mode != "lexical"
                else None
            )
        assert lexical is not None
        if retrieval_mode == "lexical":
            contexts = lexical.retrieve(case.question, top_k)
        else:
            assert dense is not None
            dense_contexts = dense.retrieve(case.question, max(top_k, 20))
            if retrieval_mode == "dense":
                contexts = dense_contexts[:top_k]
            else:
                contexts = reciprocal_rank_fusion(
                    [lexical.retrieve(case.question, max(top_k, 20)), dense_contexts],
                    top_k=top_k,
                )
        retrieval_seconds = time.perf_counter() - case_started
        recall, reciprocal_rank = _retrieval_scores(case, contexts)
        if recall is not None:
            retrieval_recalls.append(recall)
            reciprocal_ranks.append(reciprocal_rank or 0.0)
        prepared.append((case, contexts, recall, reciprocal_rank, retrieval_seconds))

    generation_started = time.perf_counter()
    generated = generator.generate_batch(
        [(case.question, contexts) for case, contexts, *_rest in prepared]
    )
    generation_seconds_per_case = (
        (time.perf_counter() - generation_started) / len(prepared) if prepared else 0.0
    )
    for (case, contexts, recall, reciprocal_rank, retrieval_seconds), (
        answer,
        evidence_indices,
    ) in zip(prepared, generated, strict=True):
        evidence = [contexts[index].text for index in evidence_indices]
        predictions[case.question_id] = {"answer": answer, "evidence": evidence}
        rows.append(
            {
                "paper_id": case.paper_id,
                "question_id": case.question_id,
                "question": case.question,
                "predicted_answer": answer,
                "predicted_evidence": evidence,
                "retrieved": [asdict(item) for item in contexts],
                "retrieval_recall_at_k": recall,
                "retrieval_reciprocal_rank": reciprocal_rank,
                "latency_seconds": retrieval_seconds + generation_seconds_per_case,
            }
        )
    official = qasper_metrics(cases, predictions)
    aggregate = {
        **official,
        "retrieval_recall_at_k": (
            sum(retrieval_recalls) / len(retrieval_recalls) if retrieval_recalls else None
        ),
        "retrieval_mrr": (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None
        ),
        "retrieval_eligible_cases": len(retrieval_recalls),
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "dense_model": dense_model if retrieval_mode != "lexical" else None,
        "source_split": source_split,
        "model_calls": generator.model_calls,
        "generation_batch_calls": generator.batch_calls,
        "latency_seconds": time.perf_counter() - started,
    }
    return rows, aggregate
