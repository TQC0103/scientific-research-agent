"""Run the production graph with pinned CUDA Hugging Face model adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from app.config import settings
from app.evaluation.end_to_end import NODE_TRACE_KEY, run_end_to_end, write_end_to_end_outputs
from app.evaluation.loader import load_suite

DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B"
DEFAULT_LLM_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


class TransformersRuntime:
    """One shared deterministic decoder-only model with LangChain-like wrappers."""

    def __init__(self, model_name: str, revision: str | None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("The end-to-end Transformers runtime requires CUDA.")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=torch.float16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.calls = 0

    def wrapper(self, *, num_predict: int = 1000, **_: Any) -> TransformersChat:
        return TransformersChat(self, max_new_tokens=num_predict)

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", padding=True).to("cuda:0")
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        self.calls += 1
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        del generated, output, inputs
        self.torch.cuda.empty_cache()
        return text


class TransformersChat:
    def __init__(self, runtime: TransformersRuntime, *, max_new_tokens: int) -> None:
        self.runtime = runtime
        self.max_new_tokens = max_new_tokens

    def invoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(
            content=self.runtime.generate(prompt, max_new_tokens=self.max_new_tokens)
        )


class SentenceTransformerEmbeddings:
    """Production vector-store protocol backed by a pinned SentenceTransformer."""

    def __init__(
        self,
        model_name: str,
        revision: str | None,
        *,
        batch_size: int,
        device: str,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        self.device = device
        self.batch_size = batch_size
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        prompt_name = "query" if "query" in getattr(self.model, "prompts", {}) else None
        return self.model.encode(
            [text],
            prompt_name=prompt_name,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()


def _install_adapters(
    llm: TransformersRuntime,
    embeddings: SentenceTransformerEmbeddings,
    expected_revisions: dict[str, str],
) -> Any:
    import app.agent.graph as graph_module
    import app.models.claim_verifier as claim_module
    import app.models.llm as llm_module
    import app.models.verifier as verifier_module
    import app.retrieval.vector_store as vector_module
    from app.db.database import get_paper

    original_metadata = graph_module.get_arxiv_metadata
    metadata_cache: dict[str, dict[str, Any]] = {}

    def pinned_metadata(paper_id: str) -> dict[str, Any]:
        base_id = paper_id.split("v", 1)[0]
        if base_id not in metadata_cache:
            paper = original_metadata(expected_revisions[base_id])
            actual = paper.get("versioned_id")
            if actual != expected_revisions[base_id]:
                raise ValueError(
                    f"Pinned arXiv revision mismatch for {base_id}: expected "
                    f"{expected_revisions[base_id]}, received {actual}."
                )
            metadata_cache[base_id] = paper
        return dict(get_paper(base_id) or metadata_cache[base_id])

    llm_module.get_llm = llm.wrapper
    verifier_module.get_llm = llm.wrapper
    claim_module.get_llm = llm.wrapper
    graph_module.get_arxiv_metadata = pinned_metadata
    vector_module._embeddings = lambda: embeddings
    settings.ollama_model = f"hf:{DEFAULT_LLM_MODEL}@{DEFAULT_LLM_REVISION}"
    settings.ollama_embed_model = (
        f"hf:{DEFAULT_EMBEDDING_MODEL}@{DEFAULT_EMBEDDING_REVISION}"
    )
    return graph_module.research_graph


def _invoke(graph: Any, payload: dict, config: dict) -> dict:
    state = dict(payload)
    events = []
    for update in graph.stream(payload, config, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node, values in update.items():
            node_update = dict(values) if isinstance(values, dict) else {}
            events.append({"node": str(node), "update": node_update})
            state.update(node_update)
    state[NODE_TRACE_KEY] = events
    return state


def _run_suite(suite_path: Path, output_dir: Path, graph: Any, *, limit: int = 0) -> Any:
    suite = load_suite(suite_path)
    if limit:
        suite = suite.model_copy(update={"cases": suite.cases[:limit]})
    report = run_end_to_end(
        suite,
        lambda payload, config: _invoke(graph, payload, config),
        config_name="hybrid_verified_citation_scoped_v4_qwen3_4b_fp16",
    )
    write_end_to_end_outputs(report, output_dir)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-revision", default=DEFAULT_LLM_REVISION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_EMBEDDING_REVISION)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--smoke-cases", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings.data_dir = args.data_dir
    settings.ensure_directories()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))["sources"]
    expected = {
        item["versioned_id"].rsplit("v", 1)[0]: item["versioned_id"] for item in sources
    }
    llm = TransformersRuntime(args.llm_model, args.llm_revision)
    embeddings = SentenceTransformerEmbeddings(
        args.embedding_model,
        args.embedding_revision,
        batch_size=args.embedding_batch_size,
        device="cuda:1" if llm.torch.cuda.device_count() > 1 else "cpu",
    )
    graph = _install_adapters(llm, embeddings, expected)

    smoke = _run_suite(args.suite, args.output_dir / "smoke", graph, limit=args.smoke_cases)
    if smoke.aggregate.execution_failures:
        raise RuntimeError("End-to-end smoke failed; the full suite was not started.")
    full = _run_suite(args.suite, args.output_dir / "full", graph)
    runtime = {
        "llm_physical_calls": llm.calls,
        "embedding_document_calls": embeddings.document_calls,
        "embedding_query_calls": embeddings.query_calls,
        "embedding_device": embeddings.device,
        "smoke_execution_failures": smoke.aggregate.execution_failures,
        "full_execution_failures": full.aggregate.execution_failures,
    }
    (args.output_dir / "adapter_runtime.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
