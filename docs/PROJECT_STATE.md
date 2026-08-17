# Project state

Last updated: 2026-08-18

## Current baseline

- Version: `0.3.0`
- Branch: `main`
- Latest implementation commit before this documentation update: `271997f`
- Runtime: Python 3.11, Ollama, `qwen3:4b-instruct`,
  `qwen3-embedding:0.6b`
- Verification: Ruff passed; 22 pytest tests passed; Ollama doctor and Gradio
  import passed

## Implemented

- Version-aware arXiv discovery, SQLite metadata, and FTS5 local-first search
- Lazy validated PDF ingestion with SHA-256 and abstract-only failure fallback
- Page/section-aware chunking and identity-checked per-paper FAISS indexes
- Hybrid dense + lexical retrieval with reciprocal-rank fusion
- Structured LLM evidence verifier, bounded query rewrite, and fail-closed stop
- Verified-passage-only synthesis and deterministic citation metadata
- Shared LangGraph path for CLI `ask`, CLI `chat`, and Gradio UI
- Six-case baseline runner and complete architecture visualization

## Latest evaluation

On six fixed questions for `1706.03762v7`, the final workflow made the expected
decision in 6/6 cases: four evidence-backed answers and two correct abstentions.
The positional-encoding case required one retrieval rewrite. This is a small
calibration set and is not a general accuracy estimate.

## Known issues

- Section detection can inherit incorrect labels around mid-page headings and
  tables (notably pages 6 and 8 of the Transformer paper).
- The 4B verifier required prompt calibration and schema consistency repair;
  broader labeled evaluation is still needed.
- Repeated local verifier calls are slow on the 4 GB laptop GPU.
- arXiv `last_revised` search is bounded and not an exhaustive corpus harvest.

## Next priorities

1. Build a 50-question page-level evidence set and measure fused Recall@K.
2. Fix section boundaries and table-associated metadata.
3. Connect and use Kaggle Control Plane/GPU for parallel model/evaluation runs.
