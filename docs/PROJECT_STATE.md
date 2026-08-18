# Project state

Last updated: 2026-08-18

## Current baseline

- Version: `0.4.0`
- Branch: `main`
- Latest pushed commit before this implementation: `245844a`
- Runtime: Python 3.11, Ollama, `qwen3:4b-instruct`,
  `qwen3-embedding:0.6b`
- Verification: Ruff passed; 28 pytest tests passed; Ollama doctor and Gradio
  import passed

## Implemented

- Version-aware arXiv discovery, SQLite metadata, and FTS5 local-first search
- Lazy validated PDF ingestion with SHA-256 and abstract-only failure fallback
- Page/section-aware chunking and identity-checked per-paper FAISS indexes
- Hybrid dense + lexical retrieval with reciprocal-rank fusion
- Structured LLM evidence verifier, bounded query rewrite, and fail-closed stop
- Per-paper evidence/query/retry maps and required coverage for comparisons
- Verified-passage-only synthesis and deterministic citation metadata
- Shared LangGraph path for CLI `ask`, CLI `chat`, and Gradio UI
- Six-case baseline runner and complete architecture visualization
- Reproducible two-case multi-paper smoke runner

## Latest evaluation

On six fixed questions for `1706.03762v7`, the final workflow made the expected
decision in 6/6 cases: four evidence-backed answers and two correct abstentions.
The positional-encoding case required one retrieval rewrite. This is a small
calibration set and is not a general accuracy estimate.

Multi-paper smoke used `1706.03762v7` and `1810.04805v2`. A self-attention
comparison passed with evidence and citations from both papers. A broader
architecture-and-training-objective comparison correctly stopped after the
Transformer side exhausted three verifier calls without direct loss/objective
evidence; the BERT side was independently sufficient after one call.

## Known issues

- Section detection can inherit incorrect labels around mid-page headings and
  tables (notably pages 6 and 8 of the Transformer paper).
- The 4B verifier required prompt calibration and schema consistency repair;
  broader labeled evaluation is still needed.
- Repeated local verifier calls are slow on the 4 GB laptop GPU.
- Comparison intent without explicit paper IDs currently uses a bilingual
  keyword heuristic; it is not yet a general query planner.
- Per-paper coverage reduces evidence leakage, but a separate final
  claim-by-claim cross-paper answer verifier is not implemented.
- arXiv `last_revised` search is bounded and not an exhaustive corpus harvest.

## Next priorities

1. Add claim-by-claim verification of the synthesized cross-paper answer.
2. Build a 50-question page-level evidence set and measure fused Recall@K.
3. Fix section boundaries and table-associated metadata.
4. Connect and use Kaggle Control Plane/GPU for parallel model/evaluation runs.
