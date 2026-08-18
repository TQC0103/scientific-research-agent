# Project state

Last updated: 2026-08-18

## Current baseline

- Version: `0.4.0`
- Branch: `agent/structured-query-planner` (patch target)
- Base main commit for this implementation: `34ceb544`
- Runtime: Python 3.11, Ollama, `qwen3:4b-instruct`,
  `qwen3-embedding:0.6b`
- Verification: Ruff passed; 36 pytest tests passed in the repository development environment.
  Python compilation and `git diff --check` also passed during patch construction, together
  with 10 focused planner/discovery tests in an isolated stub harness before the full suite
  was executed locally.

## Implemented

- Version-aware arXiv discovery, SQLite metadata, and FTS5 local-first search
- Lazy validated PDF ingestion with SHA-256 and abstract-only failure fallback
- Page/section-aware chunking and identity-checked per-paper FAISS indexes
- Hybrid dense + lexical retrieval with reciprocal-rank fusion
- Structured semantic query planner for automatic discovery
- Deterministic planner validation against trusted candidate IDs and auto-index budgets
- Conservative `coverage=all` fallback when planner invocation or structured parsing fails
- Explicit paper IDs as hard constraints that bypass probabilistic planning
- Structured LLM evidence verifier, bounded query rewrite, and fail-closed stop
- Per-paper evidence/query/retry maps and required coverage for comparisons
- Verified-passage-only synthesis and deterministic citation metadata
- Shared LangGraph path for CLI `ask`, CLI `chat`, and Gradio UI
- Planner decisions, dimensions, warnings, and errors exposed through `chat --trace`
- Six-case baseline runner and complete architecture visualization
- Reproducible two-case multi-paper smoke runner

## Latest evaluation

On six fixed questions for `1706.03762v7`, the pre-planner workflow made the expected
decision in 6/6 cases: four evidence-backed answers and two correct abstentions.
The positional-encoding case required one retrieval rewrite. This is a small
calibration set and is not a general accuracy estimate.

Multi-paper smoke used `1706.03762v7` and `1810.04805v2`. A self-attention
comparison passed with evidence and citations from both papers. A broader
architecture-and-training-objective comparison correctly stopped after the
Transformer side exhausted three verifier calls without direct loss/objective
evidence; the BERT side was independently sufficient after one call.

The new structured planner still requires a labeled semantic-intent evaluation on the
local Qwen runtime; unit tests validate schema handling, policy normalization, candidate
allow-listing, explicit-ID bypass, automatic-budget enforcement, and conservative fallback.

## Known issues

- Section detection can inherit incorrect labels around mid-page headings and
  tables (notably pages 6 and 8 of the Transformer paper).
- The 4B verifier required prompt calibration and schema consistency repair;
  broader labeled evaluation is still needed.
- The new planner uses the same local 4B reasoning model, so planner latency and semantic
  calibration need measurement on realistic query sets.
- Conservative planner fallback intentionally prefers over-coverage/abstention to silently
  answering a multi-source question from one paper; this can reduce availability during
  planner outages.
- Repeated local reasoning-model calls are slow on the 4 GB laptop GPU.
- Per-paper coverage reduces evidence leakage, but a separate final
  claim-by-claim cross-paper answer verifier is not implemented.
- arXiv `last_revised` search is bounded and not an exhaustive corpus harvest.

## Next priorities

1. Build a labeled planner-intent set for semantic routing and fallback calibration.
2. Build a 50-question page-level evidence set.
3. Add claim-by-claim verification of the synthesized cross-paper answer.
4. Measure fused Recall@K plus planner coverage/abstention behavior.
5. Fix section boundaries and table-associated metadata.
6. Connect and use Kaggle Control Plane/GPU for parallel model/evaluation runs.
