# Project state

Last updated: 2026-08-21

## Current baseline

- Version: `0.4.0`
- Branch: `codex/consolidate-v05-evaluation`
- Baseline commit before this implementation: `34ceb54`
- Runtime: Python 3.11, Ollama, `qwen3:4b-instruct`,
  `qwen3-embedding:0.6b`
- Verification: JSON Schema 1.1.0 and four fixtures validated; Ruff passed; 47
  pytest tests passed. Native QASPER loaded all 5,049 questions and native
  SciFact loaded all 300 labeled dev claims. No local model benchmark was run.
  The earlier Ollama doctor and Gradio import checks remain the latest runtime
  smoke checks.

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
- Versioned evaluation data contract with exact-revision papers, stable evidence
  anchors, answer/abstention expectations, challenge labels, and four schema
  fixtures
- Provenance, split/freeze metadata, and a publication gate that blocks fixture,
  synthetic, or unreviewed internal results from being presented as held-out
- Executable semantic suite loader plus native QASPER/SciFact adapters,
  deterministic metrics, and checksum-pinned public-dataset download
- Portable QASPER BM25/dense/hybrid runner with gold-hidden prediction,
  batched Transformers generation with CUDA-OOM batch fallback, explicit
  held-out access, and JSONL/metric outputs
- Archived self-contained QASPER R11 source with a pinned isolated T4 runtime,
  verified public-dataset download, source manifest, and recorded dev metrics

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

A two-question QASPER dev lexical/no-model smoke reached retrieval Recall@5
`0.75`, MRR `0.6667`, and evidence F1 `0.3095`. Its answer F1 `0.0` is expected
because the smoke generator always abstains; this is not an external model score.

The completed external QASPER v0.3 dev R11 run produced all 1,005 predictions.
Across 892 retrieval-eligible cases, Recall@5 was `0.4605` lexical, `0.4957`
dense, and `0.5237` hybrid. Hybrid MRR was `0.3886`, evidence F1 `0.2396`, and
answer F1 `0.1651`. This supports retaining hybrid retrieval but is not a strong
absolute-quality result. Answer F1 is not apples-to-apples because only hybrid
used the Qwen generator.

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
- SciFact evaluates scientific claim labels and rationales; the current verifier
  only judges evidence sufficiency, so using SciFact as its score would be invalid.
- External QASPER R11 still retrieves only about half of annotated evidence at
  K=5. Generation took roughly 7,332 seconds and the complete job roughly 8,013
  seconds on a Kaggle T4.
- R11 emitted non-fatal missing-`wrapt` sitecustomize warnings and ignored
  deterministic sampling flags; clean these before a future external rerun.
- Dense/generator modes depend on optional Sentence Transformers and
  Transformers runtimes; keep heavy runs on Kaggle rather than the laptop.

## Next priorities

1. Curate and independently review a 10-case internal regression suite.
2. Add retrieval and retrieval-rewrite metrics to the internal suite.
3. Evaluate verifier sufficiency and false-positive/false-negative behavior.
4. Add citation safety and claim-level verification, then use native SciFact for
   claim-label/rationale evaluation.
5. Add end-to-end regression comparison without committing runtime outputs.
6. Fix section boundaries and table-associated metadata.
