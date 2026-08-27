# Project state

Last updated: 2026-08-27

## Current baseline

- Version: `0.4.0`
- Branch: `main`
- Baseline commit before this implementation: `34ceb54`
- Runtime: Python 3.11, Ollama, `qwen3:4b-instruct`,
  `qwen3-embedding:0.6b`
- Verification: JSON Schema 1.1.0, four fixtures, and the ten-case development
  suite validated; Ruff passed and all 72 pytest tests passed. Native QASPER
  loaded all 5,049 questions and native
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
- Ten-case internal development suite v0.1.2 over pinned Transformer v7 and
  BERT v2 sources: eight answer cases, two abstentions, multi-paper coverage,
  missing and unsupported evidence, a numeric ablation, and partial evidence
- Structured advisory LLM judge, separate report aggregation, and an isolated
  deterministic T4 package; judge output cannot change human-review/publication
  metadata
- Generated human-review HTML combining questions, expected decisions, criteria,
  evidence, challenge labels, and per-case judge findings
- Revision-safe internal retrieval evaluator with normalized quote matching,
  evidence-group Recall@K, annotation-relative Precision@K, MRR, item coverage,
  required-paper coverage, macro paper recall, per-case diagnostics, and
  JSON/JSONL reports
- Six-configuration internal retrieval diagnostic using identical verified
  PDFs, chunks, questions and total K: BM25, pinned Qwen3 dense, global RRF,
  min-max CombSUM, and per-paper variants of both fusion methods; outputs remain
  JSON/Markdown artifacts and production RRF is unchanged

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

The corrected internal-suite advisory R4 audit completed all 10 cases with
Qwen2.5-3B-Instruct on a Tesla T4. Kaggle capacity queueing took about 17 minutes;
once allocated, the job ran for about 4 minutes 36 seconds. All eight answer
cases passed, including the repaired single-head ablation case. The two
abstention cases received `needs_revision` because selected evidence cannot
verify document-wide absence. Those flags are routed to human review rather
than treated as dataset failures or an accuracy score. Mean lint scores were
4.8 clarity, 4.5 entailment, 4.5 answer alignment, 4.7 citation specificity, and
4.8 challenge validity.

The completed internal retrieval R4 run used the exact two pinned PDFs and two
visible Kaggle Tesla T4 devices. Across nine retrieval-eligible cases at K=5,
lexical Recall/Precision/MRR was `0.7778/0.2000/0.6667`, dense was
`0.7222/0.2222/0.6389`, and hybrid was `0.7222/0.2000/0.5000`; all arms produced
all 10 predictions. Lexical missed the BERT masked-LM case and the two-paper
comparison. Dense recovered masked-LM and one of the comparison's two evidence
groups but missed sinusoidal position encoding and GLUE/MultiNLI. Hybrid
inherited those dense misses under the current RRF. This development result does
not establish a hybrid advantage and contrasts with the larger QASPER dev run,
where hybrid had the highest Recall@5.

Internal retrieval R5 compared untuned fusion and paper-ranking variants on the
same inputs. Global and per-paper min-max CombSUM reached Recall@5 `0.8333`,
Precision@5 `0.2444`, and MRR `0.6204`. Both recovered sinusoidal position
encoding and GLUE/MultiNLI but lost masked-LM. Per-paper RRF stayed at Recall@5
`0.7222` and MRR `0.4944`; for the comparison case it swapped the covered paper
instead of covering both. This trade-off does not justify a production change.

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
- The internal quote matcher uses a documented 0.8 token-recall threshold that
  is unit-tested but not yet calibrated against real retrieved chunks; inspect
  match diagnostics during the first ablation before treating it as fixed.
- The internal baseline uses one global ranking across declared papers; R5 adds
  fixed-total-K per-paper diagnostics. Production instead runs a sequential
  per-paper verifier loop, so neither evaluation branch reproduces the complete
  graph trace.
- The R5 fusion diagnostic improved aggregate Recall with normalized CombSUM
  but swapped one successful case for two others and still missed one paper in
  the comparison. Nine eligible development cases are insufficient for fusion
  selection, statistical testing, or parameter optimization.
- The ten internal cases are repo-authored and tuned development data. The two
  negative cases were human-adjudicated after a full-paper audit on 2026-08-27,
  but the other eight cases still lack independent review and the suite remains
  development-only.
- Control Plane currently reports source-root validation failures as a generic
  offline error because its plugin catches HTTP 4xx as `URLError`; packages must
  be staged beneath the configured KCP experiments root until that UI/plugin
  diagnostic is fixed.
- The completed internal-judge R2 used an environment under `/kaggle/working`,
  so local result collection unnecessarily traverses the environment before the
  report. The committed template moves all non-report files to `/tmp`.
- R4 confirmed that the runner's inactive-sampling and deprecated `torch_dtype`
  warning cleanup works. Kaggle still emits non-fatal global
  `sitecustomize`/missing-`wrapt` warnings outside the isolated environment.

## Next priorities

1. Independently review the remaining eight answer cases before freezing any
   benchmark snapshot.
2. Build Task 6's verifier evaluator for sufficiency, false-positive/negative,
   abstention, and rewrite-recovery behavior.
3. Add citation safety and claim-level verification, then use native SciFact for
   claim-label/rationale evaluation.
4. Add end-to-end regression comparison without committing runtime outputs.
5. Fix section boundaries and table-associated metadata.
