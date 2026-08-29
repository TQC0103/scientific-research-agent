# Project state

Last updated: 2026-08-29

## Current baseline

- Version: `0.4.0`
- Release status: v0.5 evaluation/grounding work is in progress on `main`;
  `v0.5.0` is not tagged yet.
- Branch: `main`
- Task 6 implementation commits: `174d1c3`, `6ab0811`; Task 9 commit:
  `1dc5f9d`; Task 7 commit: `0e77634`; Task 8 commits: `ea3a973`, `a4d37e7`;
  SciFact commit: `7959e07`; Task 10 commit: `70e762d`.
- Runtime: Python 3.11, Ollama, `qwen3:4b-instruct`,
  `qwen3-embedding:0.6b`
- Verification: JSON Schema 1.1.0, four fixtures, the ten-case development
  suite, the 22-case controlled verifier definition, citation fixtures, and the
  claim-verification contract, synthetic claim-verifier outputs, and Task 11
  report schema/node-stream/baseline behavior validated; Ruff passed and all 139
  pytest tests passed. Native QASPER loaded all 5,049
  questions and native SciFact loaded all 300 labeled dev claims. No local model
  benchmark was run.
  The earlier Ollama doctor and Gradio import checks remain the latest runtime
  smoke checks.

## Implemented

- Version-aware arXiv discovery, SQLite metadata, and FTS5 local-first search
- Lazy validated PDF ingestion with SHA-256 and abstract-only failure fallback
- Page/section-aware chunking and identity-checked per-paper FAISS indexes
- Hybrid dense + lexical retrieval with reciprocal-rank fusion
- Structured LLM evidence verifier, bounded query rewrite, and fail-closed stop
- Per-paper evidence/query/retry maps and required coverage for comparisons
- Verified-passage-only synthesis, fail-closed numeric citation labels, and
  deterministic citation metadata; missing/invalid labels never receive an
  automatic source
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
- Controlled verifier benchmark using the production prompt/parser, 22
  initial/recovery snapshots, fail-closed parsing, sufficiency FP/FN metrics,
  supported-passage metrics, bounded rewrite recovery, final abstention, model
  call/latency accounting, and an isolated pinned Qwen3-4B T4 package
- Deterministic citation-safety contract and evaluator with citation precision,
  completeness, unsupported-claim rate, invalid-citation rate, per-case
  diagnostics, null empty denominators, and JSON/Markdown reports
- Task 7 atomic-claim contract with answer-substring traceability, ordered
  numeric citations, per-evidence entails/partial/does-not-support relations,
  derived claim verdicts, strict cross-reference validation, generated JSON
  Schema, Task 9 metric adaptation, and six structural fixture shapes
- Task 8 structured claim-verifier callable with one bounded extraction/verification
  prompt over approved passages, immutable answer/evidence-count guards,
  Sources-block removal, strict Task 7 parsing, and fail-closed errors
- Task 10 production graph integration after citation-safe synthesis: structured
  claim verification, deterministic supported/repairable/unsupported routing,
  exactly one evidence-only repair, re-verification, and fail-closed abstention;
  graph state preserves attempts, revision count, history, and validation errors
- Task 11 end-to-end evaluator and `python -m evaluation.run` command over the
  production node-update stream, with versioned schema, automatic node/final-state
  traces, exact-suite identity, registered metrics, case-level failure isolation,
  runtime provenance, ignored JSONL/JSON/Markdown output, and informational
  baseline comparison without hard-coded thresholds
- Seven-case Task 8 synthetic development benchmark using the production
  prompt/parser, structural extraction/verdict/relationship metrics, fail-closed
  citation accounting, raw-response diagnostics, and a narrow pinned Qwen3-4B
  Kaggle T4 package
- Native-label SciFact oracle-document evaluator over the 300-case public dev
  split, with cited abstract preservation, strict structured output, fail-closed
  parsing, three-way macro F1, binary support diagnostics, best-gold rationale
  sentence F1/exact match, joint label+rationale accuracy, raw responses, and a
  checksum-pinned isolated Kaggle runner

## Current production request path

1. Discover exact papers or local/remote candidates and establish `any` versus
   required `all` paper coverage.
2. Reuse current indexes or lazily download, validate, parse, chunk, embed, and
   index selected revisions; fall back to labeled abstract evidence on failure.
3. Run hybrid retrieval separately per paper and let the evidence verifier retry
   with at most two focused query rewrites per paper.
4. If coverage remains insufficient, return paper-specific gaps without synthesis.
5. Otherwise synthesize only from approved passages, validate every numeric label,
   and construct source metadata from trusted records.
6. Verify atomic claims. Return when supported; revise partial/mixed content once
   and verify again; otherwise abstain. No branch can revise more than once.

This flow is shared by CLI and Gradio. CLI `--trace` currently prints discovery,
paper selection, coverage, and retrieval attempts, but not the new claim bundle
or revision diagnostics.

The evaluation runner observes the same compiled graph through LangGraph's update
stream. Its richer per-case artifact contains every node update and the complete
serializable final state; this does not change the user-facing CLI/UI trace.

## Latest evaluation

Before Task 10 integration, six fixed questions for `1706.03762v7` made the
expected decision in 6/6 cases: four evidence-backed answers and two correct
abstentions.
The positional-encoding case required one retrieval rewrite. This is a small
calibration set and is not a general accuracy estimate. It does not exercise the
new post-synthesis claim-verification or repair nodes.

The pre-Task10 multi-paper smoke used `1706.03762v7` and `1810.04805v2`. A
self-attention comparison passed with evidence and citations from both papers. A broader
architecture-and-training-objective comparison correctly stopped after the
Transformer side exhausted three verifier calls without direct loss/objective
evidence; the BERT side was independently sufficient after one call.
Its graph regression has since been updated to pass through a mocked supported
claim bundle, but no equivalent live Task 10 model run has been recorded yet.

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

The Task 6 verifier R2 run completed 22 controlled development cases with the
official FP16 `Qwen/Qwen3-4B` revision on Kaggle. Initial accuracy was `0.8636`,
false-positive rate `0.0000`, false-negative rate `0.3000`, supported-passage
precision/recall `0.6750/1.0000`, rewrite recovery `0.7000`, final abstention
accuracy `1.0000`, and bounded-flow accuracy `0.7273`, with zero parse failures.
All six flow failures involved comparisons: three complete positive snapshots
were rejected and three recoveries remained rejected. This is a repo-authored
development diagnostic, not held-out accuracy, and the FP16 Transformers runtime
is not bit-identical to the quantized production Ollama tag.

Task 9's five synthetic citation fixtures exercise one supported citation, one
wrong existing citation, one missing citation, one invalid evidence ID, and one
claim that needs no citation. Their aggregate `0.3333/0.5000/0.7500/0.3333`
precision/completeness/unsupported/invalid rates are intentionally imperfect
contract checks, not model-quality results. No LLM or GPU run was needed.

Task 7's structural fixture validates six claim shapes: fully supported,
partial, unsupported, wrong citation, citation not required, and required but
uncited. The bundle maps directly into Task 9 metrics with precision `0.2500`,
completeness `0.8000`, unsupported-claim rate `0.8000`, and invalid-citation
rate `0.0000`. These deliberately mixed values test the contract rather than a
model; no LLM or GPU run occurred.

Task 8's mocked-model unit suite exercises the same six shapes through the real
prompt, invocation, parser, and Task 7 validators. It additionally verifies
fenced JSON handling, prompt evidence numbering, immutable answer/evidence
inputs, deterministic Sources-block removal, malformed-verdict rejection, and
no model load for empty input. These are implementation tests, not live Qwen
accuracy. A separate live development diagnostic is recorded below; neither
suite is held-out or independently reviewed.

The corrected Task 8 R3 diagnostic ran seven synthetic cases through pinned
FP16 `Qwen/Qwen3-4B` on Kaggle. Schema validity was `0.8571`, exact source-span/
citation extraction `0.7143`, exact-case agreement `0.5714`, claim-verdict
accuracy `0.7500`, and evidence-relationship accuracy `0.8571`. Fail-closed
citation metrics were precision `0.5714`, completeness `0.8571`, unsupported
claim rate `0.4286`, and invalid-citation rate `0.0000`. The three non-exact
cases exposed distinct boundaries: partial versus unsupported labeling, a
hallucinated citation label on an uncited answer that strict parsing rejected,
and a harmless source-span punctuation difference on a compound sentence. The
seven repo-authored synthetic cases are diagnostic only.

The external SciFact R3 run processed all 300 public dev claims with their
cited abstracts supplied to pinned FP16 `Qwen/Qwen3-4B`. Native three-way label
accuracy was `0.7233` and macro F1 `0.7070`; per-label F1 was `0.8016` SUPPORT,
`0.7090` NOT_ENOUGH_INFO, and `0.6104` CONTRADICT. Rationale sentence F1 was
`0.7438`, exact rationale match `0.6330`, and joint label+rationale exact match
`0.5266`. Binary SUPPORT detection accuracy was `0.8300`, with false-positive/
false-negative rates `0.1705/0.1694`. Re-parsing the same saved outputs after
making the non-metric reason optional left one genuinely malformed response.
This is an external dev oracle-document diagnostic, not retrieval, Task 8, or
end-to-end LangGraph accuracy.

## Known issues

- Section detection can inherit incorrect labels around mid-page headings and
  tables (notably pages 6 and 8 of the Transformer paper).
- The 4B verifier required prompt calibration and schema consistency repair;
  the controlled Task 6 run now shows a 30% initial false-negative rate on
  positive cases, concentrated entirely in comparison scopes.
- Supported-passage precision was only 67.5% because the model often selected a
  topical distractor even while explaining that it did not answer the question.
  Passage-selection semantics need correction before claim verification relies
  on those indices.
- Repeated local verifier calls are slow on the 4 GB laptop GPU.
- Comparison intent without explicit paper IDs currently uses a bilingual
  keyword heuristic; it is not yet a general query planner.
- Per-paper coverage reduces evidence leakage, and Task 10 now checks the final
  answer claim by claim. Its accuracy still depends on the same 4B model that
  performs both claim extraction and entailment judgment.
- Numeric citation validation proves only that a label maps to an approved
  passage; semantic support is now checked separately and fails closed, but has
  not yet been validated on an independently reviewed end-to-end suite.
- The Task 7 contract proves structural traceability and verdict consistency,
  but cannot determine whether a paraphrase is truly atomic or a passage
  semantically entails it.
- Task 8 currently asks one 4B model call to perform both extraction and
  verification. The synthetic live diagnostic is useful for failure discovery,
  but exact-substring compliance, claim coverage, and entailment accuracy have
  not been measured on independently checked outputs.
- The Task 8 R3 run shows that a structurally constrained model can still invent
  a citation label absent from `source_text`; Task 10 now treats that parse
  failure as an abstention.
- Task 10 uses a conservative repair policy: partial claims and mixed supported/
  unsupported answers may be revised once, while an answer with no supported
  factual claim abstains immediately. It does not yet identify user-designated
  key claims separately or distinguish contradiction from missing evidence in
  its final user-facing abstention reason.
- CLI `--trace` and Gradio expose the final answer but do not yet render Task 10
  claim assessments, revision history, or claim-verifier errors. The state exists
  for Task 11 and future UI diagnostics.
- Task 11 has deterministic mocked-graph coverage but no live suite result yet.
  The committed runner uses configured production dependencies; the heavy first
  run must use a prepared Kaggle Control Plane environment rather than the laptop.
- End-to-end `answer_f1` is lexical overlap with the committed reference, not a
  semantic or LLM-judge score. Claim support rates are verifier judgments, not
  independently annotated entailment accuracy.
- LLM-node calls are counted from graph attempts, but embedding calls remain
  `null` because the retriever does not expose a reliable counter yet.
- Baseline comparison requires exact suite identity and reports any directional
  delta; it is informational and intentionally has no regression threshold until
  real baseline variance is observed.
- Exact source-span scoring is sensitive to punctuation boundaries in compound
  sentences, and the partial-versus-unsupported distinction needs independent
  annotation guidance before it becomes a regression threshold.
- arXiv `last_revised` search is bounded and not an exhaustive corpus harvest.
- SciFact's runner evaluates native three-way claim labels and rationale
  selection with cited documents supplied. Its result is not a score for
  retrieval, the binary evidence verifier, Task 8 `partial`, or LangGraph.
- SciFact R3 over-predicted CONTRADICT for 26/112 NOT_ENOUGH_INFO claims and
  achieved only `0.6104` CONTRADICT F1. Graph repair must distinguish direct
  contradiction from merely incomplete evidence rather than reuse one negative
  relationship blindly.
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

1. Package and run Task 11 through Kaggle Control Plane on the ten development
   cases, inspect the trace/failures, and only then save a v0.5 development
   baseline. Do not tag or set thresholds yet.
2. Instrument embedding calls only at the production retriever boundary.
3. Independently review the remaining eight answer cases before freezing any
   benchmark snapshot.
4. Independently review and expand the seven Task 8 development cases, then
   calibrate the partial-versus-unsupported boundary before freezing results.
5. Run Task 10 on reviewed cases and calibrate repair versus immediate
   abstention, including distinct incomplete-evidence and contradiction reasons.
6. Fix section boundaries and table-associated metadata.
