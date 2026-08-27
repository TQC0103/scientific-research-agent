# Evaluation data contract

The committed evaluation data is a versioned source artifact. Runtime model
outputs and reports belong under `data/evaluations/` and remain Git-ignored.

## Layout

- `schema/evaluation-suite.schema.json`: JSON Schema Draft 2020-12 contract,
  currently schema version `1.1.0`.
- `suites/v0_5/schema_fixtures.json`: four illustrative cases; these are schema
  fixtures, not a reported benchmark.
- `suites/v0_5/development_10.json`: eight answer and two abstention cases over
  exact Transformer v7 and BERT v2 sources; this is a tunable development suite,
  not held-out evidence.
- `suites/v0_5/development_10_sources.json`: source URLs, PDF hashes, and page
  counts used while checking the committed quotes and page anchors.
- `suites/v0_5/verifier_development.json`: 22 controlled initial/recovery
  evidence snapshots derived from the development suite; this is a prompt and
  flow diagnostic, not an independently labeled benchmark.
- `suites/v0_5/citation_safety_fixtures.json`: five synthetic records covering
  supported, wrong, missing, invalid, and citation-free claims; scores validate
  metric semantics only.
- `app/evaluation/`: executable loader, semantic validator, public-dataset
  adapters, deterministic metrics, and a portable QASPER runner.
- `scripts/download_external_benchmarks.py`: checksum-pinned QASPER v0.3 and
  SciFact downloader. Downloads remain under ignored `data/evaluations/`.
- `kaggle/qasper_v0_5_r11/`: immutable source snapshot and provenance record
  for the completed external QASPER R11 development ablation.
- `kaggle/internal_judge_v0_5/`: isolated T4 template for advisory review of the
  ten internal cases.
- `kaggle/verifier_v0_5/`: isolated T4 template for the production verifier
  prompt/parser benchmark.

The former empty `questions.json` and `ground_truth.json` placeholders were
removed so there is one canonical data shape.

## Two evaluation tracks

Public numbers and repository regression numbers must never be merged:

1. **External benchmarks** preserve the native data and official semantics.
   QASPER measures full-paper answer and evidence F1 across 5,049 questions.
   SciFact provides SUPPORT, CONTRADICT, and NOT_ENOUGH_INFO claim labels plus
   rationale sentences. The adapters do not rewrite these into repo-authored
   arXiv fixtures.
2. **Internal suites** exercise exact arXiv revisions, page citations,
   required-paper coverage, query rewrites, abstention, and graph behavior.
   They are regression evidence, not generalization evidence.

Download and verify the public sources with:

```powershell
python scripts/download_external_benchmarks.py
```

The script pins each archive by SHA-256 and writes a runtime manifest. No model,
embedding, or GPU work is performed during download or parsing.

`scripts/run_qasper.py` reads native paper paragraphs without exposing gold
references to retrieval or generation. It supports BM25 lexical retrieval,
optional Sentence Transformers dense retrieval, reciprocal-rank hybrid fusion,
and optional Transformers answer generation. It writes per-case JSONL plus
aggregate official-style answer/evidence F1, retrieval Recall@K/MRR, latency,
and model-call counts under ignored runtime storage. Test-set execution is
blocked unless `--allow-test` is supplied explicitly.

`scripts/prepare_qasper_kaggle_job.py` creates the Control Plane source under
ignored `data/evaluations/kaggle_jobs/qasper_v0_5/`. The committed template in
`evaluation/kaggle/qasper_v0_5/` runs lexical and dense retrieval diagnostics
on dev, followed by hybrid retrieval plus answer generation. It never packages
the QASPER test split. The prepared directory has exactly one top-level Python
entrypoint as required by Control Plane; the portable runner is packaged as
an embedded, path-checked ZIP inside `main.py`, because Kaggle kernel source
uploads do not retain arbitrary local modules, data files, or requirements
files. The reviewed pinned requirements are therefore embedded alongside the
application archive and materialized under `/kaggle/working`. At runtime it is
expanded under `/kaggle/working`, then the entrypoint downloads the pinned
QASPER train/dev archive, verifies its SHA-256, and extracts only the dev member
before evaluation. Model dependencies install into a clean
`/kaggle/working/qasper_env` virtual environment with no system-site packages.
Because Kaggle's stdlib `venv` lacks a working `ensurepip`, the entrypoint first
bootstraps pinned `virtualenv` into ignored working storage; it does not modify
the system package directory.
The job requires a Tesla T4, performs an actual CUDA matrix operation, and
writes `runtime.json` plus `resolved-requirements.txt` before evaluation; a
failed environment preflight cannot consume the full benchmark loop.

QASPER's native multiple references are retained, and its answer/evidence
metrics take the best score across annotator references, matching the released
evaluator. SciFact's native document and rationale sentence IDs are retained.
SciFact is a target for the future claim verifier; it must not be presented as
an evaluation of the current binary evidence-sufficiency verifier.

## Provenance and publication gate

Every internal case records `evaluation_split`, `provenance`, and `annotation`.
Every suite records `benchmark_status`:

- `schema_fixture`: demonstrates structure only; scores are prohibited;
- `development`: may be used for implementation and prompt tuning;
- `frozen`: immutable held-out evaluation source with a `frozen_at` timestamp.

The executable `assert_publishable()` gate accepts only frozen suites with test
cases. Repo-curated test cases additionally require an independent reviewer and
adjudication; synthetic test cases are rejected. This is a guardrail against
reporting tuned fixtures as held-out benchmark results.

## Stable identity and references

Every case pins each paper to an exact arXiv revision. A gold passage is anchored
by `versioned_id`, `source_type`, `page`, and an exact `quote`. `chunk_index` is
optional because it can change when chunking configuration or section detection
changes; it must never be the only gold identifier.

`answer_criteria` are atomic rubric items. Every gold evidence item lists the
criteria it supports. Multiple interchangeable passages may share an
`evidence_group_id`, so a system can satisfy the group without retrieving every
duplicate passage.

`required_paper_ids` uses versioned IDs. An answer case is complete only when all
required criteria and required papers are supported. Cross-paper evidence must
not satisfy the missing side of another paper.

## Decision contract

`expected.decision` is either `answer` or `abstain`. Abstention cases additionally
identify one of these reasons:

- `evidence_missing`: the requested value or fact is not established by the
  pinned source, even if related evidence exists;
- `unsupported_question`: the question assumes an experiment, result, or premise
  that the source does not support;
- `unresolved_conflict`: available evidence conflicts and cannot support one
  unqualified answer.

`forbidden_claims` records high-risk outputs that must not appear. `challenge`
labels negative, adversarial, partial, conflicting, or cross-paper-leakage
conditions without overloading the main question type.

## Citation safety contract

`app/evaluation/citations.py` scores explicit atomic claim records against stable
evidence IDs. It deliberately does not split prose or decide entailment; those
are later claim-verifier responsibilities. Each record states whether a claim
requires citation, the evidence IDs it cites, the evidence available to the
answer, and the independently assigned IDs that support it.

The four aggregate rates are:

- `citation_precision`: supporting citation assignments divided by all citation
  assignments; unknown IDs and wrong-but-existing evidence both count against it;
- `citation_completeness`: citation-required claims with at least one available
  citation divided by all citation-required claims;
- `unsupported_claim_rate`: citation-required claims with no cited supporting
  evidence divided by all citation-required claims;
- `invalid_citation_rate`: cited IDs outside the answer's available evidence set
  divided by all citation assignments.

Metrics with an empty denominator are `null`, not zero. This distinguishes “not
applicable” from perfect or failed performance. Duplicate IDs and unknown gold
support references are rejected; predicted unknown citations remain valid input
because measuring them is the purpose of `invalid_citation_rate`.

Run the deterministic evaluator with:

```powershell
python scripts/evaluate_citations.py `
  --suite evaluation/suites/v0_5/citation_safety_fixtures.json `
  --output-dir data/evaluations/runs/citation-safety-fixtures
```

The committed fixture intentionally yields imperfect numbers to exercise every
metric. It is not generated-model output and is prohibited from being presented
as a quality benchmark. Generated JSON and Markdown reports remain Git-ignored.

## Retrieval matching contract

The internal evaluator implements the following rules:

1. Match only evidence from the same `versioned_id`.
2. Prefer normalized quote containment or token overlap within the retrieved
   chunk. Page equality alone is diagnostic, not sufficient when a quote is
   available.
3. Treat any matching member of an `evidence_group_id` as satisfying that group.
4. Report evidence-group Recall@K and MRR. For multi-paper cases, also report
   required-paper coverage and macro-average paper recall.
5. Report Precision@K only as annotation-relative precision because the gold set
   is intentionally non-exhaustive; do not interpret unlabeled relevant chunks
   as proven false positives.
6. Retrieval Recall@K, Precision@K, and MRR are `null`/not applicable for cases
   with no gold evidence. Evaluate those cases through verifier and end-to-end
   abstention metrics instead of inserting zeroes into retrieval averages.
7. Compute aggregate metrics over eligible cases and publish the eligible-case
   denominator with every metric.

`app/evaluation/retrieval.py` normalizes Unicode, case, and punctuation, then
accepts exact normalized quote containment or at least 0.8 multiset token recall
of the gold quote. It always requires the same `versioned_id`; page equality is
diagnostic only. Precision@K uses K as its denominator and is explicitly
annotation-relative because the gold passages are not exhaustive.

The evaluator consumes JSONL with one row per case:

```json
{"case_id":"transformer_scaled_dot_product_reason","retrieved":[{"versioned_id":"1706.03762v7","page":4,"text":"..."}]}
```

Run it without a model:

```powershell
python scripts/evaluate_internal_retrieval.py `
  --suite evaluation/suites/v0_5/development_10.json `
  --retrievals data/evaluations/runs/internal-retrieval/retrieved.jsonl `
  --config-name lexical-current --top-k 5 `
  --output-dir data/evaluations/runs/internal-retrieval/scored
```

It writes aggregate `metrics.json` and `per_case.jsonl`, including unmatched
evidence groups, missing required papers, and explicit missing-case counts.
Runtime inputs and reports remain ignored. Retriever execution and metric
calculation remain separate.

`app/evaluation/internal_retrieval_runner.py` builds one checksum-verified,
page-aware chunk corpus and uses it unchanged for every configuration. Lexical
uses BM25; dense uses the pinned `Qwen/Qwen3-Embedding-0.6B` Sentence
Transformers revision with its query prompt. Hybrid diagnostics include the
production-equivalent RRF constant 60, min-max-normalized CombSUM, and both
fusion methods with per-paper rank reset plus a fair quota inside the same total
K. Global RRF remains the production default.

The runner writes each arm's ranked chunks, per-case metrics, aggregate metrics,
`ablation_summary.json`, and `ablation_report.md`. Gold evidence is consulted
only after all rankings have been produced. The portable Kaggle package embeds
only the required code, downloads the two pinned PDFs remotely, verifies their
SHA-256 before parsing, uses an isolated `--system-site-packages` environment
without replacing Kaggle PyTorch, verifies actual T4 devices and CUDA execution,
and removes the environment before artifact collection.

The completed Kaggle R4 run used two visible Tesla T4 devices and scored all 10
cases (nine with retrieval gold, one no-gold abstention). At K=5, lexical
Recall/Precision/MRR was `0.7778/0.2000/0.6667`, dense was
`0.7222/0.2222/0.6389`, and hybrid was `0.7222/0.2000/0.5000`, with zero missing
predictions. Dense recovered the masked-LM case and half of the multi-paper
comparison, but missed the sinusoidal-position and GLUE/MultiNLI cases that
lexical recovered. Current RRF inherited those dense misses, so the internal
result does not show a hybrid win. The suite has only nine eligible,
repo-authored development cases; use the report for failure analysis and keep
the external QASPER result as the stronger broad retrieval signal.

R5 compared the four hybrid configurations without optimizing weights or RRF K.
Global min-max CombSUM and its per-paper counterpart both produced Recall@5
`0.8333`, Precision@5 `0.2444`, and MRR `0.6204`. They recovered sinusoidal
position encoding and GLUE/MultiNLI but lost the masked-LM hit. RRF per-paper
balancing stayed at Recall@5 `0.7222` and reduced MRR slightly to `0.4944`; in
the multi-paper case it swapped which paper was covered rather than reaching
both. No variant was promoted into production.

The diagnostic design follows the separation used by
[BEIR](https://github.com/beir-cellar/beir),
[MTEB](https://github.com/embeddings-benchmark/mteb), and
[Pyserini](https://github.com/castorini/pyserini): keep retrieval runs
reproducible and compare sparse, dense, and hybrid rankings independently of
generation. [ranx fusion](https://amenra.github.io/ranx/fusion/) provides the
specific precedent for RRF and normalized CombSUM. Its parameter optimization
was deliberately not used because this internal suite has no separate tuning
and held-out partitions. BEIR-style NDCG/MAP are also deferred until the
evidence-group annotations can be represented as defensible chunk-level qrels;
inventing relevance labels from overlapping chunks would make those metrics
misleading.

## Controlled verifier evaluation

`app/evaluation/verifier.py` materializes controlled evidence snapshots from
the exact gold quotes in `development_10.json`, then exercises the same prompt,
JSON parser, and fail-closed repairs used by `app/models/verifier.py`. Every
case has an initial snapshot and a recovery snapshot. Recovery is executed once
only when the initial decision is insufficient, matching the bounded production
behavior without constructing an open-ended agent loop.

The report separates initial sufficiency accuracy and false-positive/negative
rates from supported-passage micro precision/recall, rewrite proposal and
execution rates, recoverable-case success, final abstention accuracy, parsing
failures, latency, and model calls. `flow_accuracy` additionally requires the
correct initial rejection before counting a recovery; an unsafe initial approval
cannot be relabeled as successful recovery.

Prepare the isolated job with:

```powershell
python scripts/prepare_verifier_kaggle_job.py
```

The package runs a two-case structured-output smoke before the 22-case suite,
uses deterministic left-padded generation, pins the official
`Qwen/Qwen3-4B` revision, verifies both visible T4s with a real CUDA operation,
and places inference on device 0. The production Ollama tag is a quantized build
of the same model family, so this run tests prompt behavior but is not bit-exact
production-runtime reproduction.

Kaggle R2 completed all 22 cases with zero parse failures. Initial accuracy was
`0.8636`, false-positive rate `0.0000`, false-negative rate `0.3000`, supported
passage precision/recall `0.6750/1.0000`, rewrite recovery `0.7000`, final
abstention accuracy `1.0000`, and bounded-flow accuracy `0.7273`. The six flow
failures were all comparison cases: three already-complete positive snapshots
were rejected, and three comparison recoveries stayed rejected. The passage
precision error is also systematic: the model often listed a topical passage as
supporting while its own reason correctly said that passage lacked the requested
fact. These are development findings to guide later verifier changes, not
held-out quality claims.

## Advisory LLM judge

`app/evaluation/judge.py` builds a case-local audit prompt and validates a
structured response. The five 1–5 dimensions are question clarity, evidence
entailment, reference-answer alignment, citation specificity, and challenge
validity. Verdicts are `pass`, `needs_revision`, or `fail`; findings and a short
rationale remain visible per case.

Judge reports are generated artifacts under ignored `data/evaluations/`. They
never mutate the suite, increase `reviewer_count`, set `adjudicated`, or satisfy
`assert_publishable()`. Every abstention is forced to retain
`human_review_required=true`, because a model shown selected evidence cannot
establish that a fact is absent everywhere in a paper. This makes the judge a
useful annotation lint pass, not an accuracy metric or an independent reviewer.

`scripts/run_evaluation_judge.py` uses a dynamically imported Transformers GPU
runtime. The Kaggle package sets left padding for decoder-only generation, uses
`do_sample=False`, batches two cases at a time, verifies Tesla T4 capability 7.5
with a real CUDA operation, and records the resolved environment. Do not run the
model workload on the laptop. Code, model cache, and the isolated environment
live under Kaggle's temporary filesystem; only the small report directory is
placed under `/kaggle/working` for artifact download.

The completed R4 development audit used Qwen2.5-3B-Instruct on a Tesla T4 and
returned eight `pass` verdicts and two `needs_revision` verdicts. All eight
answer cases passed, including the corrected single-head attention ablation
case. The two flagged cases were the remaining abstentions; selected evidence
cannot prove document-wide absence, so they remain queued for human full-paper
review rather than being rewritten automatically. Mean scores were 4.8 question
clarity, 4.5 evidence entailment, 4.5 answer alignment, 4.7 citation specificity,
and 4.8 challenge validity. These are annotation-lint signals from one small
judge model, not benchmark accuracy.

The repository owner subsequently human-reviewed and retained the two
abstention cases on 2026-08-27 after a full-paper audit. Their annotation
metadata records that adjudication in suite v0.1.2. The other eight cases remain
unreviewed development annotations, so the suite is still neither frozen nor
publishable.

Render a human-readable review page by combining the committed suite with an
ignored judge report:

```powershell
python scripts/render_evaluation_review.py `
  --suite evaluation/suites/v0_5/development_10.json `
  --report data/evaluations/runs/internal_judge_v0_5_r4/judge_report.json `
  --output data/evaluations/runs/internal_judge_v0_5_r4/review.html
```

The page places flagged cases first and shows the question, expected response,
criteria, forbidden claims, gold passages, challenge labels, judge scores, and
findings together. It is a generated review aid and remains Git-ignored.

## Authoring rules

- Keep `case_id`, criterion IDs, evidence IDs, and evidence-group IDs unique in
  their relevant scope.
- Ensure `paper_id`, `versioned_id`, and `revision` agree.
- Ensure every `required_paper_id` and every gold evidence source appears in the
  case's `papers` list.
- Ensure every `supports` value names an answer criterion in the same case.
- For `answer`, provide at least one required paper, reference answer, criterion,
  and gold evidence group.
- For missing-evidence and unsupported-question abstentions, normally leave
  `answer_criteria` and `gold_evidence` empty. Partial/conflict abstentions may
  retain criteria and evidence that support only part of the requested answer;
  the Task 2 semantic validator will require the challenge label to agree.

JSON Schema covers the structural constraints. Cross-reference uniqueness and
the final answer/partial/conflict invariants are enforced by the semantic loader
in `app/evaluation/loader.py` and `app/evaluation/models.py`.

## External QASPER checkpoint

The final self-contained Kaggle source that produced the 2026-08-21 R11 result
is preserved at `evaluation/kaggle/qasper_v0_5_r11/main.py`. Its adjacent README
records the runtime, source hashes, configuration, metrics, limitations, and
output policy.

R11 produced all 1,005 QASPER v0.3 development predictions. Across 892
retrieval-eligible cases, Recall@5 was `0.4605` lexical, `0.4957` dense, and
`0.5237` hybrid. Hybrid MRR was `0.3886`, evidence F1 `0.2396`, and answer F1
`0.1651`. This is reproducible external evidence for retaining hybrid retrieval,
not a production-quality result or a replacement for the independently reviewed
internal suite. Only hybrid used the Qwen generator, so answer F1 is not an
apples-to-apples retrieval ablation.
