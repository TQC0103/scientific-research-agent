# Evaluation data contract

The committed evaluation data is a versioned source artifact. Runtime model
outputs and reports belong under `data/evaluations/` and remain Git-ignored.

## Layout

- `schema/evaluation-suite.schema.json`: JSON Schema Draft 2020-12 contract,
  currently schema version `1.1.0`.
- `suites/v0_5/schema_fixtures.json`: four illustrative cases; these are schema
  fixtures, not a reported benchmark.
- `app/evaluation/`: executable loader, semantic validator, public-dataset
  adapters, deterministic metrics, and a portable QASPER runner.
- `scripts/download_external_benchmarks.py`: checksum-pinned QASPER v0.3 and
  SciFact downloader. Downloads remain under ignored `data/evaluations/`.
- `kaggle/qasper_v0_5_r11/`: immutable source snapshot and provenance record
  for the completed external QASPER R11 development ablation.

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

## Retrieval matching contract

The initial v0.5 benchmark will use the following rules:

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

The exact quote-normalization and overlap threshold will be implemented and
tested with the loader/evaluator work. Task 1 deliberately defines the semantic
contract without adding executable validation.

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
