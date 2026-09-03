# Agentic Scientific Research Assistant

Local-first V1 for searching arXiv, lazily downloading papers, section-aware PDF
chunking, FAISS retrieval, and citation-grounded answers with LangGraph + Ollama.

The packaged project version is currently `0.4.0`. The evaluation and grounding
work described as v0.5 is implemented incrementally on `main`, but `v0.5.0` has
not been tagged yet.

## What is ready

- arXiv search with year/category filters and SQLite metadata persistence
- explicit first-submission vs last-revision dates and version-aware citations
- lazy PDF download, PyMuPDF parsing, section-aware chunking
- per-paper Ollama embeddings and FAISS indexes
- hybrid dense + lexical retrieval with reciprocal-rank fusion
- local Qwen evidence verification, query rewrite, and fail-closed abstention
- per-paper evidence coverage for explicit and automatically detected comparisons
- evidence retrieval with page/section citations
- a bounded LangGraph search/index/retrieve/answer loop
- CLI, Gradio UI, smoke tests, and local data isolation
- versioned evaluation schema with revision-pinned evidence fixtures
- validated QASPER/SciFact adapters and deterministic external metrics
- portable QASPER lexical/dense/hybrid runner with guarded test-set access
- ten-case internal development suite with an advisory structured LLM judge
- controlled verifier evaluator for sufficiency, passage selection, bounded
  rewrite recovery, and final abstention behavior
- fail-closed citation labels plus deterministic claim-to-evidence safety metrics
- versioned atomic-claim verification contract with strict cross-reference and
  verdict invariants
- bounded post-synthesis claim verification with one repair attempt and
  fail-closed abstention
- versioned end-to-end runner over production LangGraph with automatic node/state
  traces, explicit metric registration, and exact-suite baseline comparison

## Quick start (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
research-agent doctor
research-agent search "agentic RAG scientific question answering" --category cs.AI --year 2025
research-agent search "transformer attention" --year 2025 --date-mode last_revised
research-agent local-search "scientific retrieval"
research-agent index 2501.00001
research-agent ask "How does this paper reduce hallucination?" --paper-id 2501.00001
research-agent chat "Recent approaches to agentic RAG for scientific QA"
research-agent ui
```

Replace the sample arXiv ID with one returned by `search`. `chat` may search and
index up to two promising papers automatically. Both `ask` and `chat` use the
same evidence-verification workflow; `ask` limits discovery to the supplied papers.

The workflow does not treat vector similarity as proof. Qwen checks whether the
retrieved passages actually cover the question, identifies supported passages,
and proposes a focused retrieval query when information is missing. Only verified
passages reach answer synthesis. After two rewrites, unresolved questions return
an explicit insufficient-evidence response instead of a guessed answer.
The verifier-to-synthesis boundary also requires an internally complete result:
`sufficient=true`, at least one valid supporting passage, and no remaining
`missing_information`. Partial support may guide retrieval, but cannot authorize
synthesis.
For high-risk metric questions, a deterministic semantic-anchor guard also
rejects a positive verifier decision when its selected passages do not mention
the requested metric or benchmark (for example, training time is not electrical
energy and WMT BLEU is not ImageNet top-1 accuracy).

Synthesis no longer attaches `[1]` when the model omits citations. An answer
without a valid label, or with any label outside the verifier-approved evidence
set, is discarded and replaced by an explicit citation-grounding failure. Valid
labels are still resolved to trusted version, title, page, and section metadata
by code rather than by the model.

After citation-safe synthesis, the production graph performs atomic claim
verification against those same approved passages. Answers whose factual claims
are all supported return immediately. Partial claims, or a mix of supported and
unsupported claims, may be revised once using only approved evidence and are then
verified again. Code first splits the answer into immutable citation-scoped
spans; the model selects `span_#` IDs and returns evidence relationships in source order,
while code assigns claim IDs, restores `source_text`/citation labels, and derives
verdicts. A malformed response receives exactly one compact structure-
only retry without repeating the long evidence prompt; a second invalid response
fails closed. Wholly unsupported answers, citation failures, repair
failures, and unresolved claims after that one revision all produce an explicit
abstention. Both bounds are fixed, not open-ended agent loops.

## Current production flow

```text
question
  -> discover papers
  -> index/download as needed
  -> hybrid retrieve per paper (max 8 accumulated passages per paper)
  -> evidence sufficiency check + completeness invariant + semantic metric anchors
       (up to 2 query rewrites per paper)
  -> synthesize from approved passages only
  -> validate citation labels and restore trusted source metadata
  -> bind exact citation-scoped answer spans and verify atomic claims
       (at most 1 compact structure-only output retry per check)
       -> all supported: return answer
       -> partial or mixed: revise once, then verify again
       -> unsupported, invalid, or still failing: abstain
```

The specialized evaluation commands below remain useful diagnostics. Task 11
also provides one end-to-end runner over the compiled production graph. LangGraph
node updates and the final state are stored for every case, so later nodes and
fields appear automatically. Aggregate scores remain opt-in: genuinely new
capabilities need an explicit metric definition instead of receiving a guessed
score.

`MAX_ACCUMULATED_PASSAGES_PER_PAPER` defaults to `8` in `.env.example`. It
bounds the verifier prompt across retrieval rewrites; increasing it raises
context and accelerator-memory cost and requires a new benchmark config label.
`MAX_VERIFIER_PASSAGES_PER_PAPER` defaults to `6` and independently caps the
ranked prefix sent to each verifier call. The graph may still retain eight
passages for later retrieval/synthesis bookkeeping; keeping a prefix preserves
the verifier's one-based passage IDs against that stored list.

Repeated `--paper-id` options activate required coverage for every supplied
paper. Comparison-style questions without explicit IDs require the first two
discovered papers. Retrieval queries, retry counts, verifier decisions, and
approved passages are isolated per paper, so evidence from one source cannot
fill a missing side of another source. Use `chat --trace` to inspect per-paper
attempt counts and the aggregate coverage decision.

## Data layout

Runtime artifacts stay under `data/` and are ignored by Git:

- `data/research.db`: paper metadata
- `data/papers/`: downloaded PDFs
- `data/indexes/<arxiv-id>/`: FAISS index and chunk metadata
- `data/evaluations/`: generated evaluation runs and reports

Committed evaluation source data lives under `evaluation/`. Its JSON Schema,
evidence identity rules, decision taxonomy, and metric contract are documented
in `evaluation/README.md`. The initial four cases are schema fixtures.
`development_10.json` is the immutable repo-authored R10 regression slice,
while `development_25.json` preserves those ten cases exactly and adds 15
source-audited ResNet, LoRA, and RAG cases. All remain tunable development data
rather than held-out or publishable benchmarks. The 25-case suite is an
authored evaluation artifact; no R25 model score is claimed yet.

Download the checksum-pinned public QASPER v0.3 and SciFact artifacts with:

```powershell
python scripts/download_external_benchmarks.py
```

The downloaded datasets remain ignored runtime data. Loading and metric tests
are CPU-only; full embedding/model benchmark runs should use the configured
Kaggle GPU path rather than the laptop.

Run a two-question CPU-only retrieval smoke without loading a model:

```powershell
python scripts/run_qasper.py `
  --dataset data/evaluations/external/qasper/qasper-dev-v0.3.json `
  --split dev --retrieval-mode lexical --generator none --limit 2 `
  --output-dir data/evaluations/runs/qasper-smoke
```

`--generator none` always predicts `Unanswerable`; its answer F1 is therefore
not a model score. Dense/hybrid retrieval and Transformers generation are
optional heavy modes intended for Kaggle Control Plane. Test data additionally
requires `--allow-test`, so a development command cannot accidentally consume
the held-out split. Transformers prompts are passed to the pipeline together
with configurable `--generation-batch-size` (default 8); CUDA out-of-memory
errors retry at successively smaller batch sizes without moving generation onto
the laptop.

Prepare the narrow, ignored Control Plane source folder with:

```powershell
python scripts/prepare_qasper_kaggle_job.py
```

The generated folder contains one embedded application entrypoint, pinned
Kaggle requirements, a checksum manifest, and private T4 kernel metadata. It
does not package the repository root, credentials, held-out test data, or prior
evaluation runs. The remote entrypoint creates a clean virtual environment
without system packages, verifies a real CUDA matrix operation, and records its
resolved dependency fingerprint before starting the full dev benchmark.

The exact self-contained source used by the completed QASPER R11 development
run is archived under `evaluation/kaggle/qasper_v0_5_r11/`. Across 892
retrieval-eligible cases, hybrid Recall@5 was `0.5237`, compared with `0.4957`
for dense and `0.4605` for lexical retrieval. This supports retaining hybrid
retrieval, but it is not a strong absolute-quality result and the answer scores
are not apples-to-apples because only the hybrid configuration used a model.
Generated predictions and Kaggle runtime artifacts remain uncommitted.

SciFact evaluation remains separate from QASPER and the internal verifier. The
runner receives SciFact's cited abstracts, preserves native
`SUPPORT`/`CONTRADICT`/`NOT_ENOUGH_INFO` labels, and measures three-way label
accuracy plus rationale-sentence selection. It is an oracle-document diagnostic:
it does not measure retrieval or the LangGraph flow.

```powershell
python scripts/run_scifact.py `
  --corpus data/evaluations/external/scifact/data/corpus.jsonl `
  --claims data/evaluations/external/scifact/data/claims_dev.jsonl `
  --source-split dev `
  --output-dir data/evaluations/runs/scifact-v0-5 `
  --smoke-cases 3
```

This command requires CUDA. Use `scripts/prepare_scifact_kaggle_job.py` to build
the narrow ignored Control Plane package. The package downloads and verifies the
pinned public archive at runtime and never embeds external data or credentials.
Task 8's `partial` relationship is intentionally not converted into a SciFact
label; see `evaluation/CLAIM_LABELING_GUIDE.md`.

The completed 300-case dev R3 run reached label accuracy `0.7233`, macro F1
`0.7070`, rationale sentence F1 `0.7438`, and joint label+rationale exact match
`0.5266`. SUPPORT F1 was `0.8016`, NOT_ENOUGH_INFO F1 `0.7090`, and CONTRADICT
F1 `0.6104`. Re-parsing the exact saved outputs with the corrected optional
diagnostic reason left one malformed response. These are external dev
oracle-document results, not held-out, retrieval, or end-to-end scores.

The internal suite's LLM judge checks question clarity, evidence entailment,
answer alignment, citation specificity, and challenge design. It does not edit
the gold data, count as a human reviewer, prove document-wide absence, or make
the development suite publishable. Prepare its isolated T4 package with:

```powershell
python scripts/prepare_internal_judge_kaggle_job.py
```

The generated source remains under ignored runtime storage. The remote job uses
batched deterministic generation with left padding and writes
`judge_report.json`, `runtime.json`, and a resolved dependency fingerprint.
Select the 25-case suite without changing the R10 template identity with:

```powershell
python scripts/prepare_internal_judge_kaggle_job.py `
  --suite-name development_25 `
  --destination data/evaluations/kaggle_jobs/internal_judge_v0_5_r25
```

The judge runtime inherits Kaggle's compatible CUDA/PyTorch installation inside
an isolated `--system-site-packages` environment and pins the Pydantic/core pair
used by the evaluation schema.

R25 judge A2 completed 25 schema-valid calls: all 22 answer cases passed and the
three intentional abstentions were flagged for human review, as designed. The
flags do not invalidate the abstentions because selected evidence cannot prove
document-wide absence. Mean lint scores were `4.88/4.72/4.68/4.80/4.88`; these
remain advisory development signals.

Score ranked retrieval output against the internal gold evidence with:

```powershell
python scripts/evaluate_internal_retrieval.py `
  --suite evaluation/suites/v0_5/development_10.json `
  --retrievals data/evaluations/runs/internal-retrieval/retrieved.jsonl `
  --config-name hybrid-current --top-k 5 `
  --output-dir data/evaluations/runs/internal-retrieval/scored
```

The input has one JSON object per case with `case_id` and an ordered `retrieved`
array. Every chunk must include `versioned_id` and `text`; page, section, chunk
index, and retrieval scores are optional diagnostics. The evaluator reports
annotation-relative Recall@K, Precision@K, MRR, gold-evidence coverage,
required-paper coverage, macro paper recall, and per-case failure reasons. It is
CPU-only and does not invoke an embedding model or LLM.

Run the eight-configuration internal diagnostic with identical PDFs, chunking,
questions, K, and scoring using:

```powershell
python scripts/run_internal_retrieval_ablation.py `
  --suite evaluation/suites/v0_5/development_10.json `
  --sources evaluation/suites/v0_5/development_10_sources.json `
  --papers-dir data/papers `
  --modes lexical dense hybrid hybrid_score hybrid_per_paper `
    hybrid_score_per_paper hybrid_rerank hybrid_rerank_per_paper --top-k 5 `
  --output-dir data/evaluations/runs/internal-retrieval-ablation
```

Dense and hybrid default to a pinned `Qwen/Qwen3-Embedding-0.6B` revision,
matching the repo's configured embedding-model family. Run those arms through
Kaggle Control Plane, not on the laptop.
The two rerank modes score the lexical/dense candidate union with a pinned
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Each long chunk is scored as overlapping
900-character passages and max-pooled back to one chunk score; the per-paper
variant retains the fixed total K while reserving fair paper coverage.
`scripts/prepare_internal_retrieval_kaggle_job.py` builds a narrow T4 bundle
containing only required code; the remote entrypoint downloads the suite's
pinned paper revisions and verifies their SHA-256 before parsing. The Markdown
report explicitly labels results as internal development signals rather than
held-out accuracy.

Both internal packagers accept `--suite-name development_25` and a distinct
`--destination`. The R25 retrieval bundle downloads and verifies all five
source revisions from `development_25_sources.json`; it does not require local
PDFs or overwrite the R10 bundle.

The first complete Kaggle T4 comparison (`R4`, 2026-08-27) scored nine
retrieval-eligible development cases at K=5. Lexical achieved Recall `0.7778`,
Precision `0.2000`, and MRR `0.6667`; dense achieved `0.7222`, `0.2222`, and
`0.6389`; hybrid achieved `0.7222`, `0.2000`, and `0.5000`. Hybrid therefore did
not win this small internal suite. Treat this as a fusion/failure-analysis
signal, not a held-out model claim; the larger QASPER dev run still favored
hybrid Recall@5.

The follow-up R5 diagnostic added two intentionally untuned axes based on
established IR practice: min-max-normalized CombSUM and fixed-K per-paper
balancing. CombSUM reached Recall@5 `0.8333`, Precision@5 `0.2444`, and MRR
`0.6204`; per-paper balancing did not improve coverage. Because CombSUM traded
away the masked-LM hit and the suite is development-only, production remains on
the existing RRF path pending independent validation.

R25 retrieval A1 evaluated 24 eligible cases at K=5. Lexical, dense, RRF, and
CombSUM Recall@5 were `0.8333`, `0.8125`, `0.8125`, and `0.8542`; their MRRs
were `0.6354`, `0.6250`, `0.5951`, and `0.6472`. CombSUM led this diagnostic but
still missed two single-paper gold passages and complete annotated LoRA/RAG
coverage, so production RRF remains unchanged pending failure analysis.

R25 windowed-reranker A2 (`job_941b1b0763c2472e8d71cf78bff8a7b9`) recovered
all three A1 target failures. Global and per-paper reranking reached Recall@5
`0.8958`, Precision@5 `0.2667`, MRR `0.6688`, and required-paper coverage
`0.8958`. This is a development ablation, not a production-default change:
`resnet_identity_shortcut_cost` regressed while three target cases improved, and
the BERT masked-LM and Transformer/BERT comparison misses remain.

Run the controlled Task 6 verifier diagnostic with the production prompt and
response parser using:

```powershell
python scripts/run_verifier_benchmark.py `
  --definition evaluation/suites/v0_5/verifier_development.json `
  --source-suite evaluation/suites/v0_5/development_10.json `
  --output-dir data/evaluations/runs/verifier-v0-5
```

The Transformers runner requires CUDA and is intended for Kaggle Control Plane;
`scripts/prepare_verifier_kaggle_job.py` creates the narrow ignored job source.
The completed R2 development run used the official FP16 `Qwen/Qwen3-4B`
revision on a T4 and reached initial accuracy `0.8636`, false-positive rate
`0.0000`, false-negative rate `0.3000`, rewrite recovery `0.7000`, and final
abstention accuracy `1.0000`. Bounded-flow accuracy was `0.7273`. These 22
controlled snapshots are repo-authored development diagnostics, not held-out
accuracy; the local Ollama model is a quantized runtime of the same family, not
a bit-identical inference target.

Evaluate an explicit claim-to-evidence citation record without a model using:

```powershell
python scripts/evaluate_citations.py `
  --suite evaluation/suites/v0_5/citation_safety_fixtures.json `
  --output-dir data/evaluations/runs/citation-safety-fixtures
```

The included five cases are contract fixtures, so their scores test metric
behavior and must not be reported as model quality. Real answer evaluation will
consume the same contract through the implemented claim-verification graph path.

Task 7 defines that contract in `app/models/claims.py`. Each atomic claim
keeps both a normalized `claim_text` and exact `source_text` from the answer,
its visible numeric citation labels, whether citation is required, one
assessment per cited passage, and a derived `supported`, `partial`,
`unsupported`, or `not_required` verdict. Export the synchronized schema after
changing the contract with:

```powershell
python scripts/export_claim_verification_schema.py `
  --output evaluation/schema/claim-verification.schema.json
```

The committed fixture covers all verdict shapes and connects directly to the
Task 9 citation metrics. Task 8 implements
`app/models/claim_verifier.py`: a bounded Qwen attempt extracts claims and checks
each attached label against verifier-approved evidence, then the Task 7 parser
rejects altered answers, altered evidence counts, invented source text, missing
links, and inconsistent verdicts. The standalone Task 8 diagnostic remains
one-shot with the original contract-shaped model output. Production instead
pre-segments the answer, asks the model for `source_span_id` plus ordered semantic
relationships, and derives claim IDs, exact source text, visible labels, and
verdicts in code. It may retry malformed structure exactly
once without changing the answer or evidence. Task 10 invokes this path after citation-safe
synthesis. Fully supported answers finish; partial or mixed answers receive one
evidence-only revision and are checked again; wholly unsupported, malformed,
uncited, or still-failing answers abstain. The revision count is stored in graph
state and cannot exceed one. A seven-case synthetic development diagnostic exercises
direct support, partial support, wrong/missing citations, organizational text,
compound claims, and mixed citation quality through the production prompt and
parser. It is deliberately not held-out or independently reviewed accuracy.

Run the batch only on a CUDA host (normally through Kaggle Control Plane):

```powershell
python scripts/run_claim_verifier_benchmark.py `
  --suite evaluation/suites/v0_5/claim_verifier_development.json `
  --output-dir data/evaluations/runs/claim-verifier-v0-5 `
  --smoke-cases 1
```

`scripts/prepare_claim_verifier_kaggle_job.py` builds the narrow ignored Kaggle
source from `evaluation/kaggle/claim_verifier_v0_5/`. Runtime reports remain
ignored and must not be committed. The first corrected T4 diagnostic reached
schema validity `0.8571`, exact source/citation extraction `0.7143`, claim
verdict accuracy `0.7500`, and evidence-relationship accuracy `0.8571` over
seven synthetic cases. Treat these as failure-discovery numbers, not model
quality claims.

Run the complete internal suite through production LangGraph with:

```powershell
python -m evaluation.run `
  --suite evaluation/suites/v0_5/development_10.json `
  --config hybrid_verified `
  --output-dir data/evaluations/runs/end-to-end-v0-5
```

`--config` is a stable run label; it does not silently switch production
retrieval or model settings. Outputs are a schema-valid `report.json`, compact
`metrics.json`, one rich `per_case.jsonl` row per case, and readable `report.md`.
Each case includes
ordered node updates, the serializable final graph state, retrieval diagnostics,
final decision, claim status, repair count, counted LLM-node calls, latency, and
failure reasons. Embedding calls are explicitly `null` until instrumented.

Compare a later identical-suite/config run with a prior aggregate using:

```powershell
python -m evaluation.run `
  --suite evaluation/suites/v0_5/development_10.json `
  --config hybrid_verified `
  --baseline data/evaluations/baselines/v0_5/metrics.json `
  --output-dir data/evaluations/runs/end-to-end-v0-5-next
```

Comparison requires the same suite fingerprint, ordered case IDs, dataset
version, and config label. It reports directional deltas but enforces no quality
threshold. Runtime outputs and baselines remain ignored. The full run invokes
the configured production models and indexing path, so execute it through the
Kaggle Control Plane/GPU workflow, not as a heavy laptop job. Build the narrow,
ignored Kaggle source with:

```powershell
python -m scripts.prepare_end_to_end_kaggle_job
```

Select R25 and an explicit retrieval configuration without changing R10:

```powershell
python -m scripts.prepare_end_to_end_kaggle_job `
  --suite-name development_25 `
  --retrieval-mode rrf `
  --config-name r25_production_rrf_qwen3_4b_fp16 `
  --destination data/evaluations/kaggle_jobs/end_to_end_v0_5_r25_rrf
```

For a focused GPU regression, repeat `--case-id` to select only named suite
cases. The smoke gate and measured run use the same filtered set, and the bundle
manifest records the selection:

```powershell
python -m scripts.prepare_end_to_end_kaggle_job `
  --suite-name development_25 `
  --case-id lora_all_resource_reduction_factors_missing `
  --kernel-slug owner/sra-e2e-lora-guard-r14 `
  --title "SRA LoRA coverage guard R14" `
  --destination data/evaluations/kaggle_jobs/sra-e2e-lora-guard-r14
```

`--retrieval-mode windowed_rerank` enables the pinned cross-encoder candidate;
`rrf` remains the production default. Bundle manifests record suite, config,
retrieval mode, model revisions, and exact source hashes.

For an immutable rerun, pass the final Kaggle identity before the manifest is
hashed, for example `--kernel-slug owner/project-r8 --title "Project R8"`.
Editing `kernel-metadata.json` afterward makes its recorded provenance hash stale.
The packager rejects malformed identities and Kaggle kernel slugs or titles over
50 characters before a Control Plane submission can consume a job slot.

The committed template lives in `evaluation/kaggle/end_to_end_v0_5/`. It creates
an isolated system-site-aware runtime without replacing Kaggle Torch, pins the
Qwen3-4B and Qwen3-Embedding revisions, runs one smoke case before the full
suite, and keeps only reports under `/kaggle/working`. With two visible T4s it
uses device 0 for generation and device 1 for embeddings; with one T4 it keeps
generation on device 0 and moves embeddings to CPU instead of oversubscribing
the inference GPU.

The first clean live checkpoint was Kaggle R4 on 2026-08-29. All ten cases and
the report schema completed with no execution or tool errors. Decision accuracy
was `0.4000`, answer-case accuracy `0.5000`, abstention accuracy `0.0000`,
Recall@5 `0.6667`, MRR `0.5556`, and claim-verifier failure rate `0.4000`.
These are development diagnostics, not a saved regression baseline: both
negative questions were incorrectly answered, and four positive answers failed
closed on invalid claim-verifier structure. The apparently perfect supported-
claim and citation-completeness rates apply only to successfully parsed final
claim bundles and must not be read as overall grounding accuracy.

Kaggle R5 (`job_bc2c4caca10e4b36ab3177b7058d2aac`) re-ran the identical
suite after adding semantic metric anchors and one structure-only claim-output
retry. Decision accuracy rose to `0.6000` and both negative cases abstained;
the ImageNet case exercised the anchor guard directly. This is still not a
baseline: all four prior claim-structure failures remained after the retry, and
the final energy verifier call hit CUDA OOM after two clean insufficient-evidence
decisions. R5 therefore identifies the next fixes—deterministic answer-span
binding and bounded verifier context—rather than establishing release quality.

Kaggle R6 (`job_097f59e9ea3b45e383deae420c7c4bd0`) applied the eight-passage
per-paper bound and initial span binding. It removed the CUDA OOM and all tool
errors, but decision accuracy and claim failure remained `0.6000`/`0.4000`
because the model still had to echo code-owned top-level fields. R7
(`job_842ae6a6bd924b62acd7171e47d44009`) removed those fields: decision accuracy
reached `0.9000`, answer-case accuracy `0.8750`, both abstentions remained
correct, claim failure fell to `0.1000`, and mean case latency fell from R6's
132.2 seconds to 65.2 seconds. The sole remaining failure copied an incorrect
citation label inside a multi-paper assessment; production v3 now binds those
labels and derives verdicts in code as well. These remain development runs, not
a frozen or independently reviewed baseline.

R8 (`job_b280142291d440049ead48c9d3a3c20b`) verified that change: structural
claim-verifier failures reached `0.0000`, with zero execution/tool errors and
58.1-second mean case latency. Decision accuracy remained `0.9000` because the
comparison answer placed `[1]` after the second sentence in a shared citation
scope; sentence-level spans treated the first factual sentence as uncited, the
one allowed revision did not change it, and the graph correctly abstained.
Production v4 therefore uses exact citation-scoped spans while still requiring
the model to assess every atomic claim against the scoped evidence.

R9 (`job_2ec379f20ff44e90895c446a294abb4b`) validated v4 on the identical
suite: decision, answer-case, and abstention accuracy were all `1.0000`; claim-
verifier, citation-safety, execution, and tool-error rates were all `0.0000`.
The comparison case verified on its first attempt with no answer revision. Mean
case latency was 51.9 seconds with 32 graph LLM calls. Retrieval did not change:
Recall@5 remained `0.6667` and MRR `0.5556`. R9 is therefore a clean development
checkpoint, not a publishable score or frozen baseline. The eight answer cases
were subsequently source-audited against the checksum-pinned PDFs on 2026-08-30
and retained as suite v0.1.3. Because the cases remain tuned development data,
R10 repeated the R9 configuration under the new exact suite identity. It
completed with identical quality metrics and 32 graph calls;
its aggregate is the first local development regression baseline. It remains
Git-ignored and must not be presented as held-out accuracy.

The next internal suite is committed separately as
`evaluation/suites/v0_5/development_25.json`. Its first ten parsed case objects
are regression-tested against `development_10.json`, and its five source
revisions are checksum-pinned in `development_25_sources.json`. Keeping the two
suite identities separate means future R25 diagnostics cannot overwrite or be
mistaken for the R10 baseline.

The first R25 end-to-end controlled pair completed on 2026-08-31. Production
RRF R12 reached decision accuracy `0.9200`, answer F1 `0.4334`, Recall@5
`0.8333`, MRR `0.6403`, and claim-verifier failure `0.0400`. Windowed reranker
R13 improved Recall@5 to `0.8542` and MRR to `0.6719`, but decision accuracy
fell to `0.8400`, answer F1 to `0.4015`, claim-verifier failure rose to `0.1200`,
and mean latency increased by 5.7 seconds/case. RRF therefore remains the
production default. Both are development diagnostics, not held-out scores.

Focused R14 (`job_79e3fe9372784af88b7aa5ba7ae4e3dc`) ran only the adversarial
LoRA missing-factor case. It abstained correctly with retrieval Recall@5 `1.0`
and made zero synthesis or claim-verifier calls. The clean smoke trace preserved
partial evidence while reporting the missing numerical latency factor on all
three bounded checks. The duplicate measured path also abstained but hit CUDA
OOM on its third verifier call, so this validates the safety boundary rather
than establishing a clean performance baseline; bounding verifier context is
the next runtime fix.

`first_submitted_at` is the first arXiv submission and `last_revised_at` is the
retrieved arXiv version's update time. Neither is a journal publication date.
Versioned IDs such as `1706.03762v7` are preserved in citations, and a new
revision invalidates stale PDF/index metadata.

The Atom API has no last-revision date filter. `--date-mode last_revised` scans
a bounded relevance window and validates dates client-side, so it is useful for
focused queries but is not a complete historical harvest. Use arXiv OAI-PMH for
exhaustive corpus synchronization.

PDF downloads are staged in a temporary file and verified by signature and
PyMuPDF before atomic rename. HTTP/content failures are recorded; the graph can
fall back to abstract-only evidence and labels it as `Abstract` rather than
inventing a page citation. See `docs/METADATA_AND_PDF_POLICY.md`.

## Scope

This is intentionally V1. It does not ingest all of arXiv, fine-tune models, run
a multi-agent swarm, or process figures/tables yet. See `docs/ROADMAP.md`.

Implementation decisions, failures, fixes, and baseline evaluation results are
recorded in `docs/DEVELOPMENT_LOG.md`.

Architecture is documented as one module-only overview plus a detailed diagram
for every module in `docs/SYSTEM_VISUALIZATION.md`. The current implementation,
known issues, verification status, and next priorities are maintained in
`docs/PROJECT_STATE.md`. Future coding sessions follow the update contract in
`AGENTS.md` so these documents stay synchronized with the code.
