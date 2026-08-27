# Agentic Scientific Research Assistant

Local-first V1 for searching arXiv, lazily downloading papers, section-aware PDF
chunking, FAISS retrieval, and citation-grounded answers with LangGraph + Ollama.

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
- standalone structured claim extractor/verifier over verifier-approved passages

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

Synthesis no longer attaches `[1]` when the model omits citations. An answer
without a valid label, or with any label outside the verifier-approved evidence
set, is discarded and replaced by an explicit citation-grounding failure. Valid
labels are still resolved to trusted version, title, page, and section metadata
by code rather than by the model.

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
in `evaluation/README.md`. The initial four cases are schema fixtures, while
`development_10.json` is a repo-authored development regression suite. Neither
is a held-out or publishable benchmark.

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

Run the six-configuration internal diagnostic with identical PDFs, chunking,
questions, K, and scoring using:

```powershell
python scripts/run_internal_retrieval_ablation.py `
  --suite evaluation/suites/v0_5/development_10.json `
  --sources evaluation/suites/v0_5/development_10_sources.json `
  --papers-dir data/papers `
  --modes lexical dense hybrid hybrid_score hybrid_per_paper `
    hybrid_score_per_paper --top-k 5 `
  --output-dir data/evaluations/runs/internal-retrieval-ablation
```

Dense and hybrid default to a pinned `Qwen/Qwen3-Embedding-0.6B` revision,
matching the repo's configured embedding-model family. Run those arms through
Kaggle Control Plane, not on the laptop.
`scripts/prepare_internal_retrieval_kaggle_job.py` builds a narrow T4 bundle
containing only required code; the remote entrypoint downloads the two pinned
paper revisions and verifies their SHA-256 before parsing. The Markdown report
explicitly labels results as internal development signals rather than held-out
accuracy.

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
consume the same contract after claim extraction and verification are added.

Task 7 defines that next boundary in `app/models/claims.py`. Each atomic claim
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
Task 9 citation metrics. Task 8 now implements
`app/models/claim_verifier.py`: one bounded Qwen call extracts claims and checks
each attached label against verifier-approved evidence, then the Task 7 parser
rejects altered answers, altered evidence counts, invented source text, missing
links, and inconsistent verdicts. The function is standalone and is not yet a
LangGraph node; tests use synthetic model responses, so no live-model quality
claim is made.

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
