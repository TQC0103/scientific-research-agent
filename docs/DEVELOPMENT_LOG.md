# Development log

This log records what was built, why decisions were made, failures discovered
during real runs, and the evidence used to choose the next step.

## 2026-08-18 — Project bootstrap

### Starting goal

Build a local-first scientific research assistant using LangGraph and an
open-weight reasoning model no larger than 8B. The V1 scope is arXiv discovery,
lazy full-text ingestion, single/multi-paper retrieval, grounded answers, and
verified citations. Multimodal processing, fine-tuning, and production-scale
infrastructure were deliberately deferred.

### Machine and model decision

The development machine has Python 3.11, 16 GB RAM, an RTX 3050 Ti Laptop GPU
with 4 GB VRAM, and sufficient local storage. Ollama was installed with:

- `qwen3:4b-instruct` for reasoning and synthesis;
- `qwen3-embedding:0.6b` for document/query vectors.

The 4B model was selected instead of 8B to fit the laptop more comfortably.
Python dependencies are isolated with `uv` and pinned in `uv.lock`; the project
does not require a cloud LLM API.

### Initial V1 implementation

The first vertical slice implemented:

1. arXiv keyword search with year/category filters;
2. SQLite metadata persistence and FTS5 title/abstract search;
3. lazy PDF download;
4. PyMuPDF page extraction and section-aware chunking;
5. Qwen embeddings and a per-paper FAISS index;
6. evidence retrieval and Qwen answer synthesis;
7. deterministic citation formatting;
8. a bounded LangGraph discovery/index/retrieve/check/synthesize loop;
9. CLI commands and a minimal Gradio UI.

Runtime PDF, database, index, model, virtualenv, and `.env` files are excluded
from Git. The public repository is
`https://github.com/TQC0103/scientific-research-agent`.

### Bugs found during initial testing

- arXiv sometimes returned results outside the requested year for a compound
  query. A client-side year/category validation step was added.
- `Methods` was not recognized as a section heading. Section normalization was
  corrected.
- The model wrote plausible but altered section names in its Sources list.
  Bibliography generation was removed from the model; code now maps citation
  labels to trusted chunk metadata.
- The initial citation output lost arXiv revisions such as `v7`. Stable base IDs
  and exact versioned IDs are now stored separately.

### First end-to-end paper

`Attention Is All You Need` (`1706.03762v7`) was used for the first real run:

```text
metadata → PDF → 34 chunks → Qwen embeddings → FAISS → answer
```

Retrieval found the multi-head-attention evidence on page 5 and the local model
produced a cited answer. This validated the complete vertical slice.

## 2026-08-18 — Metadata, revision, and PDF hardening

The meaning of arXiv dates was clarified:

- `first_submitted_at` is Atom `published` (version 1 submission);
- `last_revised_at` is Atom `updated` for the retrieved revision;
- neither is a journal/conference publication date;
- `journal_published_at` remains null until an external DOI/publisher source is
  introduced.

The database now stores `arxiv_id`, `versioned_id`, and `version`. A revision
change invalidates PDF/index artifact metadata. Each index records the arXiv
revision, PDF SHA-256, embedding model, and build timestamp. Retrieval only
trusts an index when all three identities match: revision, exact PDF bytes, and
embedding model.

PDF downloads now use a temporary `.part` file, verify the `%PDF-` signature,
open the file with PyMuPDF, require at least one page, and atomically rename the
verified file. Failures are classified as `unavailable`, `download_failed`, or
`no_text_layer`. LangGraph records the failure and falls back to abstract-only
evidence without inventing a page number.

SQLite FTS5 was connected to LangGraph. Discovery searches meaningful local
terms first; if the local catalog is insufficient it calls arXiv and merges
results. Stopwords and strict `AND` matching prevent common question words from
incorrectly making the local catalog appear sufficient.

## 2026-08-18 — Baseline evaluation before new features

No feature was added during evaluation. Six questions were asked against
`1706.03762v7`. Qwen3 4B produced answers; a separate Codex LLM judge compared
the answers with the retrieved passages.

| Metric | Baseline |
|---|---:|
| Answer correctness | 91.7% |
| Faithfulness | 75.0% |
| Citation accuracy | 41.7% |
| Strict retrieval pass | 50.0% |
| Correct stopping/abstention | 50.0% |
| Combined rubric | 69.4% |

Important failure modes:

- A positional-encoding answer was factually correct but unsupported by the
  retrieved chunks; the model used internal knowledge and attached an invalid
  citation.
- BLEU values were retrieved correctly, but the section parser labeled the
  page-8 result table as `Attention` rather than `Results`.
- The three attention applications were answered correctly, but page 3 was
  labeled `Background` instead of `Model Architecture`.
- A negative ImageNet answer was correct but stronger than the retrieved
  coverage justified.
- A limitations query retrieved References/appendix content. The model safely
  abstained, but the current numeric evidence check incorrectly considered the
  chunks sufficient.

### Current conclusion

The synthesis model performs well when retrieval is correct. The main
bottlenecks are retrieval coverage, semantic evidence sufficiency, and section
metadata—not answer fluency. The current evidence rule only counts chunks and
checks the maximum vector similarity; similarity measures topical closeness,
not whether a passage directly answers the question.

### Agreed next direction

Before multi-paper features, add an LLM evidence verifier that reads the
question and retrieved chunks, returns structured support/missing-information
decisions, rewrites the retrieval query when evidence is insufficient, and
prevents synthesis from using unsupported claims. Re-run the same six baseline
cases unchanged after implementation.

## Verification history

The initial implementation passed 4 tests. Metadata/PDF/FTS5 hardening expanded
the suite to 12 passing tests, covering version parsing, migration and cache
invalidation, FTS routing, PDF validation, abstract fallback, chunk metadata,
legacy IDs, and verified citations. Ruff, Ollama doctor checks, Gradio import,
versioned re-indexing, and CLI graph traces also passed.

Key commits:

```text
5007275 Initial V1 scientific research agent
6cc9869 Harden arXiv metadata and PDF lifecycle
c4af443 Document development history and baseline evaluation
```

## 2026-08-18 — LLM evidence verifier and hybrid retrieval

The numeric evidence gate was replaced with a structured local verifier using
`qwen3:4b-instruct`. It receives the question and retrieved passages and returns:

- whether the evidence is sufficient;
- a reason and specific missing information;
- a focused replacement retrieval query;
- the passage numbers that directly support synthesis.

Only passages selected by the verifier reach answer synthesis. If evidence is
still insufficient after at most two query rewrites, the graph returns an
insufficient-evidence response and does not call the synthesis model. Verifier
errors also fail closed. The controlled `ask` command was moved onto the same
graph as `chat`, removing the former bypass around evidence verification.

### Failures found while calibrating

The first prompt was too strict: it rejected all six baseline cases, including
passages that explicitly stated the answer, because it demanded experimental
comparisons or enumerated formatting that the questions did not request. A
semantic sufficiency rubric and general positive/negative calibration examples
fixed that behavior without embedding Transformer-specific answers in code.

A second failure was more dangerous: the 4B verifier incorrectly treated
self-attention and decoder masking as the mechanism that represents token
order. Inspection showed that the actual positional-encoding chunk was dense
rank 11, outside the six passages shown to the verifier. Retrieval was therefore
changed from FAISS-only to hybrid dense + lexical ranking using reciprocal-rank
fusion. The correct chunk became fused rank 5 and the verifier selected page 6.
The BLEU result table likewise appears near the top for exact metric terms.

The model also once returned a contradictory structure: `sufficient=false`, an
empty missing-information list, a supporting passage, and a reason concluding
that the evidence was fully sufficient. Schema post-processing now repairs this
specific logical contradiction. A fixed Ollama seed makes repeated verifier
runs reproducible, and a generic missing-information fallback forces a changed
query when the model merely repeats the current query.

### Post-change six-case result

The unchanged six questions were run end to end against `1706.03762v7`:

| Case | Expected action | Result | Verifier calls |
|---|---|---:|---:|
| Multi-head rationale | Answer | Answered from pages 5/2/4 | 1 |
| Token order | Answer | Answered from positional encoding on page 6 | 2 |
| Base/big EN-DE BLEU | Answer | Answered from Table 2 on page 8 | 1 |
| Three attention uses | Answer | Answered from pages 5/3 | 1 |
| ImageNet negative claim | Abstain | Insufficient evidence | 3 |
| Explicit limitations | Abstain | Insufficient evidence | 3 |

The verifier made the expected answer/abstain decision in 6/6 cases. All four
answered cases used the correct supporting page; the two negative/exhaustive
questions stopped without synthesis. This is a small calibration set, not a
general accuracy claim. The known section-parser errors remain: page 6 and the
page-8 table can still inherit `Attention` instead of their true headings.

Local execution was slow because the 4 GB laptop GPU must hold/swap both the
embedding and 4B reasoning models, and insufficient cases invoke the verifier
three times. Future batch and model-comparison benchmarks should use the user's
Kaggle Control Plane/GPU and parallel execution when that connection is
available; the laptop remains appropriate for unit tests and small smoke runs.

The suite now contains 22 passing tests, including JSON extraction, query
rewrite routing, fail-closed behavior, verified-passage filtering, contradictory
output repair, hybrid lexical scoring, and all earlier metadata/PDF tests.

## 2026-08-18 — Living architecture documentation

Architecture documentation was promoted to a maintained project artifact.
`docs/SYSTEM_VISUALIZATION.md` contains one module-only system diagram and one
detailed internal diagram for each implemented module. `docs/PROJECT_STATE.md`
is the concise current-state handoff, while this development log remains the
chronological record of decisions, failures, and evaluations.

A root `AGENTS.md` now requires future implementation sessions to update the
visualization when interactions change, refresh project state after work, and
record meaningful decisions or evaluations here. `.remember/remember.md` stays
Git-ignored as a short cross-session handoff, not as the architecture source of
truth.

## 2026-08-18 — Per-paper evidence coverage

Multi-paper questions previously pooled all retrieved chunks into one verifier
decision. That allowed evidence from paper A to hide missing coverage in paper
B. Version 0.4 stores retrieval queries, accumulated chunks, retry counts, and
verifier decisions in maps keyed by base arXiv ID. Repeated explicit
`--paper-id` values require every supplied paper. Automatically discovered
comparison questions use a bilingual intent heuristic and require the first two
candidates; ordinary discovery retains `coverage=any` and may try a second paper
only if the first is insufficient.

Each required paper is retrieved and verified independently. A failed check
rewrites and reruns only that paper's query. Synthesis receives the union of
approved passages only after all required paper sides are sufficient. Otherwise
the answer lists coverage gaps by arXiv ID and skips synthesis.

### Real smoke failures and correction

The first real comparison between `1706.03762v7` and `1810.04805v2` exposed a
prompt failure: although evidence was isolated correctly, the 4B verifier still
demanded that each paper contain information about the other paper. A
paper-scoped verification question and calibration example corrected this.

The next run passed both papers, but the generated answer claimed that the
Transformer paper used a "standard language modeling objective" from only
abstract/conclusion evidence. Per-paper coverage was therefore tightened to
require direct evidence for every requested comparison dimension. After the
change, the architecture-and-training-objective case marked BERT sufficient in
one call but stopped after three Transformer calls because direct loss/objective
evidence remained missing. It did not synthesize the unsupported comparison.

A positive self-attention comparison then passed in one verifier call per paper
and produced citations from both arXiv revisions. These are two smoke cases, not
a general multi-paper accuracy benchmark. They demonstrate both successful
cross-paper synthesis and paper-specific fail-closed behavior. The suite grew
from 22 to 28 passing tests, including required-paper discovery, isolated retry,
per-paper supported-passage filtering, full two-paper graph execution, and
routing to the next required paper.

## 2026-08-21 — Versioned evaluation data contract

Before implementing claim verification or changing LangGraph, v0.5 work began
with a shared evaluation data contract. The two empty, unconstrained
`evaluation/questions.json` and `evaluation/ground_truth.json` placeholders were
replaced by JSON Schema Draft 2020-12 and a versioned suite layout.

The schema pins papers to exact arXiv revisions and separates expected
answer/abstention decisions, atomic answer criteria, forbidden claims, required
paper coverage, gold evidence, and challenge labels. Gold evidence uses the
versioned paper ID, source type, page, section, and exact quote as its stable
anchor. Chunk indexes are optional because chunking changes can renumber them.
Equivalent passages can share an evidence-group ID.

The matching contract defines evidence-group Recall@K and MRR, required-paper
coverage for comparisons, and annotation-relative Precision@K. Retrieval metrics
are not applicable for negative cases with no gold evidence, preventing correct
abstention cases from being inserted as retrieval zeroes. Four illustrative
fixtures cover a supported single-paper method question, a required two-paper
comparison, missing evidence, and an adversarial unsupported premise. They are
schema examples and are not reported as benchmark results.

Executable loading, cross-reference validation, metric calculation, retrieval
ablation, and graph changes remain deliberately out of scope for this step.

The Draft 2020-12 schema and all four fixtures validated successfully. Ruff
passed and the unchanged application suite passed all 28 tests. Verification
ran from the repository's new non-OneDrive location at
`C:\Users\ASUS\Documents\GitHub\scientific-research-agent`.

## 2026-08-21 — Public benchmark provenance and adapters

The initial schema was intentionally not treated as evaluation evidence. It was
revised to version 1.1.0 with mandatory fixture/development/test splits,
provenance, annotation metadata, and suite freeze status. An executable
publication gate rejects schema fixtures, unfrozen suites, synthetic test cases,
and repo-curated test cases without independent review and adjudication.

QASPER and SciFact remain native external benchmarks rather than being rewritten
as repo-authored arXiv cases. The QASPER adapter preserves multiple annotator
answers, unanswerable labels, and paragraph evidence; its deterministic answer
and evidence F1 take the best reference in the same manner as the released
evaluator. The SciFact adapter preserves SUPPORT, CONTRADICT, and
NOT_ENOUGH_INFO labels and resolves gold rationale sentence indices against the
official corpus. SciFact is reserved for the future claim verifier because the
current evidence-sufficiency verifier solves a different task.

The official QASPER v0.3 and SciFact archives were downloaded to ignored runtime
storage. A reproducible downloader now pins all three archives by SHA-256 and
performs path-safe extraction. Full CPU-only parsing found the expected 5,049
QASPER questions (2,593 train, 1,005 dev, 1,451 test) and 300 labeled SciFact dev
claims (124 SUPPORT, 64 CONTRADICT, 112 NOT_ENOUGH_INFO). These are dataset
inventory counts, not model scores.

No model or embedding batch was run locally. Kaggle Control Plane was not
available among the tools exposed to this session, so the future external model
benchmark remains pending rather than falling back to the laptop GPU.

## 2026-08-21 — Portable QASPER runner and Kaggle control boundary

A native QASPER runner now separates paper text from QA annotations before
retrieval or generation. It implements dependency-free BM25, optional Sentence
Transformers dense retrieval, reciprocal-rank hybrid fusion, and optional
Transformers answer generation. Dense document embeddings are created once per
active paper rather than once per question. Predictions retain exact paragraph
strings so the released QASPER evidence metric remains meaningful.

The CLI writes per-case JSONL and aggregate answer/evidence F1, retrieval
Recall@K/MRR, denominators, latency, and model-call counts to ignored runtime
storage. Test-set access requires an explicit `--allow-test`; development runs
cannot consume held-out data accidentally. A two-question lexical/no-model smoke
produced retrieval Recall@5 `0.75`, MRR `0.6667`, and evidence F1 `0.3095` in
about 0.01 seconds. Answer F1 was `0.0` by design because no-model mode always
abstains; it is not a model score.

A personal `kaggle-control-plane` plugin was also scaffolded outside the repo.
It exposes bounded MCP tools for status, account/quota inspection, submission,
monitoring, job actions, and artifact download while redacting credentials and
requiring explicit source directories. Its manifest and skill validated, four
offline contract tests passed, and the stdio MCP handshake returned all tools.
The current Codex task cannot load a newly created plugin, and the running
Control Plane desktop API did not answer HTTP requests, so no remote batch was
submitted. No dense encoder, generator model, or GPU benchmark was run locally.
The QASPER dev job source was prepared under ignored runtime storage with a
per-file SHA-256 manifest; it excludes credentials, repository-wide files,
previous runs, and the held-out test split.

## 2026-08-21 — First QASPER dev GPU submission

After restarting Windows, the installed Kaggle Control Plane plugin reported two
enabled accounts with available GPU quota. The first submission was rejected
locally before reaching Kaggle because the prepared source contained both
`main.py` and `run_qasper.py` at top level; no remote quota was consumed. The
packager now places the runner at `app/run_qasper.py`, invokes it as
`python -m app.run_qasper`, excludes Python caches, and has a regression test for
the single-entrypoint contract. The Kaggle bootstrap also installs Pydantic and
Accelerate when missing.

Control Plane's configured source allowlist still points to its dedicated
experiments directory, so a narrow credential-free staging copy was used rather
than broadening the allowlist or moving the repository back. GPU job
`job_c810b8e0e5164f528654f23a3b2a7300` was accepted by Kaggle and reached
`running`, then failed remotely before producing artifacts. Control Plane's log
download contained orchestration events but no Kaggle traceback, and the private
kernel was unavailable in the current browser account, so another retry would be
speculative. No external model score is recorded. Ruff passed and all 43 pytest
tests passed locally; no dense/model workload was run on the laptop.

## 2026-08-21 — Remote failure diagnostics and self-contained Kaggle source

Kaggle Control Plane was upgraded so failed jobs automatically call the Kaggle
output API, redact credential values from downloaded text artifacts, and append
the bounded remote `.log` to the existing log download. The same endpoint can
fetch diagnostics on demand for failures created before the upgrade. Its backend
suite passed all 18 tests with one platform-specific skip, and the rebuilt
desktop app was installed without replacing accounts, encrypted tokens, or job
history.

The recovered r2 traceback showed that Kaggle kernel source uploads did not keep
the packaged QASPER JSON. r3 therefore downloaded the checksum-pinned upstream
train/dev archive and extracted only the dev member, but its new automatic
traceback showed that nested local Python modules were also absent. The r4
packager embeds a path-checked ZIP of the portable runner inside the single
top-level `main.py`, expands it under `/kaggle/working`, and retains the pinned
runtime dataset download. Ruff and all 43 local pytest tests passed. GPU job
`job_c379a019f6b845178e37145736ae7f3f` remained running beyond both earlier
failure points at the latest check; no benchmark score is recorded yet.

## 2026-08-21 — Kaggle P100 runtime compatibility

The upgraded Control Plane successfully downloaded r4's private remote log. It
showed that dataset download, embedded application extraction, imports, and the
lexical run all completed. Dense encoding then failed because Kaggle assigned a
Tesla P100 (`sm_60`) while the preinstalled PyTorch build only contained kernels
for `sm_70` and newer.

The Kaggle bootstrap now probes the assigned device capability against
`torch.cuda.get_arch_list()` before model work. For the exact P100 mismatch it
installs the pinned PyTorch 2.7.1 CUDA 11.8 wheel and verifies `sm_60` support;
unknown incompatibilities fail explicitly instead of silently moving a large
benchmark onto CPU. Ruff passed and all 44 pytest tests passed. No model workload
was run on the laptop. Corrected r5 job
`job_6edee16ce413407eaeb14ea2e4f4ac8f` was accepted by Kaggle and reached
`running`, but failed while validating the newly installed wheel. Its automatic
traceback showed the 905 MB Torch CUDA 11.8 wheel installed successfully, then
the fresh Python import exited unsuccessfully because the install had omitted
its dependency set. The bootstrap now installs matching Torch 2.7.1 and
Torchvision 0.22.1 with dependencies and includes bounded probe stderr in any
future failure.

Corrected r6 job `job_6fa1d28aac674a80957f75c25c71e441` was submitted through
Kaggle Control Plane on account `acct_d321057bf0954d048b448711e0efed7f`
with GPU acceleration. It remained `running` beyond the point where r5's
dependency-free validation had already failed; aggregate metrics and artifacts
remain pending.

## 2026-08-21 — Isolated T4 evaluation runtime

r6 later failed after the full 1,005-case lexical stage. Its downloaded log
confirmed that Torch 2.7.1 CUDA 11.8 and CUDA dependencies installed, but the
Kaggle system environment still exposed TorchCodec/Torchaudio components built
for Torch 2.10 CUDA 12.8. Importing Sentence Transformers therefore loaded an
ABI-incompatible `libtorchcodec` before dense retrieval. The lexical metrics are
retained only as partial diagnostics; no dense/hybrid result exists.

The runtime design no longer mutates Kaggle's system Python or attempts a P100
downgrade. The Control Plane path now requests an explicit Tesla T4, while the
job creates a clean venv without system-site packages, installs pinned direct
requirements, verifies T4 capability with a CUDA matrix operation, and records
the complete resolved freeze plus its SHA-256 before starting evaluation. This
environment boundary prevents leftover Kaggle audio/video packages from
entering the text-only benchmark.

r7 verified that the new Control Plane preserves `NvidiaTeslaT4` in job
metadata and successfully submits the exact CLI accelerator, but Kaggle dropped
the separate top-level requirements text file and the preflight stopped after
two seconds without installing packages or entering the dataset loop. The
packager now embeds the reviewed requirements inside the single accepted Python
entrypoint and materializes them only inside `/kaggle/working`.

r8 materialized the requirements successfully but Kaggle's Python installation
could not complete stdlib `venv` because its `ensurepip` command exits nonzero.
The job again stopped in about three seconds before package installation or
evaluation. The bootstrap now installs pinned `virtualenv` 20.36.1 into a
separate working-directory target, uses its bundled seed wheels to create the
clean environment, and leaves Kaggle's system site-packages unchanged.

r9 `job_ff63a643af4f42ca86baf5f87b58d94a` was accepted with persisted
`NvidiaTeslaT4` metadata and remained running beyond both earlier preflight
failures. Its terminal dense/hybrid artifacts remained pending at that check;
no heavy workload was moved to the laptop.

The live r9 log subsequently confirmed a successful isolated runtime: Python
3.12.13, Tesla T4 capability 7.5, Torch 2.10.0+cu128, CUDA runtime 12.8, and a
CUDA matrix result of 262144.0. The resolved dependency freeze hash is
`acd1f87cac27f70514f090795c1145290cf45bed684a19714ec7559194a619ae`.
All 1,005 lexical dev cases completed before dense retrieval began. Kaggle's
global `sitecustomize` emitted a non-fatal missing-`wrapt` warning, but it did
not prevent the isolated imports, CUDA operation, or lexical run.

The live log later reached `Device set to use cuda:0`, confirming the Qwen
hybrid-answer generator loaded on the T4 after dense retrieval. Transformers
warned that sampling flags were ignored; this is expected because the runner
uses deterministic `do_sample=False`. The terminal hybrid metrics remain pending.

## 2026-08-21 — Batched T4 answer generation

r9 was cancelled locally after Transformers warned that repeated sequential
pipeline calls underused the T4. Control Plane cancellation stops its monitor,
so the record correctly retains `remote_may_be_running=true`; it is not treated
as proof that Kaggle terminated the kernel.

The QASPER runner now retrieves without consulting gold annotations, collects
the resulting prompts, and submits them to the Transformers pipeline together
with batch size 8. Output order remains tied to question IDs, model calls count
questions, and a CUDA out-of-memory error retries the pipeline at 4, 2, then 1.
No CPU or laptop fallback was added. Metrics now also record physical generation
batch calls. A regression test verifies one batched pipeline invocation and
evidence-index filtering.

An initial r10 dispatch was stopped before Kaggle returned a remote job because
the currently loaded MCP cache dropped accelerator metadata. The signed-in
Kaggle page confirmed that kernel URL did not exist. After reconciling that
account, r10 `job_b6bdf13428fa4c31bb4d0bfaefd1d927` was submitted through the
Control Plane backend with explicit `NvidiaTeslaT4` metadata from the narrow
source directory. Ruff passed and all 46 pytest tests passed; no model workload
ran on the laptop.

r10's live generation log later exposed decoder-only right padding during
batched inference. Because right padding can change decoder-only outputs, r10's
answer-generation result is not eligible as a baseline even if the kernel
finishes. The generator now forces tokenizer left padding, reuses the EOS token
for padding when needed, and clears inactive sampling parameters while retaining
deterministic `do_sample=False`. A regression test locks this batch contract.

## 2026-08-21 — External QASPER R11 checkpoint archived

The final self-contained Kaggle R11 source bundle was recovered from the local
Control Plane experiment directory and checked against its source manifest. The
executed `main.py` SHA-256 was
`f6dc6c080374d00894b181344745abac9aeecd29425e8112c12dd0931f5dd157`.
Restoring Windows CRLF line endings reproduces that manifest hash. The committed
kernel metadata replaces the submitted Kaggle owner with `replace-me`, and the
manifest separately records the sanitized LF repository hashes. Downloading a
duplicate runtime artifact from Kaggle was therefore unnecessary.

R11 embeds the exact application snapshot, pins the model environment, verifies
the public QASPER archive by SHA-256, requires Tesla T4 compute capability 7.5,
and runs lexical-only, dense-only, and hybrid-plus-generation configurations.
The completed log contained 1,005 predictions and no missing cases. Retrieval
metrics used 892 eligible QASPER dev cases:

| Configuration | Recall@5 | MRR | Evidence F1 | Answer F1 |
|---|---:|---:|---:|---:|
| Lexical, no generator | 0.4605 | 0.3072 | 0.1606 | 0.1350 |
| Dense, no generator | 0.4957 | 0.3538 | 0.1823 | 0.1350 |
| Hybrid + Qwen2.5-1.5B-Instruct | 0.5237 | 0.3886 | 0.2396 | 0.1651 |

This establishes a reproducible external reason to retain hybrid retrieval, not
a claim that retrieval or answer quality is already strong. Recall@5 still
recovers only about half of annotated evidence. Answer F1 is not a clean
retrieval ablation because the lexical and dense configurations made no model
calls while the hybrid configuration made 1,005. The hybrid generation phase
took roughly 7,332 seconds and the complete Kaggle job roughly 8,013 seconds,
so environment reuse and generation throughput remain optimization targets.

The source bundle is committed under `evaluation/kaggle/qasper_v0_5_r11/`.
Predictions, downloaded data, model caches, resolved environments, and runtime
reports remain uncommitted artifacts.

## 2026-08-21 — Local repository consolidation

Evaluation implementation work had diverged between the canonical checkout
under `C:\Users\ASUS\Documents\GitHub` and a stale OneDrive checkout. The
canonical repository retained the newer schema 1.1.0 loader, provenance gate,
external adapters, runner, packaging scripts, and tests; the OneDrive checkout
retained the final R11 provenance snapshot. The two lines were reconciled by
keeping the newer canonical implementation and importing only the immutable R11
bundle and its verified result record. The OneDrive checkout is removed only
after verification and GitHub publication succeed.

## 2026-08-21 — Ten-case internal development suite and advisory judge

The first internal v0.5 development suite initially contained ten repo-authored
cases over exact Transformer v7 and BERT v2 revisions: seven answer cases and
three abstentions spanning fact, method, result, multi-paper comparison,
unsupported questions, missing evidence, and partial evidence. Exact PDF URLs, SHA-256
hashes, and page counts are recorded beside the suite. The cases remain mutable
development data with zero independent reviewers and cannot pass the publication
gate.

An advisory LLM judge now audits question clarity, quote-to-criterion
entailment, answer alignment, citation specificity, and challenge validity. Its
structured report is stored outside the dataset, and abstention cases always
retain a human-review requirement. The judge therefore catches likely authoring
errors without pretending to prove negative evidence or becoming an independent
annotator.

The GPU package uses Qwen2.5-3B-Instruct on an exact Tesla T4, creates a clean
virtual environment, verifies CUDA capability 7.5, uses left padding and
deterministic batched generation, and saves runtime provenance. Initial submits
from the GitHub checkout were rejected because the desktop app permits sources
only beneath its configured experiments root; the plugin misreported that HTTP
validation error as an offline API. Staging only the three-file narrow package
under the allowed root resolved submission without moving the repository or
exposing credentials. Job `job_ab4973a1512145ae8ba4c6321d025b7a` was submitted
on account `acct_91aab80993c247d4bb787a59c7d43fef` and was later cancelled before
reporting because its embedded suite failed the independent JSON Schema check.

R1 was cancelled after the independent JSON Schema check found two suite/source
mismatches that the semantic loader had accepted: `fact` instead of the enum
value `single_paper_fact`, and an explicit null optional reference answer. The
suite was corrected, both validators now pass, and the loader now uses the same
question-type enum as JSON Schema. R2 `job_d6abb593ac3a4ad9808cf05563e0949a`
ran the schema-valid snapshot on account
`acct_d321057bf0954d048b448711e0efed7f` and reached Kaggle COMPLETE.

R2 also exposed an artifact-boundary defect: placing the isolated environment
under `/kaggle/working` causes Kaggle output collection to download that entire
environment before the small judge report. The template now keeps code,
dependencies, and model cache under `/tmp`; only the report directory crosses
the `/kaggle/working` persistence boundary.

R3 `job_9883b1aa1230436e9458eb9855daf25a` verified the corrected boundary and
succeeded in 292 seconds on Tesla T4 capability 7.5. The resolved environment
hash was `49eeff9c46a9d77656003c79b49f6356e094c8548061d366558ad99abf912a5e`.
The structured report validated all 10 unique cases: seven `pass` and three
`needs_revision`. Every answer case passed. The three flags correspond exactly
to abstention cases where the model cannot prove document-wide absence from
empty or partial gold evidence, so they remain human-review items and no gold
annotation was changed automatically. Mean lint scores were 4.7 clarity, 4.1
evidence entailment, 4.4 answer alignment, 4.4 citation specificity, and 4.7
challenge validity.

R3 also emitted non-fatal Kaggle `sitecustomize`/`wrapt` noise and a Transformers
warning for inactive sampling values inherited from the model's generation
configuration. The runner now clears `temperature`, `top_p`, and `top_k`, and
uses the current `dtype` load argument. R3 remains valid because generation was
explicitly deterministic and no sampling argument was passed to `generate`.

## 2026-08-21 — Source audit, suite v0.1.1, and advisory R4

A full-paper audit found that the original single-head-attention abstention was
invalid: Transformer v7 page 9 explicitly reports that single-head attention is
0.9 BLEU worse than the best head-count setting. The case was corrected into a
numeric answer case with the exact passage and criterion. The suite is now
v0.1.1 with eight answer cases and two abstentions. The energy question remains
a valid partial-evidence abstention: the paper reports 3.5 days on eight P100
GPUs but does not report electrical energy consumption. Both JSON Schema and
semantic validation accept the corrected suite.

R4 `job_655a5faf69f5478cab66078afd223788` waited about 17 minutes for Kaggle T4
capacity, then ran for about 4 minutes 36 seconds and succeeded. The isolated
runtime used Tesla T4 capability 7.5 and resolved environment hash
`49eeff9c46a9d77656003c79b49f6356e094c8548061d366558ad99abf912a5e`.
Only four report/provenance files crossed the output boundary.

The report validated 10 unique cases: eight `pass`, two `needs_revision`, and no
`fail`. Every answer case passed, including the corrected single-head case. The
energy and ImageNet abstentions remain human-review items because the judge
cannot establish document-wide absence from selected evidence. Mean lint scores
were 4.8 clarity, 4.5 entailment, 4.5 answer alignment, 4.7 citation specificity,
and 4.8 challenge validity. R4 also confirmed removal of the inactive-generation
and deprecated load-argument warnings; Kaggle's non-fatal global missing-`wrapt`
warning remains outside the isolated runtime.

## 2026-08-27 — Human adjudication of abstention cases

The repository owner reviewed the two R4 abstention findings and explicitly
retained both cases. The training-energy case remains an `evidence_missing`
abstention because Transformer v7 reports 3.5 days on eight P100 GPUs but no
electrical energy value. The ImageNet case remains an `unsupported_question`
abstention because the paper reports no ImageNet top-1 experiment. Suite v0.1.2
records one human reviewer and adjudication for those two cases only. The eight
answer cases remain unreviewed development annotations, so the suite is not
frozen or publishable.

## 2026-08-27 — Internal retrieval evaluator

The internal evaluation path now scores ranked chunks independently of any one
retriever. A match requires the same pinned `versioned_id` plus normalized quote
containment or at least 0.8 multiset token recall; page equality is diagnostic
only. The report separates evidence-group Recall@K, annotation-relative
Precision@K, MRR, gold-evidence item coverage, required-paper coverage, and
macro paper recall. Cases without gold evidence receive null retrieval metrics
and are excluded from aggregates rather than contributing artificial zeroes.

The JSONL input/output boundary records missing case predictions, unmatched gold
groups, missing required papers, and fewer-than-K results. This keeps scoring
deterministic and CPU-only while allowing the next ablation runner to produce
lexical, dense, and hybrid rankings on Kaggle when a model is required. Ruff
passed and all 65 pytest tests passed; no embedding or LLM benchmark was run on
the laptop.

A one-row contract smoke then exercised the CLI against suite v0.1.2. It found
nine retrieval-eligible cases (the partial-evidence energy abstention is
eligible) and one no-gold ImageNet case, matched the supplied case, and reported
the other nine input rows as missing. The resulting 1/9 aggregate values are a
deliberately incomplete wiring check, not retrieval-quality measurements.

## 2026-08-27 — Internal retrieval ablation runner

The ablation runner now creates one chunk corpus from the two checksum-pinned
PDF revisions and holds chunking, questions, K, matcher, and aggregation fixed
across lexical, dense, and hybrid arms. Lexical uses BM25. Dense uses the pinned
`Qwen/Qwen3-Embedding-0.6B` Hugging Face revision corresponding to the repo's
configured embedding-model family and applies the model's query prompt when
available. Hybrid fuses up to 20 candidates per arm with reciprocal-rank
constant 60. Gold evidence is read only after ranked outputs exist.

Multi-paper cases use one global ranking over all declared-paper chunks. This
keeps K comparable and makes paper-coverage failures measurable, but it is an
evaluation boundary: the production graph retrieves and verifies papers
sequentially. Generated output includes rankings and metrics per arm plus an
aggregate JSON and human-readable Markdown failure report.

A local CPU lexical run over the exact PDFs completed all 10 cases with nine
retrieval-eligible cases and no missing predictions. Recall@5 was `0.7778`,
annotation-relative Precision@5 `0.2000`, and MRR `0.6667`. It missed the BERT
masked-LM mechanism evidence and both groups in the two-paper architecture
comparison. Dense/hybrid were deliberately not run on the laptop.

The Kaggle package embeds only required source modules and the two PDFs after
local checksum validation. It creates a `--system-site-packages` environment,
installs no PyTorch replacement, pins non-system model-library extras, includes
`wrapt`, verifies every visible T4 plus a real CUDA operation, records the
resolved environment, and removes its environment before artifact collection.
Ruff passed and all 71 pytest tests passed before packaging.
