# System visualization

This document is the architecture source of truth for the implemented system.
The first diagram intentionally shows only module interactions. Each following
diagram opens one module and shows its internal flow. Planned features must not
be drawn as current behavior.

## Module interaction overview

```mermaid
flowchart LR
    UI["Interface module"] --> AGENT["Agent orchestration module"]
    AGENT <--> DISCOVERY["Discovery and catalog module"]
    AGENT <--> INGESTION["PDF ingestion module"]
    AGENT <--> RETRIEVAL["Hybrid retrieval module"]
    AGENT <--> VERIFIER["Evidence verification module"]
    AGENT --> ANSWER["Answer and citation module"]
    ANSWER --> UI

    DISCOVERY <--> STORAGE["Persistence module"]
    INGESTION <--> STORAGE
    RETRIEVAL <--> STORAGE

    RETRIEVAL <--> MODELS["Local model runtime module"]
    VERIFIER <--> MODELS
    ANSWER <--> MODELS

    EVALUATION["Evaluation module"] <--> MODELS
    EVALUATION -->|"end-to-end run"| AGENT
    EVALUATION --> STORAGE
```

## 1. Interface module

Implementation: `app/cli.py`, `ui/gradio_app.py`.

```mermaid
flowchart TD
    USER["User"] --> CLI["Typer CLI"]
    USER --> WEB["Gradio UI"]

    CLI -->|"doctor"| HEALTH["Directory and Ollama checks"]
    CLI -->|"search / local-search"| SEARCH["Discovery API"]
    CLI -->|"index"| INDEX["Ingestion API"]
    CLI -->|"retrieve"| RETRIEVE["Retrieval API"]
    CLI -->|"ask / chat"| GRAPH["Research graph"]
    WEB -->|"research question"| GRAPH

    GRAPH --> RESULT["Answer or insufficient-evidence result"]
    RESULT --> CLI
    RESULT --> WEB
```

## 2. Agent orchestration module

Implementation: `app/agent/graph.py`, `app/agent/state.py`.

```mermaid
flowchart TD
    START(["START"]) --> DISCOVER["discover"]
    DISCOVER --> COVERAGE["Choose any/all coverage and required paper IDs"]
    COVERAGE --> INDEX["index_next required paper"]
    INDEX --> RETRIEVE["retrieve_evidence"]
    RETRIEVE --> CHECK["check_evidence"]

    CHECK -->|"coverage policy satisfied"| SYNTH["synthesize answer"]
    CHECK -->|"paper insufficient + rewrite available"| RETRIEVE
    CHECK -->|"required paper not processed"| INDEX
    CHECK -->|"a required paper exhausts retries"| SYNTH

    SYNTH --> ENOUGH{"Evidence sufficient?"}
    ENOUGH -->|"no"| COVERAGE_GAP["paper-specific coverage gaps"]
    ENOUGH -->|"yes"| CITES{"Citation-safe answer?"}
    CITES -->|"no"| CLAIM_ABSTAIN["claim-grounding abstention"]
    CITES -->|"yes"| CLAIMS["verify_claims"]
    CLAIMS -->|"all factual claims supported"| END(["END"])
    CLAIMS -->|"partial or mixed; revision count = 0"| REVISE["revise once from approved evidence"]
    CLAIMS -->|"unsupported / invalid / already revised"| CLAIM_ABSTAIN
    REVISE -->|"valid citation-safe revision"| CLAIMS
    REVISE -->|"failure"| CLAIM_ABSTAIN
    COVERAGE_GAP --> END
    CLAIM_ABSTAIN --> END

    LIMITS["Limits: 2 query rewrites; 2 auto-indexed papers; 6 tool loops; 1 claim revision"] -.-> CHECK
```

The state carries the original question, coverage mode, required paper IDs,
selected/failed papers, and per-paper maps for retrieval queries, accumulated
chunks, verifier results, and attempt counters. It also preserves approved
evidence, citation validity, the structured claim bundle, claim-verifier calls,
one revision count/history, and fail-closed errors. Aggregate fields remain for
CLI trace and final routing.

## 3. Discovery and catalog module

Implementation: `app/tools/arxiv_search.py`, `app/db/database.py`.

```mermaid
flowchart TD
    INPUT["Question or explicit arXiv IDs"] --> EXPLICIT{"Explicit IDs?"}
    EXPLICIT -->|"yes"| META["Fetch exact arXiv metadata"]
    EXPLICIT -->|"no"| FTS["Search SQLite FTS5 title + abstract"]
    FTS --> COUNT{"At least 3 local candidates?"}
    COUNT -->|"yes"| LOCAL["Use local candidates"]
    COUNT -->|"no"| ARXIV["Query arXiv Atom API"]
    ARXIV --> VALIDATE["Validate ID, version, dates, year, category"]
    VALIDATE --> UPSERT["Upsert metadata into SQLite"]
    UPSERT --> MERGE["Merge local + remote by base arXiv ID"]
    META --> UPSERT
    LOCAL --> OUTPUT["Ranked candidate papers"]
    MERGE --> OUTPUT
    OUTPUT --> INTENT{"Explicit multi-ID or comparison intent?"}
    INTENT -->|"yes"| ALL["coverage=all; require each paper"]
    INTENT -->|"no"| ANY["coverage=any; try up to two papers"]
```

The catalog preserves base ID, versioned ID, version number, first-submitted
time, last-revised time, DOI, journal reference, categories, abstract, and PDF
URL. Journal publication date is not inferred from arXiv dates.

## 4. PDF ingestion module

Implementation: `app/tools/paper_download.py`, `app/ingestion/pdf_parser.py`,
`app/ingestion/chunking.py`, `app/ingestion/indexing.py`.

```mermaid
flowchart TD
    PAPER["Selected paper metadata"] --> CURRENT{"Index identity current?"}
    CURRENT -->|"yes"| READY["Reuse existing index"]
    CURRENT -->|"no"| LAZY["Lazy PDF request"]
    LAZY --> PART["Download versioned .part file"]
    PART --> VALIDATE["Check PDF signature, PyMuPDF open, page count"]
    VALIDATE -->|"invalid / unavailable"| FAILURE["Classify failure"]
    FAILURE --> ABSTRACT["Abstract-only fallback; no fake page"]
    VALIDATE -->|"valid"| ATOMIC["Atomic rename + SHA-256"]
    ATOMIC --> PAGES["Extract text by page"]
    PAGES --> SECTIONS["Detect section headings"]
    SECTIONS --> CHUNKS["Create overlapping page-aware chunks"]
    CHUNKS --> EMBED["Embed chunks"]
    EMBED --> SAVE["Save FAISS, chunks, and index identity"]
    SAVE --> READY
```

An index is reusable only when arXiv revision, exact PDF SHA-256, and embedding
model all match. A revision change invalidates stale PDF/index metadata.

## 5. Hybrid retrieval module

Implementation: `app/retrieval/vector_store.py`.

```mermaid
flowchart TD
    QUERY["Current query for one paper"] --> DENSE_EMBED["Qwen query embedding"]
    DENSE_EMBED --> FAISS["FAISS cosine ranking"]
    QUERY --> TOKENS["Normalize meaningful lexical terms"]
    TOKENS --> LEXICAL["BM25-like lexical ranking over chunks"]
    FAISS --> RRF["Reciprocal-rank fusion"]
    LEXICAL --> RRF
    RRF --> TOP["Top fused chunks per selected paper"]
    TOP --> MERGE["Merge with earlier attempts for the same paper"]
    MERGE --> DEDUPE["Deduplicate; cap at 12 passages per paper"]
    DEDUPE --> EVIDENCE["Paper-isolated evidence map"]
```

`score` remains the dense normalized-dot-product score for inspection;
`retrieval_score` controls fused ordering and is accompanied by dense and
lexical ranks.

## 6. Evidence verification module

Implementation: `app/models/verifier.py`, `app/agent/graph.py`.

```mermaid
flowchart TD
    INPUT["Question + required paper set"] --> LOOP["For each required paper"]
    LOOP --> SCOPE["Paper-scoped question + that paper's passages"]
    SCOPE --> QWEN["Qwen3 4B verifier; temperature 0; fixed seed"]
    QWEN --> JSON["Per-paper JSON: sufficient, reason, missing, query, supported IDs"]
    JSON --> VALIDATE["Parse schema and validate passage IDs"]
    VALIDATE --> CONSISTENCY["Repair one defined boolean/list contradiction"]
    CONSISTENCY --> DIMENSIONS{"Every requested dimension covered for this paper?"}

    DIMENSIONS -->|"yes"| PAPER_OK["Mark this paper covered"]
    DIMENSIONS -->|"no"| RETRIES{"This paper's rewrite budget remains?"}
    RETRIES -->|"yes"| REWRITE["Rewrite query only for this paper"]
    REWRITE --> RETRIEVAL["Run hybrid retrieval for this paper"]
    RETRIEVAL --> SCOPE
    RETRIES -->|"no"| PAPER_GAP["Record paper-specific gap"]

    PAPER_OK --> AGGREGATE{"All required papers covered?"}
    PAPER_GAP --> AGGREGATE
    AGGREGATE -->|"yes"| FILTER["Union approved passages, preserving paper identity"]
    FILTER --> SYNTHESIS["Cross-paper synthesis"]
    AGGREGATE -->|"no"| STOP["Coverage gaps by arXiv ID; skip synthesis"]

    ERROR["Model, JSON, or validation failure"] --> STOP
```

Semantic similarity is treated only as retrieval evidence, never as proof that
a passage answers the question. Negative and exhaustive questions require
enough scope; silence in a few passages is not accepted as proof. For a
multi-paper question, each paper must cover every requested comparison
dimension on its own side. Missing evidence about another paper is ignored
during that paper's check, but missing dimensions within the paper are not.

## 7. Answer and citation module

Implementation: `app/models/llm.py`, `app/models/claims.py`,
`app/models/claim_verifier.py`.

```mermaid
flowchart TD
    VERIFIED["Verifier-approved passages only"] --> PROMPT["Evidence-only answer prompt"]
    PROMPT --> QWEN["Qwen3 4B synthesis"]
    QWEN --> RAW["Concise answer with numeric citation labels"]
    RAW --> PARSE["Resolve every numeric label against supplied passages"]
    PARSE -->|"missing or any invalid label"| CITEFAIL["Discard generated answer; explicit grounding failure"]
    PARSE -->|"all labels valid"| METADATA["Resolve trusted arXiv version, title, page, section"]
    METADATA --> FINAL["Answer + deterministic Sources block"]

    INSUFFICIENT["Verifier says insufficient"] --> ABSTAIN["Reason + missing information"]
    ABSTAIN --> FINAL

    FINAL --> CLAIMPROMPT["One bounded atomic extraction + verification prompt"]
    CLAIMPROMPT --> CLAIMQWEN["Qwen3 4B structured JSON"]
    CLAIMQWEN --> CLAIMGUARD["Lock answer/evidence count + Task 7 validation"]
    CLAIMGUARD --> CLAIMBUNDLE["Ordered claims + evidence links + derived verdicts"]
    CLAIMBUNDLE -->|"all supported"| VERIFIED_FINAL["Return answer"]
    CLAIMBUNDLE -->|"partial or mixed; no prior revision"| REPAIR["One evidence-only revision"]
    REPAIR --> SOURCES["Restore trusted Sources block"]
    SOURCES --> CLAIMPROMPT
    CLAIMBUNDLE -->|"unsupported, invalid, or post-repair failure"| CLAIMFAIL["Explicit claim-grounding abstention"]
```

The model never authors bibliographic metadata. Abstract fallback citations are
labeled `Abstract`; full-text citations use stored page and section metadata.
Code never invents a citation when the model omits one. Missing citations and
labels outside the verifier-approved passage list fail closed before any Sources
block is rendered. The production graph then strips that deterministic block,
checks atomic claims against only approved passages, and permits at most one
repair. The repair model cannot author source metadata; code restores it from
trusted passage records before re-verification. Invalid structured output,
wholly unsupported answers, and unresolved post-repair claims abstain.

## 8. Persistence module

Implementation: `app/db/database.py`, runtime files under `data/`.

```mermaid
flowchart LR
    DISCOVERY["Discovery"] --> SQLITE["SQLite paper metadata"]
    SQLITE <--> FTS["FTS5 title + abstract index"]

    INGESTION["Ingestion"] --> PDFS["Versioned verified PDFs"]
    INGESTION --> INDEXES["Per-paper index directories"]

    INDEXES --> FAISS["index.faiss"]
    INDEXES --> CHUNKS["chunks.json"]
    INDEXES --> META["index_meta.json"]

    META --> IDENTITY["Version + PDF SHA-256 + embedding model"]
    IDENTITY -->|"match"| REUSE["Allow index reuse"]
    IDENTITY -->|"mismatch"| REBUILD["Download/rebuild lifecycle"]
```

All runtime databases, PDFs, indexes, evaluation outputs, credentials, and
handoff memory are ignored by Git.

## 9. Local model runtime module

Implementation: `app/config.py`, `app/models/llm.py`,
`app/retrieval/vector_store.py`.

```mermaid
flowchart TD
    CONFIG["Environment-backed settings"] --> OLLAMA["Ollama service"]
    OLLAMA --> EMBEDDING["qwen3-embedding:0.6b"]
    OLLAMA --> REASONING["qwen3:4b-instruct"]

    EMBEDDING --> INDEX_BUILD["Document vector generation"]
    EMBEDDING --> QUERY_EMBED["Query vector generation"]
    REASONING --> VERIFY["Evidence verification"]
    REASONING --> ANSWER["Evidence-grounded synthesis"]
    REASONING --> CLAIMS["Atomic claim verification + one answer repair"]

    DOCTOR["doctor command"] --> CONFIG
    DOCTOR --> OLLAMA
```

The laptop remains the local smoke-test runtime. Kaggle GPU is the preferred
future execution environment for parallel batch evaluations and model-size
comparisons when its Control Plane connection is available.

## 10. Evaluation data artifacts

Implementation: `evaluation/schema/evaluation-suite.schema.json`,
`evaluation/schema/claim-verification.schema.json`,
`evaluation/suites/v0_5/schema_fixtures.json`,
`evaluation/suites/v0_5/development_10.json`, `app/evaluation/`,
`app/models/claims.py`, `scripts/download_external_benchmarks.py`,
`evaluation/README.md`.

```mermaid
flowchart TD
    SCHEMA["Committed JSON Schema 1.1.0"] --> LOADER["Pydantic loader + semantic cross-reference validation"]
    LOADER --> CASES["Versioned internal evaluation cases"]
    CASES --> PAPERS["Exact arXiv revisions + required-paper set"]
    CASES --> EXPECTED["Answer/abstain decision + atomic criteria"]
    CASES --> GOLD["Gold evidence groups"]
    CASES --> PROVENANCE["Fixture/development/test + provenance + annotation"]
    GOLD --> ANCHOR["Versioned ID + source type + page + exact quote"]
    GOLD --> OPTIONAL["Optional chunk index; never sole identity"]

    RANKED["Ranked retrieval JSONL"] --> MATCHER["Revision-safe normalized quote matcher"]
    GOLD --> MATCHER
    MATCHER --> CASEMETRICS["Per-case group recall / precision / RR + diagnostics"]
    CASEMETRICS --> INTERNALMETRICS["Eligible-case aggregate + paper coverage"]
    INTERNALMETRICS --> INTERNALOUTPUTS["Ignored metrics JSON + per-case JSONL"]
    CASES --> ABLATION["Gold-hidden BM25 / Qwen dense candidate runs"]
    PAPERS --> ABLATION
    ABLATION --> FUSION["RRF or min-max CombSUM"]
    FUSION --> GLOBAL["Global fixed-K ranking"]
    FUSION --> PERPAPER["Per-paper rank reset + fair fixed-K quota"]
    GLOBAL --> RANKED
    PERPAPER --> RANKED
    ABLATION --> ABLATIONREPORT["Ignored JSON summary + Markdown comparison"]
    ABLATION -.-> ABLATIONGPU["Pinned isolated T4 package via Control Plane"]

    CASES --> VDEFINITION["Controlled initial + recovery evidence snapshots"]
    VDEFINITION --> VPROMPT["Production verifier prompt + parser"]
    VPROMPT --> VMODEL["Pinned Qwen3-4B deterministic T4 batches"]
    VMODEL --> VINITIAL["Initial sufficiency + supported passages"]
    VINITIAL -->|"insufficient; once only"| VRECOVERY["Material rewrite + recovery snapshot"]
    VINITIAL --> VMETRICS["FP/FN + selection + flow metrics"]
    VRECOVERY --> VMETRICS
    VMETRICS --> VOUTPUTS["Ignored JSON + Markdown verifier report"]

    CLAIMSCHEMA["Task 7 atomic claim + evidence-link contract"] --> CLAIMFIXTURE["Traceable structural fixture"]
    CLAIMFIXTURE --> CLAIMBRIDGE["Validated labels + entails links to stable evidence IDs"]
    CLAIMBRIDGE --> CITEMETRICS["Precision + completeness + unsupported + invalid rates"]
    CITEMETRICS --> CITEOUTPUTS["Ignored JSON + Markdown citation report"]
    CLAIMSCHEMA --> CLAIMMODEL["Task 8 one-call extraction + verification — implemented"]
    CLAIMMODEL --> CLAIMBUNDLE2["Strict validated claim bundle"]
    CLAIMBUNDLE2 --> CLAIMBRIDGE
    CLAIMMODEL --> CLAIMBENCH["7-case synthetic development benchmark"]
    CLAIMBENCH --> CLAIMMETRICS["Schema + extraction + verdict + relationship metrics"]
    CLAIMMETRICS --> CLAIMGPU["Pinned isolated T4 package via Control Plane"]
    CLAIMMETRICS --> CLAIMOUTPUTS["Ignored JSON + Markdown report"]
    CLAIMMODEL --> CLAIMGRAPH["Task 10 bounded verify / repair / abstain graph"]
    CASES --> E2E["Task 11 end-to-end graph runner"]
    CLAIMGRAPH --> E2E
    E2E --> NODETRACE["Ordered LangGraph node updates + final state"]
    NODETRACE --> E2EMETRICS["Registered decision / retrieval / claim / latency metrics"]
    E2EMETRICS --> E2EOUTPUTS["Ignored full/aggregate JSON + per-case JSONL + Markdown"]
    E2EBASELINE["Prior exact-suite metrics.json"] --> E2ECOMPARE["Directional deltas; no threshold"]
    E2EMETRICS --> E2ECOMPARE
    E2ECOMPARE --> E2EOUTPUTS

    CASES --> JUDGEPROMPT["Case-local advisory judge prompt"]
    JUDGEPROMPT --> JUDGEMODEL["Batched deterministic Qwen on isolated T4"]
    JUDGEMODEL --> JUDGEJSON["Validated verdict + five scores + findings"]
    JUDGEJSON --> HUMAN["Human review still required; publication state unchanged"]

    DOWNLOAD["Checksum-pinned public downloader"] --> QASPER["Native QASPER v0.3 adapter"]
    DOWNLOAD --> SCIFACT["Native SciFact adapter"]
    QASPER --> RUNNER["Portable lexical / dense / hybrid runner"]
    RUNNER --> GUARD["Gold-hidden prediction + explicit test access gate"]
    GUARD --> GENERATOR["No-model smoke or Transformers generation"]
    GENERATOR --> BATCH["Batched pipeline inference + bounded OOM batch fallback"]
    BATCH --> METRICS["Official-style answer/evidence F1 + Recall@K/MRR"]
    SCIFACT --> SDOCS["Supplied cited abstracts — oracle-document mode"]
    SDOCS --> SMODEL["Pinned Qwen3-4B native 3-way classifier"]
    SMODEL --> METRICS2["Label macro F1 + rationale + joint metrics"]
    METRICS2 --> SPACKAGE["Isolated T4 package + ignored report"]
    DOWNLOAD --> RUNTIME["Ignored data/evaluations/external/"]

    RUNNER --> OUTPUTS["Ignored predictions JSONL + metrics JSON"]
    RUNNER --> PACKAGE["Narrow dev-only Kaggle source package + checksums"]
    PACKAGE --> VENV["Clean Kaggle venv + pinned direct requirements"]
    VENV --> PREFLIGHT["T4 identity + CUDA matmul + dependency fingerprint"]
    PREFLIGHT -.-> KAGGLE["Heavy dense/model batch via Kaggle Control Plane"]
    PREFLIGHT --> OUTPUTS
    R11["Archived QASPER R11 source snapshot"] --> R11ENV["Embedded app + pinned isolated T4 runtime"]
    R11ENV --> R11RUN["Lexical / dense / hybrid + generation"]
    R11RUN --> R11RESULT["Recorded dev metrics; runtime outputs not committed"]
    SPACKAGE --> OUTPUTS
```

Committed suites are source artifacts, while generated model responses, metric
reports, and baselines are runtime artifacts and remain outside version control.
The LLM judge is an annotation-lint branch: it reads committed development cases
and writes a separate ignored report, but has no edge back into annotation or
freeze state. Abstention cases always retain a human-review flag because selected
context cannot prove document-wide absence.
Negative cases have no retrieval denominator; their retrieval metrics are not
applicable and they are evaluated later through verifier/abstention behavior.
Internal retrieval scoring is isolated from retriever execution. Matching first
requires the pinned `versioned_id`, then normalized quote containment or the
documented token-recall threshold; page equality alone never creates a match.
Evidence-group, item, and required-paper coverage remain distinct so one passage
cannot hide a missing paper in a comparison case.
The internal ablation branch creates one checksum-validated chunk corpus and
holds it fixed across BM25 and pinned Qwen3 dense retrieval. It compares the
production-equivalent global RRF path with min-max CombSUM and diagnostic
per-paper rank/quota variants, always under one total K. Per-paper diagnostics
approximate the production graph's paper isolation but do not reproduce its
sequential verifier loop. They remain evaluation branches and cannot silently
change the production retriever.
The controlled verifier branch resolves committed evidence IDs to exact gold
quotes and uses the production prompt/parser without invoking retrieval or
synthesis. Its recovery edge is bounded to one execution, and initial decision,
passage-selection, recovery, abstention, parsing, latency, and call-count metrics
remain separate. The suite and outputs are development artifacts and do not
cross the publication gate.
The Task 7 claim contract preserves the exact answer substring behind each
normalized atomic claim, visible citations, evidence relationships, and a
derived verdict. Task 8 supplies the one-call extraction and verification
implementation used by Task 10's bounded production graph. Its controlled
development branch sends the same verification prompts through one pinned model,
parses every response with the production validator, and separately reports
schema validity, exact extraction, claim verdicts, evidence relationships, and
citation diagnostics. The seven synthetic cases are not a publication set.
Validated bundles adapt to the citation-safety branch through stable evidence IDs. Unknown predicted IDs are
measured, unknown gold support IDs are rejected, and empty metric denominators
remain null. The committed inputs are fixtures for contract validation rather
than model quality results.
External datasets remain in their native format, preventing repo-authored schema
adaptation from silently changing official answer, evidence, or claim labels.
The no-model mode is a retrieval smoke only and is never presented as an answer
model result. Dense encoders are cached per active paper. Generator prompts are
submitted to Transformers together and internally batched; a CUDA OOM halves
the batch down to one instead of falling back to local compute. Test-set access
is an explicit CLI decision rather than the default development path. The R11
directory is a historical, immutable source snapshot; future evaluator changes
remain in `app/evaluation/` and the packaging script rather than being made in
the archived runner.

Task 11 end-to-end reporting is a versioned observer of the compiled production
graph. The production adapter records each LangGraph node update and reconstructs
the final state; per-case traces therefore retain new nodes and fields. Metric
meaning remains explicit through a direction registry. Exact suite fingerprint,
ordered cases, dataset version, and config must match before baseline comparison.
Comparison is informational and has no hard-coded pass threshold. Outputs remain
ignored JSONL/JSON/Markdown runtime artifacts. Embedding-call instrumentation and
the first live development baseline remain future work.

## Maintenance rule

Every architecture-changing commit must update the overview if module-level
edges changed and the matching detailed diagram if internal flow changed. The
current implementation status and next priorities live in `PROJECT_STATE.md`;
decisions, failures, and evaluation evidence remain chronological in
`DEVELOPMENT_LOG.md`. `AGENTS.md` enforces this checklist for future sessions.
