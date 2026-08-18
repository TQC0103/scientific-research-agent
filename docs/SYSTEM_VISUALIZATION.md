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
    AGENT <--> PLANNER["Query planning module"]
    AGENT <--> INGESTION["PDF ingestion module"]
    AGENT <--> RETRIEVAL["Hybrid retrieval module"]
    AGENT <--> VERIFIER["Evidence verification module"]
    AGENT --> ANSWER["Answer and citation module"]
    ANSWER --> UI

    DISCOVERY <--> STORAGE["Persistence module"]
    INGESTION <--> STORAGE
    RETRIEVAL <--> STORAGE

    PLANNER <--> MODELS["Local model runtime module"]
    RETRIEVAL <--> MODELS
    VERIFIER <--> MODELS
    ANSWER <--> MODELS
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
    START(["START"]) --> DISCOVER["discover candidates"]
    DISCOVER --> EXPLICIT{"Explicit paper IDs?"}
    EXPLICIT -->|"yes"| HARD["Hard coverage=all; bypass planner"]
    EXPLICIT -->|"no"| PLAN["Structured semantic query planner"]
    PLAN --> VALIDATE["Allow-list candidate IDs; enforce policy consistency + budget"]
    PLAN -->|"model / JSON / schema failure"| FALLBACK["Conservative coverage=all fallback"]
    HARD --> INDEX["index_next required paper"]
    VALIDATE --> INDEX
    FALLBACK --> INDEX
    INDEX --> RETRIEVE["retrieve_evidence"]
    RETRIEVE --> CHECK["check_evidence"]

    CHECK -->|"coverage policy satisfied"| SYNTH["synthesize answer"]
    CHECK -->|"paper insufficient + rewrite available"| RETRIEVE
    CHECK -->|"required paper not processed"| INDEX
    CHECK -->|"a required paper exhausts retries"| ABSTAIN["paper-specific coverage gaps"]

    SYNTH --> END(["END"])
    ABSTAIN --> END

    LIMITS["Hard limits: 2 query rewrites; 2 auto-indexed papers; 6 tool loops"] -.-> VALIDATE
    LIMITS -.-> CHECK
```

The state carries the original question, structured query plan, planner status/error/warnings,
coverage mode, required paper IDs, selected/failed papers, and per-paper maps for retrieval
queries, accumulated chunks, verifier results, and attempt counters. Semantic intent is
probabilistic; candidate allow-lists, budgets, coverage invariants, retries, and routing remain
deterministic harness responsibilities.

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
```

The catalog preserves base ID, versioned ID, version number, first-submitted
time, last-revised time, DOI, journal reference, categories, abstract, and PDF
URL. Journal publication date is not inferred from arXiv dates.

## 4. Query planning module

Implementation: `app/models/planner.py`, `app/agent/graph.py`.

```mermaid
flowchart TD
    INPUT["Question + discovered candidate metadata"] --> PROMPT["Planner prompt; metadata treated as untrusted data"]
    PROMPT --> QWEN["Qwen3 4B planner; temperature 0"]
    QWEN --> JSON["Structured intent, coverage, IDs, count, dimensions, reason"]
    QWEN -->|"invocation failure"| ERROR["Planner failure"]
    JSON --> PYDANTIC["Pydantic schema validation"]
    JSON -->|"invalid JSON / schema"| ERROR
    PYDANTIC --> ALLOW["Reject IDs outside trusted candidate set"]
    ALLOW --> CONSISTENCY["Multi-source intent => coverage=all and >=2 sides"]
    CONSISTENCY --> BUDGET["Cap execution set; preserve logical coverage requirement"]
    BUDGET --> PLAN["Validated query_plan in AgentState"]

    ERROR --> FALLBACK["coverage=all over trusted candidates within budget"]
    FALLBACK --> PLAN
```

Explicit user-supplied paper IDs bypass this model completely and remain hard constraints.
Planner outputs are proposals, not authority: Python owns the candidate allow-list, automatic
indexing budget, cross-field consistency rules, and conservative failure policy. If the logical
requirement exceeds the automatic execution budget, the harness preserves that larger requirement
so aggregate coverage fails closed instead of pretending the truncated execution set is complete.
The planner's extracted dimensions are retained for tracing and future evaluation; the evidence verifier still
checks the original user question so an incorrect planner dimension cannot silently strengthen it.

## 5. PDF ingestion module

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

## 6. Hybrid retrieval module

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

## 7. Evidence verification module

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

## 8. Answer and citation module

Implementation: `app/models/llm.py`.

```mermaid
flowchart TD
    VERIFIED["Verifier-approved passages only"] --> PROMPT["Evidence-only answer prompt"]
    PROMPT --> QWEN["Qwen3 4B synthesis"]
    QWEN --> RAW["Concise answer with numeric citation labels"]
    RAW --> PARSE["Accept only labels that map to supplied passages"]
    PARSE --> METADATA["Resolve trusted arXiv version, title, page, section"]
    METADATA --> FINAL["Answer + deterministic Sources block"]

    INSUFFICIENT["Verifier says insufficient"] --> ABSTAIN["Reason + missing information"]
    ABSTAIN --> FINAL
```

The model never authors bibliographic metadata. Abstract fallback citations are
labeled `Abstract`; full-text citations use stored page and section metadata.

## 9. Persistence module

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

## 10. Local model runtime module

Implementation: `app/config.py`, `app/models/llm.py`, `app/models/planner.py`,
`app/retrieval/vector_store.py`.

```mermaid
flowchart TD
    CONFIG["Environment-backed settings"] --> OLLAMA["Ollama service"]
    OLLAMA --> EMBEDDING["qwen3-embedding:0.6b"]
    OLLAMA --> REASONING["qwen3:4b-instruct"]

    EMBEDDING --> INDEX_BUILD["Document vector generation"]
    EMBEDDING --> QUERY_EMBED["Query vector generation"]
    REASONING --> PLAN["Semantic query planning"]
    REASONING --> VERIFY["Evidence verification"]
    REASONING --> ANSWER["Evidence-grounded synthesis"]

    DOCTOR["doctor command"] --> CONFIG
    DOCTOR --> OLLAMA
```

The laptop remains the local smoke-test runtime. Kaggle GPU is the preferred
future execution environment for parallel batch evaluations and model-size
comparisons when its Control Plane connection is available.

## Maintenance rule

Every architecture-changing commit must update the overview if module-level
edges changed and the matching detailed diagram if internal flow changed. The
current implementation status and next priorities live in `PROJECT_STATE.md`;
decisions, failures, and evaluation evidence remain chronological in
`DEVELOPMENT_LOG.md`. `AGENTS.md` enforces this checklist for future sessions.
