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
