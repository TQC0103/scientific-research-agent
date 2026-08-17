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
```
