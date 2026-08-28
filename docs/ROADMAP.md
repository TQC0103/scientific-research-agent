# Roadmap

## V1 (current)

- Search and persist arXiv metadata
- Preserve arXiv revisions and distinguish first submission from last revision
- Lazy download and text extraction
- Section-aware chunks and FAISS retrieval
- Hybrid dense/lexical retrieval with reciprocal-rank fusion
- LLM evidence verifier, bounded query rewrite, and fail-closed abstention
- Per-paper comparison coverage with isolated evidence, rewrites, and retries
- Single/multi-paper grounded answers
- Bounded LangGraph loop and Gradio demo

## Next validation work

The v0.5 evaluation schema, provenance/publication gate, semantic loader,
QASPER/SciFact adapters, deterministic metrics, checksum-pinned downloader,
portable QASPER runner, four non-benchmark fixtures, and completed external
QASPER R11 development checkpoint are preserved. R11 supports retaining hybrid
retrieval but does not replace an independently reviewed internal benchmark.
Continue in this order:

1. Independently review the remaining eight answer cases before freezing a
   benchmark snapshot. The two abstention cases were human-adjudicated on
   2026-08-27; the advisory LLM lint pass does not replace review.
2. Independently validate or expand the new 22-snapshot verifier development
   diagnostic. Task 6's runner is implemented: its first T4 result found zero
   false positives but a 30% false-negative rate on positive comparison scopes,
   70% rewrite recovery, and 67.5% supported-passage precision.
3. Review and expand the implemented seven-case Task 8 synthetic development
   diagnostic into independently checked atomic claims before setting thresholds.
   The production prompt/parser, deterministic structural metrics, CUDA runner,
   and isolated T4 package now exist. Citation safety is fail-closed:
   automatic `[1]` fallback is removed, invalid labels discard synthesis, and
   deterministic precision/completeness/unsupported/invalid metrics exist.
   Atomic claim, source-span, evidence-link, and derived-verdict models plus a
   synchronized JSON Schema exist. One bounded Qwen call performs extraction
   and entailment; development results are diagnostics, not publishable accuracy.
4. Analyze and preserve the implemented SciFact oracle-document runner. It uses
   native three-way labels and rationale sets and remains separate from the
   evidence-sufficiency verifier, retrieval, and Task 8 `partial` relationship.
   The 300-case dev R3 checkpoint is complete; CONTRADICT is the weakest class
   and joint label+rationale exact match is `0.5266`.
5. Validate the implemented Task 10 graph path on reviewed cases. Production now
   performs claim verification after synthesis, permits one repair, and otherwise
   abstains; no live-model quality claim is made from unit tests.
6. Add Task 11 end-to-end reporting and regression comparison without committing runtime
   evaluation outputs.
7. Correct section-boundary metadata around tables and mid-page headings.
8. Benchmark 1.7B/4B/8B on identical evidence and traces, preferring Kaggle GPU
   for batch runs over the 4 GB laptop GPU.

## Deferred

- Figure/table extraction and a separate VLM tool
- QLoRA/distillation
- PostgreSQL/Qdrant and deployment
- Scheduled arXiv discovery
- DOI/Crossref enrichment for journal publication dates and venues
