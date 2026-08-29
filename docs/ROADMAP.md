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
- Citation-safe synthesis followed by atomic claim verification
- At most one evidence-only answer repair, re-verification, and fail-closed
  abstention
- Versioned end-to-end graph runner with automatic node/final-state traces,
  registered metrics, failure isolation, and exact-suite baseline comparison
- Bounded LangGraph loop and Gradio demo

## Next validation work

The v0.5 evaluation schema, provenance/publication gate, semantic loader,
QASPER/SciFact adapters, deterministic metrics, checksum-pinned downloader,
portable QASPER runner, four non-benchmark fixtures, and completed external
QASPER R11 development checkpoint are preserved. R11 supports retaining hybrid
retrieval but does not replace an independently reviewed internal benchmark.
Continue in this order:

1. Use the completed Task 11 R5 trace to replace free-form claim source-span and
   citation-field copying with deterministic binding to answer spans. The one
   structure-only model retry recovered none of R4's four malformed bundles.
   Also bound verifier context so a third rewrite cannot exhaust T4 memory.
   R5 reached decision accuracy `0.6000` and abstention accuracy `1.0000`, but
   retained claim failure rate `0.4000` and one tool-error-assisted abstention;
   do not freeze a baseline or tag `v0.5.0` from this run.
2. Add embedding-call instrumentation and abstention-reason classification only
   when production exposes reliable observations; do not estimate them.
3. Independently review the remaining eight answer cases before freezing a
   benchmark snapshot. The two abstention cases were human-adjudicated on
   2026-08-27; the advisory LLM lint pass does not replace review.
4. Independently validate or expand the new 22-snapshot verifier development
   diagnostic. Task 6's runner is implemented: its first T4 result found zero
   false positives but a 30% false-negative rate on positive comparison scopes,
   70% rewrite recovery, and 67.5% supported-passage precision.
5. Review and expand the implemented seven-case Task 8 synthetic development
   diagnostic into independently checked atomic claims before setting thresholds.
   The production prompt/parser, deterministic structural metrics, CUDA runner,
   and isolated T4 package now exist. Citation safety is fail-closed:
   automatic `[1]` fallback is removed, invalid labels discard synthesis, and
   deterministic precision/completeness/unsupported/invalid metrics exist.
   Atomic claim, source-span, evidence-link, and derived-verdict models plus a
   synchronized JSON Schema exist. One bounded Qwen call performs extraction
   and entailment; development results are diagnostics, not publishable accuracy.
6. Analyze and preserve the implemented SciFact oracle-document runner. It uses
   native three-way labels and rationale sets and remains separate from the
   evidence-sufficiency verifier, retrieval, and Task 8 `partial` relationship.
   The 300-case dev R3 checkpoint is complete; CONTRADICT is the weakest class
   and joint label+rationale exact match is `0.5266`.
7. Validate the implemented Task 10 graph path on reviewed cases. Production now
   performs claim verification after synthesis, permits one repair, and otherwise
   abstains; no live-model quality claim is made from unit tests.
8. Correct section-boundary metadata around tables and mid-page headings.
9. Benchmark 1.7B/4B/8B on identical evidence and traces, preferring Kaggle GPU
   for batch runs over the 4 GB laptop GPU.

## Deferred

- Figure/table extraction and a separate VLM tool
- QLoRA/distillation
- PostgreSQL/Qdrant and deployment
- Scheduled arXiv discovery
- DOI/Crossref enrichment for journal publication dates and venues
