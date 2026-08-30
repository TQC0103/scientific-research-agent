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

1. Independently review the remaining eight answer cases before freezing a
   benchmark snapshot. The two abstention cases were human-adjudicated on
   2026-08-27; the advisory LLM lint pass does not replace review. R9 is the
   first technically clean full-graph checkpoint (`1.0000` decision accuracy,
   zero claim/citation/execution/tool failures), but it remains tuned development
   data and is not yet a baseline.
2. After that review, freeze the exact R9 suite/config identity as the first
   regression candidate and run at least one repeat to observe variance before
   setting thresholds or tagging `v0.5.0`. Do not hard-code thresholds from one
   perfect ten-case run.
3. Expand the reviewed suite from 10 to 25 cases with new papers and failure
   modes before moving to 50. Keep the existing ten cases as a named development
   slice rather than silently changing their fingerprint.
4. Add production embedding-call instrumentation and structured abstention-
   reason classification only where the graph exposes reliable observations;
   do not estimate them from adapter behavior.
5. Independently validate or expand the 22-snapshot verifier development
   diagnostic. Task 6's runner is implemented: its first T4 result found zero
   false positives but a 30% false-negative rate on positive comparison scopes,
   70% rewrite recovery, and 67.5% supported-passage precision.
6. Review and expand the implemented seven-case Task 8 synthetic development
   diagnostic into independently checked atomic claims before setting thresholds.
   The production prompt/parser, deterministic structural metrics, CUDA runner,
   and isolated T4 package now exist. Citation safety is fail-closed:
   automatic `[1]` fallback is removed, invalid labels discard synthesis, and
   deterministic precision/completeness/unsupported/invalid metrics exist.
   Atomic claim, source-span, evidence-link, and derived-verdict models plus a
   synchronized JSON Schema exist. One bounded Qwen call performs extraction
   and entailment; development results are diagnostics, not publishable accuracy.
7. Analyze and preserve the implemented SciFact oracle-document runner. It uses
   native three-way labels and rationale sets and remains separate from the
   evidence-sufficiency verifier, retrieval, and Task 8 `partial` relationship.
   The 300-case dev R3 checkpoint is complete; CONTRADICT is the weakest class
   and joint label+rationale exact match is `0.5266`.
8. Correct section-boundary metadata around tables and mid-page headings.
9. Benchmark 1.7B/4B/8B on identical reviewed evidence and traces, preferring Kaggle GPU
   for batch runs over the 4 GB laptop GPU.

## Deferred

- Figure/table extraction and a separate VLM tool
- QLoRA/distillation
- PostgreSQL/Qdrant and deployment
- Scheduled arXiv discovery
- DOI/Crossref enrichment for journal publication dates and venues
