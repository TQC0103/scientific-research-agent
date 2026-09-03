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

1. Preserve R10's reviewed v0.1.3 identity as the first development regression
   baseline. It repeated R9's quality metrics exactly, but remains tuned data;
   comparisons are directional and still have no hard-coded pass threshold.
2. The 25-case development expansion is now authored and source-audited: it
   preserves the ten R10 cases unchanged and adds ResNet v1, LoRA v2, RAG v4,
   one new partial-evidence abstention, and a LoRA/RAG cross-paper comparison.
   Its advisory judge completed with all 22 answer cases passing; the three
   intentional abstentions remain human-retained. Retrieval-only A1 found
   CombSUM ahead of RRF at Recall@5 `0.8542` versus `0.8125`. Windowed
   cross-encoder reranking A2 recovered all three A1 target failures and reached
   `0.8958`, with one different ResNet regression. Preserve both runs as
   diagnostics, not thresholds or a production-default selection.
3. The paired 25-case production-graph run is complete. RRF reached decision
   accuracy `0.9200`; reranking improved Recall@5 from `0.8333` to `0.8542` but
   reduced decision accuracy to `0.8400` and tripled claim-verifier failures.
   Retain RRF. The shared LoRA missing-factor false positive is now protected by
   a fail-closed verifier completeness invariant and synthesis-side regression;
   a separate six-passage verifier cap now bounds rewrite prompt growth. Focused
   R15 validated the memory bound and exposed a metric/scope substitution;
   focused R16 added and validated a deterministic numerical-latency anchor.
   R17/R19 then isolated and resolved the reranker's three observed malformed
   claim outputs with a flat model contract plus a strictly one-span/one-label
   normalizer. Full R20/R21 isolated chained versus true-peak CUDA OOMs. A
   five-passage verifier prefix plus unconditional per-call cleanup then passed
   focused R22 and full R23 with zero OOM/tool/execution errors. R23 remains a
   development runtime checkpoint: decision accuracy was `0.8400`, and the next
   quality work is the ResNet numeric/absence false verification plus three LoRA
   claim-grounding/structure abstentions.
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
