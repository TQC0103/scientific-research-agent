# Roadmap

## V1 (current)

- Search and persist arXiv metadata
- Preserve arXiv revisions and distinguish first submission from last revision
- Lazy download and text extraction
- Section-aware chunks and FAISS retrieval
- Hybrid dense/lexical retrieval with reciprocal-rank fusion
- LLM evidence verifier, bounded query rewrite, and fail-closed abstention
- Structured semantic query planning with deterministic candidate/budget validation
- Per-paper comparison coverage with isolated evidence, rewrites, and retries
- Single/multi-paper grounded answers
- Bounded LangGraph loop and Gradio demo

## Next validation work

1. Add claim-by-claim verification after cross-paper synthesis.
2. Curate planner-intent cases plus 50 questions with page-level evidence labels.
3. Measure dense vs lexical vs fused Recall@K on the labeled set.
4. Correct section-boundary metadata around tables and mid-page headings.
5. Benchmark 1.7B/4B/8B using identical evidence and tool traces.
6. Run batch/model benchmarks on Kaggle GPU rather than the 4 GB laptop GPU.

## Deferred

- Figure/table extraction and a separate VLM tool
- QLoRA/distillation
- PostgreSQL/Qdrant and deployment
- Scheduled arXiv discovery
- DOI/Crossref enrichment for journal publication dates and venues
