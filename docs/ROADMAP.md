# Roadmap

## V1 (current)

- Search and persist arXiv metadata
- Preserve arXiv revisions and distinguish first submission from last revision
- Lazy download and text extraction
- Section-aware chunks and FAISS retrieval
- Hybrid dense/lexical retrieval with reciprocal-rank fusion
- LLM evidence verifier, bounded query rewrite, and fail-closed abstention
- Single/multi-paper grounded answers
- Bounded LangGraph loop and Gradio demo

## Next validation work

1. Curate 50 questions with page-level evidence labels.
2. Measure dense vs lexical vs fused Recall@K on the labeled set.
3. Correct section-boundary metadata around tables and mid-page headings.
4. Benchmark 1.7B/4B/8B using identical evidence and tool traces.
5. Run batch/model benchmarks on Kaggle GPU rather than the 4 GB laptop GPU.
6. Add structured action schemas only after collecting routing failures.

## Deferred

- Figure/table extraction and a separate VLM tool
- QLoRA/distillation
- PostgreSQL/Qdrant and deployment
- Scheduled arXiv discovery
- DOI/Crossref enrichment for journal publication dates and venues
