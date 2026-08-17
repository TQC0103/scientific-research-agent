# Roadmap

## V1 (current)

- Search and persist arXiv metadata
- Lazy download and text extraction
- Section-aware chunks and FAISS retrieval
- Single/multi-paper grounded answers
- Bounded LangGraph loop and Gradio demo

## Next validation work

1. Curate 50 questions with page-level evidence labels.
2. Measure Recall@K before changing prompts or models.
3. Add BM25 + reciprocal-rank fusion if dense retrieval misses exact terminology.
4. Benchmark 1.7B/4B/8B using identical evidence and tool traces.
5. Add structured action schemas only after collecting routing failures.

## Deferred

- Figure/table extraction and a separate VLM tool
- QLoRA/distillation
- PostgreSQL/Qdrant and deployment
- Scheduled arXiv discovery

