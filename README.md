# Agentic Scientific Research Assistant

Local-first V1 for searching arXiv, lazily downloading papers, section-aware PDF
chunking, FAISS retrieval, and citation-grounded answers with LangGraph + Ollama.

## What is ready

- arXiv search with year/category filters and SQLite metadata persistence
- lazy PDF download, PyMuPDF parsing, section-aware chunking
- per-paper Ollama embeddings and FAISS indexes
- evidence retrieval with page/section citations
- a bounded LangGraph search/index/retrieve/answer loop
- CLI, Gradio UI, smoke tests, and local data isolation

## Quick start (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
research-agent doctor
research-agent search "agentic RAG scientific question answering" --category cs.AI --year 2025
research-agent index 2501.00001
research-agent ask "How does this paper reduce hallucination?" --paper-id 2501.00001
research-agent chat "Recent approaches to agentic RAG for scientific QA"
research-agent ui
```

Replace the sample arXiv ID with one returned by `search`. `chat` may search and
index up to two promising papers automatically; `ask` is the faster, controlled
single-paper path.

## Data layout

Runtime artifacts stay under `data/` and are ignored by Git:

- `data/research.db`: paper metadata
- `data/papers/`: downloaded PDFs
- `data/indexes/<arxiv-id>/`: FAISS index and chunk metadata

## Scope

This is intentionally V1. It does not ingest all of arXiv, fine-tune models, run
a multi-agent swarm, or process figures/tables yet. See `docs/ROADMAP.md`.

