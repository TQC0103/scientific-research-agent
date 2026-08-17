# Agentic Scientific Research Assistant

Local-first V1 for searching arXiv, lazily downloading papers, section-aware PDF
chunking, FAISS retrieval, and citation-grounded answers with LangGraph + Ollama.

## What is ready

- arXiv search with year/category filters and SQLite metadata persistence
- explicit first-submission vs last-revision dates and version-aware citations
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
research-agent search "transformer attention" --year 2025 --date-mode last_revised
research-agent local-search "scientific retrieval"
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

`first_submitted_at` is the first arXiv submission and `last_revised_at` is the
retrieved arXiv version's update time. Neither is a journal publication date.
Versioned IDs such as `1706.03762v7` are preserved in citations, and a new
revision invalidates stale PDF/index metadata.

The Atom API has no last-revision date filter. `--date-mode last_revised` scans
a bounded relevance window and validates dates client-side, so it is useful for
focused queries but is not a complete historical harvest. Use arXiv OAI-PMH for
exhaustive corpus synchronization.

PDF downloads are staged in a temporary file and verified by signature and
PyMuPDF before atomic rename. HTTP/content failures are recorded; the graph can
fall back to abstract-only evidence and labels it as `Abstract` rather than
inventing a page citation. See `docs/METADATA_AND_PDF_POLICY.md`.

## Scope

This is intentionally V1. It does not ingest all of arXiv, fine-tune models, run
a multi-agent swarm, or process figures/tables yet. See `docs/ROADMAP.md`.
