# arXiv metadata and PDF policy

## Time fields

- `first_submitted_at`: when arXiv processed version 1 (`published` in Atom).
- `last_revised_at`: when arXiv processed the retrieved version (`updated` in Atom).
- `journal_published_at`: reserved for an external publisher/DOI source; it is
  never inferred from an arXiv ID or arXiv timestamps.

`--date-mode first_submitted` is the reproducible default. The arXiv Atom API
only exposes a `submittedDate` filter. `last_revised` and `active_in_period`
therefore scan a bounded relevance window and validate `updated` client-side;
they are not exhaustive historical harvests.

## Identity and revisions

The database stores both a stable work ID and the retrieved revision:

```text
arxiv_id      = 1706.03762
versioned_id  = 1706.03762v7
version       = 7
```

Citations use `versioned_id`. When metadata reports a different version, the
current PDF and vector-index artifact fields are marked stale. Re-indexing writes
an `index_meta.json` containing the revision, PDF SHA-256, embedding model, and
build timestamp. Retrieval considers an index current only when these values
match the database.

## PDF lifecycle

Downloads are written to a temporary `.part` file. Before atomic rename, the
pipeline verifies:

1. the file starts with `%PDF-`;
2. PyMuPDF can open it;
3. it contains at least one page.

The final artifact stores its SHA-256 and byte size. HTTP 404/410 becomes
`unavailable`; transient HTTP or invalid-content failures become
`download_failed`; a readable PDF without extractable text becomes
`no_text_layer`.

The LangGraph workflow treats these failures as recoverable. It records a tool
note and uses the paper abstract as lower-confidence evidence instead of
inventing full-text evidence or a page number.

