import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    published TEXT,
    updated TEXT,
    doi TEXT,
    pdf_url TEXT NOT NULL,
    pdf_path TEXT,
    indexed_at TEXT,
    created_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    arxiv_id UNINDEXED, title, abstract
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    settings.ensure_directories()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_paper(paper: dict) -> None:
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        conn.execute(
            """INSERT INTO papers (
                arxiv_id, title, abstract, authors_json, categories_json,
                published, updated, doi, pdf_url, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                title=excluded.title, abstract=excluded.abstract,
                authors_json=excluded.authors_json, categories_json=excluded.categories_json,
                published=excluded.published, updated=excluded.updated,
                doi=excluded.doi, pdf_url=excluded.pdf_url""",
            (
                paper["arxiv_id"], paper["title"], paper["abstract"],
                json.dumps(paper["authors"]), json.dumps(paper["categories"]),
                paper.get("published"), paper.get("updated"), paper.get("doi"),
                paper["pdf_url"], now,
            ),
        )
        conn.execute("DELETE FROM papers_fts WHERE arxiv_id = ?", (paper["arxiv_id"],))
        conn.execute(
            "INSERT INTO papers_fts(arxiv_id, title, abstract) VALUES (?, ?, ?)",
            (paper["arxiv_id"], paper["title"], paper["abstract"]),
        )


def get_paper(arxiv_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["authors"] = json.loads(result.pop("authors_json"))
    result["categories"] = json.loads(result.pop("categories_json"))
    return result


def update_artifact(arxiv_id: str, *, pdf_path: str | None = None, indexed: bool = False) -> None:
    fields, values = [], []
    if pdf_path is not None:
        fields.append("pdf_path = ?")
        values.append(pdf_path)
    if indexed:
        fields.append("indexed_at = ?")
        values.append(datetime.now(UTC).isoformat())
    if not fields:
        return
    values.append(arxiv_id)
    with connect() as conn:
        conn.execute(f"UPDATE papers SET {', '.join(fields)} WHERE arxiv_id = ?", values)


def search_local(query: str, limit: int = 10) -> list[dict]:
    safe = " ".join(part.replace('"', "") for part in query.split() if part)
    if not safe:
        return []
    with connect() as conn:
        rows = conn.execute(
            """SELECT p.* FROM papers_fts f JOIN papers p USING(arxiv_id)
            WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?""",
            (safe, limit),
        ).fetchall()
    return [dict(row) for row in rows]

