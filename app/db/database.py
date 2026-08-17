import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    versioned_id TEXT,
    version INTEGER,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    primary_category TEXT,
    published TEXT,
    updated TEXT,
    first_submitted_at TEXT,
    last_revised_at TEXT,
    doi TEXT,
    journal_ref TEXT,
    journal_published_at TEXT,
    comment TEXT,
    pdf_url TEXT,
    pdf_status TEXT NOT NULL DEFAULT 'unknown',
    pdf_path TEXT,
    pdf_sha256 TEXT,
    pdf_size INTEGER,
    downloaded_at TEXT,
    indexed_version INTEGER,
    indexed_pdf_sha256 TEXT,
    indexed_at TEXT,
    created_at TEXT NOT NULL,
    metadata_checked_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    arxiv_id UNINDEXED, title, abstract
);
"""


MIGRATION_COLUMNS = {
    "versioned_id": "TEXT",
    "version": "INTEGER",
    "primary_category": "TEXT",
    "first_submitted_at": "TEXT",
    "last_revised_at": "TEXT",
    "journal_ref": "TEXT",
    "journal_published_at": "TEXT",
    "comment": "TEXT",
    "pdf_status": "TEXT NOT NULL DEFAULT 'unknown'",
    "pdf_sha256": "TEXT",
    "pdf_size": "INTEGER",
    "downloaded_at": "TEXT",
    "indexed_version": "INTEGER",
    "indexed_pdf_sha256": "TEXT",
    "metadata_checked_at": "TEXT",
}

SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "các",
    "cho",
    "của",
    "does",
    "for",
    "how",
    "in",
    "is",
    "là",
    "những",
    "of",
    "paper",
    "papers",
    "the",
    "this",
    "to",
    "và",
    "what",
    "which",
    "with",
}


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(papers)")}
    for name, definition in MIGRATION_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {name} {definition}")
    conn.execute(
        """UPDATE papers SET
        first_submitted_at = COALESCE(first_submitted_at, published),
        last_revised_at = COALESCE(last_revised_at, updated),
        metadata_checked_at = COALESCE(metadata_checked_at, created_at)"""
    )


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    settings.ensure_directories()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _deserialize(row: sqlite3.Row | None) -> dict | None:
    if not row:
        return None
    result = dict(row)
    result["authors"] = json.loads(result.pop("authors_json"))
    result["categories"] = json.loads(result.pop("categories_json"))
    return result


def upsert_paper(paper: dict) -> None:
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        previous = conn.execute(
            "SELECT version FROM papers WHERE arxiv_id = ?", (paper["arxiv_id"],)
        ).fetchone()
        version_changed = bool(
            previous
            and previous["version"] is not None
            and paper.get("version") is not None
            and previous["version"] != paper["version"]
        )
        conn.execute(
            """INSERT INTO papers (
                arxiv_id, versioned_id, version, title, abstract, authors_json,
                categories_json, primary_category, published, updated,
                first_submitted_at, last_revised_at, doi, journal_ref,
                comment, pdf_url, created_at, metadata_checked_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                versioned_id=excluded.versioned_id, version=excluded.version,
                title=excluded.title, abstract=excluded.abstract,
                authors_json=excluded.authors_json,
                categories_json=excluded.categories_json,
                primary_category=excluded.primary_category,
                published=excluded.published, updated=excluded.updated,
                first_submitted_at=excluded.first_submitted_at,
                last_revised_at=excluded.last_revised_at,
                doi=excluded.doi, journal_ref=excluded.journal_ref,
                comment=excluded.comment, pdf_url=excluded.pdf_url,
                metadata_checked_at=excluded.metadata_checked_at""",
            (
                paper["arxiv_id"],
                paper.get("versioned_id"),
                paper.get("version"),
                paper["title"],
                paper["abstract"],
                json.dumps(paper["authors"]),
                json.dumps(paper["categories"]),
                paper.get("primary_category"),
                paper.get("published"),
                paper.get("updated"),
                paper.get("first_submitted_at"),
                paper.get("last_revised_at"),
                paper.get("doi"),
                paper.get("journal_ref"),
                paper.get("comment"),
                paper.get("pdf_url"),
                now,
                now,
            ),
        )
        if version_changed:
            conn.execute(
                """UPDATE papers SET pdf_status='stale', pdf_path=NULL,
                pdf_sha256=NULL, pdf_size=NULL, downloaded_at=NULL,
                indexed_version=NULL, indexed_pdf_sha256=NULL, indexed_at=NULL
                WHERE arxiv_id=?""",
                (paper["arxiv_id"],),
            )
        conn.execute("DELETE FROM papers_fts WHERE arxiv_id = ?", (paper["arxiv_id"],))
        conn.execute(
            "INSERT INTO papers_fts(arxiv_id, title, abstract) VALUES (?, ?, ?)",
            (paper["arxiv_id"], paper["title"], paper["abstract"]),
        )


def get_paper(arxiv_id: str) -> dict | None:
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    with connect() as conn:
        row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (base_id,)).fetchone()
    return _deserialize(row)


def update_pdf_artifact(
    arxiv_id: str,
    *,
    status: str,
    path: str | None = None,
    sha256: str | None = None,
    size: int | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE papers SET pdf_status=?, pdf_path=?, pdf_sha256=?, pdf_size=?,
            downloaded_at=? WHERE arxiv_id=?""",
            (
                status,
                path,
                sha256,
                size,
                datetime.now(UTC).isoformat() if status == "available" else None,
                arxiv_id,
            ),
        )


def set_pdf_status(arxiv_id: str, status: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE papers SET pdf_status=? WHERE arxiv_id=?", (status, arxiv_id))


def update_index_artifact(arxiv_id: str, *, version: int | None, pdf_sha256: str) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE papers SET indexed_version=?, indexed_pdf_sha256=?, indexed_at=?
            WHERE arxiv_id=?""",
            (version, pdf_sha256, datetime.now(UTC).isoformat(), arxiv_id),
        )


def search_local(query: str, limit: int = 10) -> list[dict]:
    tokens = [
        token
        for token in re.findall(r"[\w.-]+", query.lower(), flags=re.UNICODE)
        if len(token) >= 2 and token not in SEARCH_STOPWORDS
    ]
    if not tokens:
        return []
    fts_query = " AND ".join(f'"{token}"' for token in tokens)
    with connect() as conn:
        rows = conn.execute(
            """SELECT p.* FROM papers_fts f JOIN papers p USING(arxiv_id)
            WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?""",
            (fts_query, limit),
        ).fetchall()
    return [_deserialize(row) for row in rows]
