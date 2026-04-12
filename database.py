"""
EduPilot — SQLite Database Layer
==================================
Handles all persistent storage that isn't vector embeddings:

  • chat_sessions    — one row per browser session
  • chat_messages    — every user / assistant turn with pipeline metadata
  • uploaded_documents — metadata for every file a user uploads
  • document_chunks  — raw text + metadata for every indexed chunk
                       (used to rebuild the BM25 index on startup)

Why SQLite?
  - Zero-config, file-based, ships with Python
  - Chunk text lives here; Pinecone stores only the embedding vectors
  - BM25 is rebuilt on startup from this table (no BM25 serialisation needed)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from config import SQLITE_DB_PATH

# ---------------------------------------------------------------------------
# Thread-local connection pool
# ---------------------------------------------------------------------------
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (created lazily)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # better concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    """Context manager that yields a cursor and commits on success."""
    conn = _get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    UNIQUE NOT NULL,
    title       TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL,
    role             TEXT    NOT NULL CHECK(role IN ('user','assistant')),
    content          TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL DEFAULT (datetime('now')),
    intent_type      TEXT,
    detected_domains TEXT,          -- JSON array  e.g. '["AML","STAT"]'
    quality_score    REAL,
    pipeline_meta    TEXT,          -- JSON blob of full debug dict
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
);

CREATE TABLE IF NOT EXISTS uploaded_documents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    filename         TEXT    NOT NULL,
    domain           TEXT    NOT NULL,
    upload_timestamp TEXT    NOT NULL DEFAULT (datetime('now')),
    file_type        TEXT,
    chunk_count      INTEGER DEFAULT 0,
    file_size_bytes  INTEGER
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id      TEXT    UNIQUE NOT NULL,   -- same ID upserted to Pinecone
    domain        TEXT    NOT NULL,
    text          TEXT    NOT NULL,
    source_file   TEXT    NOT NULL,
    page_number   INTEGER,
    chunk_index   INTEGER,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON chat_messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_domain
    ON document_chunks(domain);
CREATE INDEX IF NOT EXISTS idx_uploads_domain
    ON uploaded_documents(domain);
"""


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    with _cursor() as cur:
        cur.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Chat session helpers
# ---------------------------------------------------------------------------

def ensure_session(session_id: str, title: str | None = None) -> None:
    """Insert a session row if it doesn't exist yet."""
    with _cursor() as cur:
        cur.execute(
            """
            INSERT OR IGNORE INTO chat_sessions (session_id, title)
            VALUES (?, ?)
            """,
            (session_id, title),
        )


def update_session_title(session_id: str, title: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE chat_sessions SET title=?, updated_at=datetime('now') WHERE session_id=?",
            (title, session_id),
        )


def list_sessions(limit: int = 20) -> list[dict]:
    """Return recent sessions ordered by last activity."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT s.session_id, s.title, s.created_at,
               COUNT(m.id) AS message_count
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON m.session_id = s.session_id
        GROUP BY s.session_id
        ORDER BY s.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> None:
    """Delete a session and all its messages."""
    with _cursor() as cur:
        cur.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        cur.execute("DELETE FROM chat_sessions  WHERE session_id=?", (session_id,))


# ---------------------------------------------------------------------------
# Chat message helpers
# ---------------------------------------------------------------------------

def save_message(
    session_id: str,
    role: str,
    content: str,
    intent_type: str | None = None,
    detected_domains: list[str] | None = None,
    quality_score: float | None = None,
    pipeline_meta: dict | None = None,
) -> None:
    """Persist one chat turn. Also touches updated_at on the parent session."""
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_messages
                (session_id, role, content, intent_type,
                 detected_domains, quality_score, pipeline_meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                intent_type,
                json.dumps(detected_domains) if detected_domains else None,
                quality_score,
                json.dumps(pipeline_meta, default=str) if pipeline_meta else None,
            ),
        )
        cur.execute(
            "UPDATE chat_sessions SET updated_at=datetime('now') WHERE session_id=?",
            (session_id,),
        )


def get_session_messages(session_id: str) -> list[dict]:
    """Return all messages for a session in chronological order."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM chat_messages WHERE session_id=? ORDER BY timestamp ASC",
        (session_id,),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("detected_domains"):
            d["detected_domains"] = json.loads(d["detected_domains"])
        if d.get("pipeline_meta"):
            d["pipeline_meta"] = json.loads(d["pipeline_meta"])
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Uploaded document helpers
# ---------------------------------------------------------------------------

def save_uploaded_doc(
    filename: str,
    domain: str,
    file_type: str,
    chunk_count: int,
    file_size_bytes: int,
) -> int:
    """Insert a row for an uploaded document. Returns the new row id."""
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO uploaded_documents
                (filename, domain, file_type, chunk_count, file_size_bytes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (filename, domain, file_type, chunk_count, file_size_bytes),
        )
        return cur.lastrowid  # type: ignore[return-value]


def list_uploaded_docs(domain: str | None = None) -> list[dict]:
    """Return uploaded documents, optionally filtered by domain."""
    conn = _get_conn()
    if domain:
        rows = conn.execute(
            "SELECT * FROM uploaded_documents WHERE domain=? ORDER BY upload_timestamp DESC",
            (domain,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM uploaded_documents ORDER BY upload_timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Document chunk helpers (used by retriever.py for BM25 + dedup)
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict]) -> None:
    """
    Bulk-insert chunk rows.
    Each dict must have: chunk_id, domain, text, source_file,
                         page_number (int|None), chunk_index (int|None).
    Ignores duplicates (INSERT OR IGNORE).
    """
    with _cursor() as cur:
        cur.executemany(
            """
            INSERT OR IGNORE INTO document_chunks
                (chunk_id, domain, text, source_file, page_number, chunk_index)
            VALUES (:chunk_id, :domain, :text, :source_file, :page_number, :chunk_index)
            """,
            chunks,
        )


def get_chunks_by_domain(domain: str) -> list[dict]:
    """Return all chunks for a domain — used to rebuild BM25 on startup."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT chunk_id, text, source_file, page_number, chunk_index "
        "FROM document_chunks WHERE domain=? ORDER BY id ASC",
        (domain,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_chunk_ids_by_domain(domain: str) -> set[str]:
    """Return the set of chunk_ids already indexed for a domain."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT chunk_id FROM document_chunks WHERE domain=?",
        (domain,),
    ).fetchall()
    return {r["chunk_id"] for r in rows}


def chunk_count_by_domain(domain: str) -> int:
    """Fast count of indexed chunks for a domain."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM document_chunks WHERE domain=?",
        (domain,),
    ).fetchone()
    return row["n"] if row else 0


def delete_chunks_by_domain(domain: str) -> None:
    """Wipe all chunks for a domain (used by retriever.reset())."""
    with _cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE domain=?", (domain,))
