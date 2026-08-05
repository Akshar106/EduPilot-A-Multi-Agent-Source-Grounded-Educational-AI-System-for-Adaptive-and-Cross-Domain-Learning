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


#: How long a connection waits for a lock before giving up.
#:
#: SQLite defaults to 0, meaning any contention raises "database is locked"
#: immediately rather than waiting. Three components hold connections to this
#: file (this module, UserStore, IndexRegistry) and requests run across a
#: worker pool, so brief contention is routine and must be waited out.
BUSY_TIMEOUT_MS = 5000


def _configure(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Apply the pragmas every EduPilot connection needs."""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")      # concurrent readers with one writer
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")    # safe under WAL, much faster
    return conn


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (created lazily)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = _configure(sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False))
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
CREATE TABLE IF NOT EXISTS self_study_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ss_session_id   TEXT    UNIQUE NOT NULL,
    name            TEXT    NOT NULL,
    description     TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS self_study_documents (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ss_session_id    TEXT    NOT NULL,
    filename         TEXT    NOT NULL,
    file_type        TEXT,
    file_size_bytes  INTEGER,
    chunk_count      INTEGER DEFAULT 0,
    upload_timestamp TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (ss_session_id) REFERENCES self_study_sessions(ss_session_id)
);

CREATE TABLE IF NOT EXISTS self_study_chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id        TEXT    UNIQUE NOT NULL,
    ss_session_id   TEXT    NOT NULL,
    text            TEXT    NOT NULL,
    source_file     TEXT    NOT NULL,
    page_number     INTEGER,
    chunk_index     INTEGER,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (ss_session_id) REFERENCES self_study_sessions(ss_session_id)
);

CREATE TABLE IF NOT EXISTS self_study_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ss_session_id   TEXT    NOT NULL,
    role            TEXT    NOT NULL CHECK(role IN ('user','assistant')),
    content         TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    quality_score   REAL,
    pipeline_meta   TEXT,
    FOREIGN KEY (ss_session_id) REFERENCES self_study_sessions(ss_session_id)
);

CREATE INDEX IF NOT EXISTS idx_ss_docs_session   ON self_study_documents(ss_session_id);
CREATE INDEX IF NOT EXISTS idx_ss_chunks_session ON self_study_chunks(ss_session_id, source_file);
CREATE INDEX IF NOT EXISTS idx_ss_msgs_session   ON self_study_messages(ss_session_id, timestamp);

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

#: Schema version. Bump when adding a migration in `_migrate`.
SCHEMA_VERSION = 2


def init_db() -> None:
    """Create all tables if they don't exist, then apply migrations."""
    with _cursor() as cur:
        cur.executescript(_SCHEMA)
    _migrate()


def _migrate() -> None:
    """
    Apply schema migrations, tracked via SQLite's `user_version` pragma.

    The original schema had no versioning — every table was `CREATE TABLE IF
    NOT EXISTS`, so a column added later would silently never appear on an
    existing database. This makes upgrades explicit and idempotent.
    """
    conn = _get_conn()
    current = conn.execute("PRAGMA user_version").fetchone()[0]

    if current < 2:
        # v2 — ownership. Sessions previously belonged to nobody, so any caller
        # who knew or guessed a session_id could read and delete it.
        _add_column_if_missing(conn, "chat_sessions", "user_id", "TEXT")
        _add_column_if_missing(conn, "self_study_sessions", "user_id", "TEXT")
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
                ON chat_sessions(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_ss_sessions_user
                ON self_study_sessions(user_id, updated_at DESC);
            """
        )
        conn.commit()

    if current < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """ALTER TABLE ADD COLUMN, skipping it when the column already exists."""
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def orphaned_session_count() -> dict[str, int]:
    """
    Count pre-auth sessions that have no owner.

    Rows created before ownership existed cannot be attributed to a user.
    They are left in place rather than deleted — destroying a student's
    history to satisfy a schema change is not an acceptable migration — but
    the scoped queries below never return them, so nobody can read them.
    """
    conn = _get_conn()
    return {
        "chat_sessions": conn.execute(
            "SELECT COUNT(*) AS n FROM chat_sessions WHERE user_id IS NULL"
        ).fetchone()["n"],
        "self_study_sessions": conn.execute(
            "SELECT COUNT(*) AS n FROM self_study_sessions WHERE user_id IS NULL"
        ).fetchone()["n"],
    }


def claim_orphaned_sessions(user_id: str) -> int:
    """
    Assign every unowned legacy session to `user_id`.

    For a single-user upgrade, so the developer running this locally keeps
    their existing history. Do not call it on a multi-user deployment.
    """
    with _cursor() as cur:
        cur.execute("UPDATE chat_sessions SET user_id=? WHERE user_id IS NULL", (user_id,))
        claimed = cur.rowcount or 0
        cur.execute(
            "UPDATE self_study_sessions SET user_id=? WHERE user_id IS NULL", (user_id,)
        )
        claimed += cur.rowcount or 0
    return claimed


# ---------------------------------------------------------------------------
# Ownership lookups
# ---------------------------------------------------------------------------


def get_session_owner(session_id: str) -> str | None:
    """user_id owning a chat session, or None when unowned or missing."""
    row = _get_conn().execute(
        "SELECT user_id FROM chat_sessions WHERE session_id=?", (session_id,)
    ).fetchone()
    return row["user_id"] if row else None


def get_ss_session_owner(ss_session_id: str) -> str | None:
    """user_id owning a study session, or None when unowned or missing."""
    row = _get_conn().execute(
        "SELECT user_id FROM self_study_sessions WHERE ss_session_id=?", (ss_session_id,)
    ).fetchone()
    return row["user_id"] if row else None


# ---------------------------------------------------------------------------
# Chat session helpers
# ---------------------------------------------------------------------------

def ensure_session(session_id: str, user_id: str, title: str | None = None) -> None:
    """
    Insert a session row if absent, owned by `user_id`.

    `user_id` is required. It comes from the caller's verified token, never
    from the request body — a client-supplied owner would reintroduce the
    original IDOR in a new shape.
    """
    if not user_id:
        raise ValueError("user_id is required to create a session")
    with _cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO chat_sessions (session_id, user_id, title) VALUES (?, ?, ?)",
            (session_id, user_id, title),
        )


def update_session_title(session_id: str, title: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE chat_sessions SET title=?, updated_at=datetime('now') WHERE session_id=?",
            (title, session_id),
        )


def list_sessions(user_id: str, limit: int = 20) -> list[dict]:
    """
    Recent sessions belonging to `user_id`, newest activity first.

    Scoped by owner in SQL rather than filtered afterwards. The previous
    version took no user at all and returned the twenty most recent sessions
    across every user, which is what made the other endpoints exploitable —
    it handed out the IDs needed to read and delete anyone's history.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT s.session_id, s.title, s.created_at, s.updated_at,
               COUNT(m.id) AS message_count
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON m.session_id = s.session_id
        WHERE s.user_id = ?
        GROUP BY s.session_id
        ORDER BY s.updated_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: str, user_id: str) -> bool:
    """
    Delete a session and its messages, only if `user_id` owns it.

    Ownership is part of the DELETE predicate rather than a prior SELECT, so
    there is no window between the check and the delete. Returns False when
    nothing matched — either the session does not exist or it is not theirs,
    and the caller must not distinguish the two.
    """
    with _cursor() as cur:
        cur.execute(
            "DELETE FROM chat_messages WHERE session_id IN "
            "(SELECT session_id FROM chat_sessions WHERE session_id=? AND user_id=?)",
            (session_id, user_id),
        )
        cur.execute(
            "DELETE FROM chat_sessions WHERE session_id=? AND user_id=?",
            (session_id, user_id),
        )
        return (cur.rowcount or 0) > 0


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
) -> int:
    """Persist one chat turn. Also touches updated_at on the parent session. Returns row id."""
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
        row_id: int = cur.lastrowid  # type: ignore[assignment]
        cur.execute(
            "UPDATE chat_sessions SET updated_at=datetime('now') WHERE session_id=?",
            (session_id,),
        )
    return row_id


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


def delete_messages_from(session_id: str, message_id: int) -> None:
    """Delete a message and all subsequent messages in the session (for edit/re-send)."""
    with _cursor() as cur:
        cur.execute(
            "DELETE FROM chat_messages WHERE session_id=? AND id >= ?",
            (session_id, message_id),
        )
        cur.execute(
            "UPDATE chat_sessions SET updated_at=datetime('now') WHERE session_id=?",
            (session_id,),
        )


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


# ---------------------------------------------------------------------------
# Self Study — Session helpers
# ---------------------------------------------------------------------------

def create_ss_session(
    ss_session_id: str, user_id: str, name: str, description: str | None = None
) -> None:
    """Create a study session owned by `user_id`."""
    if not user_id:
        raise ValueError("user_id is required to create a study session")
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO self_study_sessions (ss_session_id, user_id, name, description) "
            "VALUES (?, ?, ?, ?)",
            (ss_session_id, user_id, name, description),
        )


def list_ss_sessions(user_id: str) -> list[dict]:
    """
    Study sessions belonging to `user_id`.

    These carry the student's own uploaded documents, so cross-user exposure
    here leaks personal files, not just chat text.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT s.ss_session_id, s.name, s.description, s.created_at, s.updated_at,
               COUNT(DISTINCT d.id)            AS doc_count,
               COALESCE(SUM(d.chunk_count), 0) AS total_chunks
        FROM self_study_sessions s
        LEFT JOIN self_study_documents d ON d.ss_session_id = s.ss_session_id
        WHERE s.user_id = ?
        GROUP BY s.ss_session_id
        ORDER BY s.updated_at DESC
        """,
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_ss_session(ss_session_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM self_study_sessions WHERE ss_session_id=?",
        (ss_session_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_ss_session(ss_session_id: str) -> None:
    with _cursor() as cur:
        cur.execute("DELETE FROM self_study_messages  WHERE ss_session_id=?", (ss_session_id,))
        cur.execute("DELETE FROM self_study_chunks    WHERE ss_session_id=?", (ss_session_id,))
        cur.execute("DELETE FROM self_study_documents WHERE ss_session_id=?", (ss_session_id,))
        cur.execute("DELETE FROM self_study_sessions  WHERE ss_session_id=?", (ss_session_id,))


def touch_ss_session(ss_session_id: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "UPDATE self_study_sessions SET updated_at=datetime('now') WHERE ss_session_id=?",
            (ss_session_id,),
        )


# ---------------------------------------------------------------------------
# Self Study — Document helpers
# ---------------------------------------------------------------------------

def save_ss_document(
    ss_session_id: str,
    filename: str,
    file_type: str,
    file_size_bytes: int,
    chunk_count: int,
) -> int:
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO self_study_documents
                (ss_session_id, filename, file_type, file_size_bytes, chunk_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ss_session_id, filename, file_type, file_size_bytes, chunk_count),
        )
        return cur.lastrowid  # type: ignore[return-value]


def list_ss_documents(ss_session_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM self_study_documents WHERE ss_session_id=? ORDER BY upload_timestamp ASC",
        (ss_session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_ss_document(doc_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM self_study_documents WHERE id=?",
        (doc_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_ss_document_record(doc_id: int) -> None:
    with _cursor() as cur:
        cur.execute("DELETE FROM self_study_documents WHERE id=?", (doc_id,))


# ---------------------------------------------------------------------------
# Self Study — Chunk helpers
# ---------------------------------------------------------------------------

def save_ss_chunks(chunks: list[dict]) -> None:
    with _cursor() as cur:
        cur.executemany(
            """
            INSERT OR IGNORE INTO self_study_chunks
                (chunk_id, ss_session_id, text, source_file, page_number, chunk_index)
            VALUES (:chunk_id, :ss_session_id, :text, :source_file, :page_number, :chunk_index)
            """,
            chunks,
        )


def get_ss_chunks(ss_session_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT chunk_id, text, source_file, page_number, chunk_index "
        "FROM self_study_chunks WHERE ss_session_id=? ORDER BY id ASC",
        (ss_session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_ss_chunk_ids(ss_session_id: str) -> set[str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT chunk_id FROM self_study_chunks WHERE ss_session_id=?",
        (ss_session_id,),
    ).fetchall()
    return {r["chunk_id"] for r in rows}


def ss_chunk_count(ss_session_id: str) -> int:
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM self_study_chunks WHERE ss_session_id=?",
        (ss_session_id,),
    ).fetchone()
    return row["n"] if row else 0


def get_ss_chunk_ids_by_source(ss_session_id: str, source_file: str) -> list[str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT chunk_id FROM self_study_chunks WHERE ss_session_id=? AND source_file LIKE ?",
        (ss_session_id, f"%{source_file}%"),
    ).fetchall()
    return [r["chunk_id"] for r in rows]


def delete_ss_chunks_by_source(ss_session_id: str, source_file: str) -> int:
    with _cursor() as cur:
        cur.execute(
            "DELETE FROM self_study_chunks WHERE ss_session_id=? AND source_file LIKE ?",
            (ss_session_id, f"%{source_file}%"),
        )
        return cur.rowcount  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Self Study — Message helpers
# ---------------------------------------------------------------------------

def save_ss_message(
    ss_session_id: str,
    role: str,
    content: str,
    quality_score: float | None = None,
    pipeline_meta: dict | None = None,
) -> int:
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO self_study_messages
                (ss_session_id, role, content, quality_score, pipeline_meta)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ss_session_id,
                role,
                content,
                quality_score,
                json.dumps(pipeline_meta, default=str) if pipeline_meta else None,
            ),
        )
        row_id: int = cur.lastrowid  # type: ignore[assignment]
        cur.execute(
            "UPDATE self_study_sessions SET updated_at=datetime('now') WHERE ss_session_id=?",
            (ss_session_id,),
        )
        return row_id


def get_ss_messages(ss_session_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM self_study_messages WHERE ss_session_id=? ORDER BY timestamp ASC",
        (ss_session_id,),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("pipeline_meta"):
            d["pipeline_meta"] = json.loads(d["pipeline_meta"])
        result.append(d)
    return result
