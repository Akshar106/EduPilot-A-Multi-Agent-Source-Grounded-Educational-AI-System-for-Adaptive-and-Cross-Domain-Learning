"""
Schema definition and migrations
================================
Tables that are not vector embeddings:

  • chat_sessions        — one row per conversation, owned by a user
  • chat_messages        — every user / assistant turn with pipeline metadata
  • uploaded_documents   — metadata for every file uploaded to the course KB
  • document_chunks      — raw chunk text, kept so an index can be rebuilt
  • self_study_*         — the private, per-student mirror of the above

Pinecone stores only the embedding vectors; the text lives here.
"""

from __future__ import annotations

import sqlite3

from .connection import get_conn, transaction

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
    with transaction() as cur:
        cur.executescript(_SCHEMA)
    _migrate()


def _migrate() -> None:
    """
    Apply schema migrations, tracked via SQLite's `user_version` pragma.

    The original schema had no versioning — every table was `CREATE TABLE IF
    NOT EXISTS`, so a column added later would silently never appear on an
    existing database. This makes upgrades explicit and idempotent.
    """
    conn = get_conn()
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


# ---------------------------------------------------------------------------
# Pre-ownership rows
#
# These exist only because the v2 migration introduced `user_id` on tables that
# already had rows. They can be dropped once no deployment predates v2.
# ---------------------------------------------------------------------------


def orphaned_session_count() -> dict[str, int]:
    """
    Count pre-auth sessions that have no owner.

    Rows created before ownership existed cannot be attributed to a user.
    They are left in place rather than deleted — destroying a student's
    history to satisfy a schema change is not an acceptable migration — but
    the scoped queries elsewhere never return them, so nobody can read them.
    """
    conn = get_conn()
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
    with transaction() as cur:
        cur.execute("UPDATE chat_sessions SET user_id=? WHERE user_id IS NULL", (user_id,))
        claimed = cur.rowcount or 0
        cur.execute(
            "UPDATE self_study_sessions SET user_id=? WHERE user_id IS NULL", (user_id,)
        )
        claimed += cur.rowcount or 0
    return claimed
