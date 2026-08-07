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

import logging
import sqlite3

from .connection import get_conn, transaction

logger = logging.getLogger(__name__)

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
SCHEMA_VERSION = 3


def init_db() -> None:
    """Create all tables if they don't exist, then apply migrations."""
    with transaction() as cur:
        cur.executescript(_SCHEMA)
    _migrate()


#: Columns added after the original schema, as (table, column, declaration).
#:
#: Reconciled against the live schema on every startup rather than gated on
#: `user_version`. A version-gated migration cannot repair itself: if the
#: counter is advanced but the ALTER does not land — an interrupted startup, or
#: a version bump committed before its migration body — the column is missing
#: forever and every query touching it fails. `_add_column_if_missing` is
#: idempotent and costs one PRAGMA per column, so there is nothing to gain by
#: skipping it.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # v2 — ownership. Sessions previously belonged to nobody, so any caller who
    # knew or guessed a session_id could read and delete it.
    ("chat_sessions", "user_id", "TEXT"),
    ("self_study_sessions", "user_id", "TEXT"),
    # v3 — rolling conversation memory. Older turns are compacted into a digest
    # so a long chat stays coherent without the prompt growing without bound.
    # `summary_through_id` records the last message already folded in, so
    # summarization is incremental rather than re-reading the whole conversation.
    ("chat_sessions", "summary", "TEXT"),
    ("chat_sessions", "summary_through_id", "INTEGER"),
)

_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
    ON chat_sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ss_sessions_user
    ON self_study_sessions(user_id, updated_at DESC);
"""


def _migrate() -> None:
    """
    Bring an existing database up to `SCHEMA_VERSION`.

    The original schema had no versioning — every table was `CREATE TABLE IF
    NOT EXISTS`, so a column added later would silently never appear on an
    existing database.

    Additive changes are reconciled against the real schema, so a database
    whose `user_version` claims to be current but is missing a column repairs
    itself on the next start. `user_version` records progress; it is not the
    source of truth.
    """
    conn = get_conn()
    current = conn.execute("PRAGMA user_version").fetchone()[0]

    repaired = []
    for table, column, decl in _ADDED_COLUMNS:
        if _add_column_if_missing(conn, table, column, decl):
            repaired.append(f"{table}.{column}")

    conn.executescript(_MIGRATION_INDEXES)
    conn.commit()

    if repaired and current >= SCHEMA_VERSION:
        # The counter said "current" while the schema was not. Worth a line in
        # the log: it means a previous startup was interrupted mid-migration.
        logger.warning(
            "schema repaired columns missing despite user_version=%d: %s",
            current, ", ".join(repaired),
        )
    elif repaired:
        logger.info("schema migrated to v%d: added %s", SCHEMA_VERSION, ", ".join(repaired))

    if current < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, decl: str
) -> bool:
    """
    ALTER TABLE ADD COLUMN, skipping it when the column already exists.

    Returns True if the column was actually added, so the caller can tell a
    genuine migration from a no-op.
    """
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column in existing:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    return True


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
