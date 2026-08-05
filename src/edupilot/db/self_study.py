"""
Self Study — private per-student sessions
=========================================
The student's own uploads, kept apart from the shared course corpus. Exposure
here leaks personal files rather than lecture text, so every session read goes
through an ownership check at the route layer.
"""

from __future__ import annotations

import json

from .connection import get_conn, transaction


def get_ss_session_owner(ss_session_id: str) -> str | None:
    """user_id owning a study session, or None when unowned or missing."""
    row = get_conn().execute(
        "SELECT user_id FROM self_study_sessions WHERE ss_session_id=?", (ss_session_id,)
    ).fetchone()
    return row["user_id"] if row else None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def create_ss_session(
    ss_session_id: str, user_id: str, name: str, description: str | None = None
) -> None:
    """Create a study session owned by `user_id`."""
    if not user_id:
        raise ValueError("user_id is required to create a study session")
    with transaction() as cur:
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
    conn = get_conn()
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
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM self_study_sessions WHERE ss_session_id=?",
        (ss_session_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_ss_session(ss_session_id: str) -> None:
    with transaction() as cur:
        cur.execute("DELETE FROM self_study_messages  WHERE ss_session_id=?", (ss_session_id,))
        cur.execute("DELETE FROM self_study_chunks    WHERE ss_session_id=?", (ss_session_id,))
        cur.execute("DELETE FROM self_study_documents WHERE ss_session_id=?", (ss_session_id,))
        cur.execute("DELETE FROM self_study_sessions  WHERE ss_session_id=?", (ss_session_id,))


def touch_ss_session(ss_session_id: str) -> None:
    with transaction() as cur:
        cur.execute(
            "UPDATE self_study_sessions SET updated_at=datetime('now') WHERE ss_session_id=?",
            (ss_session_id,),
        )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def save_ss_document(
    ss_session_id: str,
    filename: str,
    file_type: str,
    file_size_bytes: int,
    chunk_count: int,
) -> int:
    with transaction() as cur:
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
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM self_study_documents WHERE ss_session_id=? ORDER BY upload_timestamp ASC",
        (ss_session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_ss_document(doc_id: int) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM self_study_documents WHERE id=?",
        (doc_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_ss_document_record(doc_id: int) -> None:
    with transaction() as cur:
        cur.execute("DELETE FROM self_study_documents WHERE id=?", (doc_id,))


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


def save_ss_chunks(chunks: list[dict]) -> None:
    with transaction() as cur:
        cur.executemany(
            """
            INSERT OR IGNORE INTO self_study_chunks
                (chunk_id, ss_session_id, text, source_file, page_number, chunk_index)
            VALUES (:chunk_id, :ss_session_id, :text, :source_file, :page_number, :chunk_index)
            """,
            chunks,
        )


def get_ss_chunks(ss_session_id: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT chunk_id, text, source_file, page_number, chunk_index "
        "FROM self_study_chunks WHERE ss_session_id=? ORDER BY id ASC",
        (ss_session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_ss_chunk_ids(ss_session_id: str) -> set[str]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT chunk_id FROM self_study_chunks WHERE ss_session_id=?",
        (ss_session_id,),
    ).fetchall()
    return {r["chunk_id"] for r in rows}


def ss_chunk_count(ss_session_id: str) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM self_study_chunks WHERE ss_session_id=?",
        (ss_session_id,),
    ).fetchone()
    return row["n"] if row else 0


def get_ss_chunk_ids_by_source(ss_session_id: str, source_file: str) -> list[str]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT chunk_id FROM self_study_chunks WHERE ss_session_id=? AND source_file LIKE ?",
        (ss_session_id, f"%{source_file}%"),
    ).fetchall()
    return [r["chunk_id"] for r in rows]


def delete_ss_chunks_by_source(ss_session_id: str, source_file: str) -> int:
    with transaction() as cur:
        cur.execute(
            "DELETE FROM self_study_chunks WHERE ss_session_id=? AND source_file LIKE ?",
            (ss_session_id, f"%{source_file}%"),
        )
        return cur.rowcount  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def save_ss_message(
    ss_session_id: str,
    role: str,
    content: str,
    quality_score: float | None = None,
    pipeline_meta: dict | None = None,
) -> int:
    with transaction() as cur:
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
    conn = get_conn()
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
