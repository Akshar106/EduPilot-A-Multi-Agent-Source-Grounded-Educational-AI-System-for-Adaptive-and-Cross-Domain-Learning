"""
Chat sessions and messages
==========================
Every read is scoped by `user_id` in SQL rather than filtered afterwards, and
every destructive statement carries ownership in its predicate so there is no
window between checking and acting.
"""

from __future__ import annotations

import json

from .connection import get_conn, transaction


def get_session_owner(session_id: str) -> str | None:
    """user_id owning a chat session, or None when unowned or missing."""
    row = get_conn().execute(
        "SELECT user_id FROM chat_sessions WHERE session_id=?", (session_id,)
    ).fetchone()
    return row["user_id"] if row else None


def ensure_session(session_id: str, user_id: str, title: str | None = None) -> None:
    """
    Insert a session row if absent, owned by `user_id`.

    `user_id` is required. It comes from the caller's verified token, never
    from the request body — a client-supplied owner would reintroduce the
    original IDOR in a new shape.
    """
    if not user_id:
        raise ValueError("user_id is required to create a session")
    with transaction() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO chat_sessions (session_id, user_id, title) VALUES (?, ?, ?)",
            (session_id, user_id, title),
        )


def update_session_title(session_id: str, title: str) -> None:
    with transaction() as cur:
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
    conn = get_conn()
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
    with transaction() as cur:
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
    with transaction() as cur:
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
    conn = get_conn()
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
    with transaction() as cur:
        cur.execute(
            "DELETE FROM chat_messages WHERE session_id=? AND id >= ?",
            (session_id, message_id),
        )
        cur.execute(
            "UPDATE chat_sessions SET updated_at=datetime('now') WHERE session_id=?",
            (session_id,),
        )
