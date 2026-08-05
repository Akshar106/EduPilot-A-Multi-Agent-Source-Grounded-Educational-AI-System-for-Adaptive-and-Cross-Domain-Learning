"""
Course knowledge-base documents and their chunks
================================================
The shared, admin-curated corpus. Chunk text is kept here so a vector index
can be rebuilt from scratch without re-parsing the source PDFs.
"""

from __future__ import annotations

from .connection import get_conn, transaction


def save_uploaded_doc(
    filename: str,
    domain: str,
    file_type: str,
    chunk_count: int,
    file_size_bytes: int,
) -> int:
    """Insert a row for an uploaded document. Returns the new row id."""
    with transaction() as cur:
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
    conn = get_conn()
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


def save_chunks(chunks: list[dict]) -> None:
    """
    Bulk-insert chunk rows.
    Each dict must have: chunk_id, domain, text, source_file,
                         page_number (int|None), chunk_index (int|None).
    Ignores duplicates (INSERT OR IGNORE).
    """
    with transaction() as cur:
        cur.executemany(
            """
            INSERT OR IGNORE INTO document_chunks
                (chunk_id, domain, text, source_file, page_number, chunk_index)
            VALUES (:chunk_id, :domain, :text, :source_file, :page_number, :chunk_index)
            """,
            chunks,
        )


def get_chunks_by_domain(domain: str) -> list[dict]:
    """Return all chunks for a domain — used to refit the sparse encoder."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT chunk_id, text, source_file, page_number, chunk_index "
        "FROM document_chunks WHERE domain=? ORDER BY id ASC",
        (domain,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_chunk_ids_by_domain(domain: str) -> set[str]:
    """Return the set of chunk_ids already indexed for a domain."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT chunk_id FROM document_chunks WHERE domain=?",
        (domain,),
    ).fetchall()
    return {r["chunk_id"] for r in rows}


def chunk_count_by_domain(domain: str) -> int:
    """Fast count of indexed chunks for a domain."""
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM document_chunks WHERE domain=?",
        (domain,),
    ).fetchone()
    return row["n"] if row else 0


def delete_chunks_by_domain(domain: str) -> None:
    """Wipe all chunks for a domain, ahead of a full re-index."""
    with transaction() as cur:
        cur.execute("DELETE FROM document_chunks WHERE domain=?", (domain,))
