"""
Operational database
====================
SQLite-backed persistence for everything that is not an embedding vector.
Import the package, not its modules::

    from edupilot import db

    db.init_db()
    db.save_message(session_id, "user", text)

The split mirrors the three concerns in the schema: `chat` (conversations),
`documents` (the shared course corpus), and `self_study` (private per-student
uploads). `connection` owns the thread-local handles; `schema` owns DDL and
migrations.
"""

from .chat import (
    delete_messages_from,
    delete_session,
    ensure_session,
    get_session_messages,
    get_session_owner,
    list_sessions,
    save_message,
    update_session_title,
)
from .connection import BUSY_TIMEOUT_MS, get_conn, transaction
from .documents import (
    chunk_count_by_domain,
    delete_chunks_by_domain,
    get_chunk_ids_by_domain,
    get_chunks_by_domain,
    list_uploaded_docs,
    save_chunks,
    save_uploaded_doc,
)
from .schema import (
    SCHEMA_VERSION,
    claim_orphaned_sessions,
    init_db,
    orphaned_session_count,
)
from .self_study import (
    create_ss_session,
    delete_ss_chunks_by_source,
    delete_ss_document_record,
    delete_ss_session,
    get_ss_chunk_ids,
    get_ss_chunk_ids_by_source,
    get_ss_chunks,
    get_ss_document,
    get_ss_messages,
    get_ss_session,
    get_ss_session_owner,
    list_ss_documents,
    list_ss_sessions,
    save_ss_chunks,
    save_ss_document,
    save_ss_message,
    ss_chunk_count,
    touch_ss_session,
)

__all__ = [
    # connection / schema
    "BUSY_TIMEOUT_MS",
    "SCHEMA_VERSION",
    "get_conn",
    "transaction",
    "init_db",
    "claim_orphaned_sessions",
    "orphaned_session_count",
    # chat
    "delete_messages_from",
    "delete_session",
    "ensure_session",
    "get_session_messages",
    "get_session_owner",
    "list_sessions",
    "save_message",
    "update_session_title",
    # documents
    "chunk_count_by_domain",
    "delete_chunks_by_domain",
    "get_chunk_ids_by_domain",
    "get_chunks_by_domain",
    "list_uploaded_docs",
    "save_chunks",
    "save_uploaded_doc",
    # self study
    "create_ss_session",
    "delete_ss_chunks_by_source",
    "delete_ss_document_record",
    "delete_ss_session",
    "get_ss_chunk_ids",
    "get_ss_chunk_ids_by_source",
    "get_ss_chunks",
    "get_ss_document",
    "get_ss_messages",
    "get_ss_session",
    "get_ss_session_owner",
    "list_ss_documents",
    "list_ss_sessions",
    "save_ss_chunks",
    "save_ss_document",
    "save_ss_message",
    "ss_chunk_count",
    "touch_ss_session",
]
