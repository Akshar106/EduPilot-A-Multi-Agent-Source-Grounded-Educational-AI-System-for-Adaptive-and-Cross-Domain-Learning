"""
Thread-local SQLite connections
===============================
Every EduPilot connection to the operational database is created here so the
pragmas below are applied exactly once, in one place.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from edupilot.core.config import SQLITE_DB_PATH

_local = threading.local()


#: How long a connection waits for a lock before giving up.
#:
#: SQLite defaults to 0, meaning any contention raises "database is locked"
#: immediately rather than waiting. Three components hold connections to this
#: file (this module, UserStore, IndexRegistry) and requests run across a
#: worker pool, so brief contention is routine and must be waited out.
BUSY_TIMEOUT_MS = 5000


def configure(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Apply the pragmas every EduPilot connection needs."""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")      # concurrent readers with one writer
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")    # safe under WAL, much faster
    return conn


def get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (created lazily)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = configure(sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False))
    return _local.conn


@contextmanager
def transaction() -> Iterator[sqlite3.Cursor]:
    """Yield a cursor, committing on success and rolling back on any exception."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
