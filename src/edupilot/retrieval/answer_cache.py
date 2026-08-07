"""
Frequency-gated semantic answer cache
=====================================
Caches answers to questions that are actually asked repeatedly, and nothing
else.

Two decisions distinguish it from a plain response cache:

**Frequency gate.** Every question increments a counter; only on the
`PROMOTE_AFTER`-th ask is the answer stored. A class of students asks a long
tail of one-off questions, and storing those buys nothing while making stale
answers more likely. The counter is cheap — a row per normalized question —
and the payload is only written for questions that earn it.

**Semantic matching.** A hit requires cosine similarity above
`SIMILARITY_FLOOR` against a cached question's embedding, so "what is a
p-value" reaches "explain p-values". Exact matching would almost never fire on
real phrasing. The floor is deliberately high: serving a confidently wrong
answer to a *near-miss* question is worse than the cache missing.

**Invalidation.** Every entry records the index version that produced it. A
rebuild promotes a new version, so entries from an older corpus stop matching
and are swept — a re-index can never serve answers grounded in documents that
have since changed.

Only answers that were not refused and that carry a measured grounding score
above `MIN_GROUNDING_TO_CACHE` are stored: a refusal or a weakly-grounded
answer is exactly what a corpus fix is meant to change.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Asks required before an answer is stored. The first two are counted only.
PROMOTE_AFTER = 3

#: Cosine similarity required to reuse a cached answer.
#:
#: Calibrated for normalized bge-small embeddings, where unrelated questions
#: land near 0.6-0.75 and genuine paraphrases sit above 0.93. Lowering this
#: trades correctness for hit rate — the wrong direction for a tutor.
SIMILARITY_FLOOR = 0.95

#: Entries older than this are ignored and swept, even if still matching.
TTL_SECONDS = 7 * 24 * 3600

#: Never cache an answer whose grounding was not measured or was weak.
MIN_GROUNDING_TO_CACHE = 0.7

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    norm_query     TEXT    NOT NULL,
    scope          TEXT    NOT NULL,   -- domain set + model + index version
    ask_count      INTEGER NOT NULL DEFAULT 1,
    embedding      BLOB,               -- float32, only once promoted
    payload        TEXT,               -- JSON response, only once promoted
    index_version  TEXT,
    created_at     REAL    NOT NULL,
    last_hit_at    REAL,
    hit_count      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(norm_query, scope)
);
CREATE INDEX IF NOT EXISTS idx_cache_scope ON answer_cache(scope, created_at);
"""

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_question(text: str) -> str:
    """
    Canonical form used for the frequency counter.

    Deliberately aggressive — this only decides whether two asks count as the
    same *for counting*. Serving a cached answer additionally requires the
    semantic check, so an over-eager normalization here cannot by itself cause
    a wrong answer.
    """
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


@dataclass
class CacheHit:
    payload: dict[str, Any]
    similarity: float
    ask_count: int
    age_seconds: float


class AnswerCache:
    """
    SQLite-backed cache. Safe to share across threads.

    Args:
        path: Database file. Shares the state directory with the other caches.
        embedder: Anything with `embed_query(str) -> np.ndarray`, normalized.
        index_version: Active vector index name; entries from another version
            are never served.
    """

    def __init__(self, path: str, embedder, index_version: str = "") -> None:
        self.path = path
        self._embedder = embedder
        self._index_version = index_version or "unversioned"
        self._lock = threading.RLock()
        self._local = threading.local()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn().executescript(_SCHEMA)
            self._conn().commit()

    def _scope(self, domains: list[str] | None, model: str) -> str:
        """
        Cache key namespace.

        Domains and model are part of it because the same question answered
        for a different domain set, or by a different model, is a different
        answer. The index version makes a rebuild invalidate everything.
        """
        doms = ",".join(sorted(domains or []))
        return f"{doms}|{model}|{self._index_version}"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def lookup(self, question: str, *, domains: list[str] | None, model: str) -> CacheHit | None:
        """Return a cached answer for a semantically equivalent question, or None."""
        scope = self._scope(domains, model)
        now = time.time()

        with self._lock:
            rows = self._conn().execute(
                "SELECT id, norm_query, embedding, payload, ask_count, created_at "
                "FROM answer_cache "
                "WHERE scope=? AND payload IS NOT NULL AND created_at > ?",
                (scope, now - TTL_SECONDS),
            ).fetchall()

        if not rows:
            return None

        try:
            probe = np.asarray(self._embedder.embed_query(question), dtype=np.float32)
        except Exception:
            logger.warning("cache lookup could not embed the query", exc_info=True)
            return None

        best_row, best_sim = None, -1.0
        for row in rows:
            if not row["embedding"]:
                continue
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            if vec.shape != probe.shape:
                continue  # embedder changed; entry is unusable
            sim = float(np.dot(probe, vec))
            if sim > best_sim:
                best_row, best_sim = row, sim

        if best_row is None or best_sim < SIMILARITY_FLOOR:
            return None

        try:
            payload = json.loads(best_row["payload"])
        except (TypeError, ValueError):
            return None

        with self._lock:
            self._conn().execute(
                "UPDATE answer_cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
                (now, best_row["id"]),
            )
            self._conn().commit()

        logger.info("answer cache hit (similarity %.3f) for %r", best_sim, question[:60])
        return CacheHit(
            payload=payload,
            similarity=best_sim,
            ask_count=int(best_row["ask_count"]),
            age_seconds=now - float(best_row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_ask(self, question: str, *, domains: list[str] | None, model: str) -> int:
        """Count this ask. Returns the running total for the question."""
        norm = normalize_question(question)
        if not norm:
            return 0
        scope = self._scope(domains, model)
        now = time.time()

        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO answer_cache (norm_query, scope, ask_count, created_at) "
                "VALUES (?, ?, 1, ?) "
                "ON CONFLICT(norm_query, scope) DO UPDATE SET ask_count = ask_count + 1",
                (norm, scope, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT ask_count FROM answer_cache WHERE norm_query=? AND scope=?",
                (norm, scope),
            ).fetchone()
        return int(row["ask_count"]) if row else 0

    def should_store(self, ask_count: int, payload: dict[str, Any]) -> bool:
        """
        Whether this answer has earned a cache entry.

        Refusals and weakly-grounded answers are excluded: those are precisely
        the answers a corpus fix is supposed to improve, and freezing them for
        a week would hide the fix.
        """
        if ask_count < PROMOTE_AFTER:
            return False
        if payload.get("refused"):
            return False
        grounding = payload.get("grounding_score")
        if grounding is None or float(grounding) < MIN_GROUNDING_TO_CACHE:
            return False
        return bool(str(payload.get("final_answer", "")).strip())

    def store(
        self, question: str, payload: dict[str, Any], *, domains: list[str] | None, model: str
    ) -> bool:
        """Promote a counted question to a full cache entry. Returns True if stored."""
        norm = normalize_question(question)
        if not norm:
            return False
        scope = self._scope(domains, model)

        try:
            vec = np.asarray(self._embedder.embed_query(question), dtype=np.float32)
        except Exception:
            logger.warning("could not embed question for caching", exc_info=True)
            return False

        # Store the response only — never the diagnostics blob, which carries
        # chunk text and timings specific to the run that produced it.
        slim = {
            k: payload.get(k)
            for k in (
                "final_answer", "intent_type", "detected_domains", "is_course_related",
                "needs_clarification", "refused", "grounding_score", "guardrail_action",
                "sources",
            )
        }

        with self._lock:
            conn = self._conn()
            conn.execute(
                "UPDATE answer_cache SET embedding=?, payload=?, index_version=?, created_at=? "
                "WHERE norm_query=? AND scope=?",
                (vec.tobytes(), json.dumps(slim), self._index_version, time.time(), norm, scope),
            )
            conn.commit()
        logger.info("cached answer for %r (scope=%s)", question[:60], scope)
        return True

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def sweep(self) -> int:
        """Delete expired entries and anything from a superseded index. Returns rows removed."""
        cutoff = time.time() - TTL_SECONDS
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "DELETE FROM answer_cache "
                "WHERE created_at < ? "
                "   OR (payload IS NOT NULL AND index_version IS NOT ? )",
                (cutoff, self._index_version),
            )
            conn.commit()
            return cur.rowcount or 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn().execute(
                "SELECT COUNT(*) AS tracked, "
                "       SUM(payload IS NOT NULL) AS cached, "
                "       COALESCE(SUM(hit_count), 0) AS hits "
                "FROM answer_cache"
            ).fetchone()
        return {
            "tracked_questions": int(row["tracked"] or 0),
            "cached_answers": int(row["cached"] or 0),
            "total_hits": int(row["hits"] or 0),
            "promote_after": PROMOTE_AFTER,
            "similarity_floor": SIMILARITY_FLOOR,
            "ttl_days": round(TTL_SECONDS / 86400, 1),
            "index_version": self._index_version,
        }


__all__ = [
    "MIN_GROUNDING_TO_CACHE",
    "PROMOTE_AFTER",
    "SIMILARITY_FLOOR",
    "TTL_SECONDS",
    "AnswerCache",
    "CacheHit",
    "normalize_question",
]
