"""
Embedding layer
===============
Pluggable embedders with an explicit token budget, an asymmetric query
instruction, and a persistent cache.

Why this replaces the previous inline `SentenceTransformer("all-MiniLM-L6-v2")`:

  * **Truncation was silent.** MiniLM's window is 256 tokens; chunks were
    ~1,000. `sentence-transformers` truncates without warning, so 47.9% of the
    indexed corpus never reached the model. `TokenBudgetError` now makes that
    a loud failure instead of invisible data loss.
  * **BGE is asymmetric.** bge-*-en-v1.5 expects an instruction prefix on the
    *query* side only. Embedding queries and passages identically — as the old
    code did — leaves measurable retrieval quality on the table.
  * **No caching.** Every re-index re-encoded every chunk. Embeddings are
    keyed by (model, content hash), so re-ingesting an unchanged document is
    nearly free.

bge-small-en-v1.5 is 384-dimensional, matching MiniLM, so the existing
Pinecone index is reused without a dimension migration.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class TokenBudgetError(ValueError):
    """Raised when text exceeds the embedding model's window.

    Silent truncation is the failure this whole module exists to prevent, so
    over-long input is an error rather than something quietly clipped.
    """


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class EmbeddingModelSpec:
    """Static facts about an embedding model that callers need before loading it."""

    __slots__ = ("name", "dimension", "max_tokens", "query_prefix", "document_prefix")

    def __init__(
        self,
        name: str,
        dimension: int,
        max_tokens: int,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.name = name
        self.dimension = dimension
        self.max_tokens = max_tokens
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix


#: Models this project has validated. `query_prefix` follows each model card.
MODEL_REGISTRY: dict[str, EmbeddingModelSpec] = {
    "BAAI/bge-small-en-v1.5": EmbeddingModelSpec(
        name="BAAI/bge-small-en-v1.5",
        dimension=384,
        max_tokens=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "BAAI/bge-base-en-v1.5": EmbeddingModelSpec(
        name="BAAI/bge-base-en-v1.5",
        dimension=768,
        max_tokens=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    # Retained so an existing index can still be read during a migration.
    "all-MiniLM-L6-v2": EmbeddingModelSpec(
        name="all-MiniLM-L6-v2",
        dimension=384,
        max_tokens=256,
    ),
}


def get_model_spec(name: str) -> EmbeddingModelSpec:
    """Look up a model's spec, falling back to conservative defaults."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    logger.warning("Unregistered embedding model %r — assuming 384d/512tok", name)
    return EmbeddingModelSpec(name=name, dimension=384, max_tokens=512)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """
    SQLite-backed embedding cache keyed by (model, content hash).

    Vectors are stored as raw float32 bytes. Thread-local connections keep it
    usable from the API's worker pool.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        self._conn().executescript(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key   TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dim   INTEGER NOT NULL,
                vec   BLOB NOT NULL
            );
            """
        )
        self._conn().commit()

    @staticmethod
    def make_key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        out: dict[str, np.ndarray] = {}
        conn = self._conn()
        # Chunk the IN clause to stay under SQLite's variable limit.
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            for key, dim, blob in conn.execute(
                f"SELECT key, dim, vec FROM embeddings WHERE key IN ({placeholders})", batch
            ):
                out[key] = np.frombuffer(blob, dtype=np.float32, count=dim)
        return out

    def put_many(self, items: Iterable[tuple[str, str, np.ndarray]]) -> None:
        rows = [
            (key, model, int(vec.shape[0]), vec.astype(np.float32).tobytes())
            for key, model, vec in items
        ]
        if not rows:
            return
        conn = self._conn()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, model, dim, vec) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        except Exception:
            # Roll back so a failed cache write cannot leave this connection
            # holding a transaction and lock out every other reader.
            conn.rollback()
            raise

    def size(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]


# ---------------------------------------------------------------------------
# Embedder interface
# ---------------------------------------------------------------------------


class Embedder(ABC):
    """Interface every embedding backend implements."""

    spec: EmbeddingModelSpec

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    @property
    def model_name(self) -> str:
        return self.spec.name

    @property
    def max_tokens(self) -> int:
        return self.spec.max_tokens

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Exact token count under this model's tokenizer."""

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        token_counts: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Embed passages for indexing. Returns (n, dim) L2-normalized float32."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Returns (dim,) L2-normalized float32."""


class SentenceTransformerEmbedder(Embedder):
    """
    `sentence-transformers` backend with token-budget enforcement and caching.

    Args:
        model_name: HuggingFace model id.
        cache: Optional embedding cache. Documents are cached; queries are not
            (they are one-off and cheap).
        strict: When True, over-budget text raises `TokenBudgetError`. When
            False it is truncated with a warning — used only for reading an
            index built by an older, smaller-window model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        cache: EmbeddingCache | None = None,
        strict: bool = True,
        device: str | None = None,
    ) -> None:
        self.spec = get_model_spec(model_name)
        self.cache = cache
        self.strict = strict
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    # -- lazy model load ---------------------------------------------------

    def _get_model(self):
        """Load the model once, under a lock so concurrent requests can't race."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("loading embedding model %s", self.spec.name)
                    model = SentenceTransformer(self.spec.name, device=self._device)
                    # Make the model's own window match what we validate against.
                    model.max_seq_length = self.spec.max_tokens
                    self._model = model
        return self._model

    def count_tokens(self, text: str) -> int:
        tok = self._get_model().tokenizer
        return len(tok.encode(text, add_special_tokens=True, truncation=False))

    # -- budget enforcement ------------------------------------------------

    def _check_budget(
        self,
        texts: Sequence[str],
        label: str,
        counts: Sequence[int] | None = None,
    ) -> None:
        """
        Reject input that would be silently truncated.

        Checked before encoding so the failure names the offending text
        instead of surfacing later as unexplained retrieval misses.

        Args:
            counts: Token counts the caller already computed. The chunker
                measures every chunk to size it, so re-tokenizing here doubles
                the tokenization work for no new information.
        """
        offenders: list[tuple[int, int]] = []
        for i, t in enumerate(texts):
            n = counts[i] if counts is not None else self.count_tokens(t)
            if n > self.spec.max_tokens:
                offenders.append((i, n))

        if not offenders:
            return

        preview = ", ".join(f"#{i}={n}tok" for i, n in offenders[:5])
        message = (
            f"{len(offenders)}/{len(texts)} {label} exceed the {self.spec.max_tokens}-token "
            f"window of {self.spec.name} ({preview}). They would be silently truncated."
        )
        if self.strict:
            raise TokenBudgetError(message)
        logger.warning("%s — truncating (strict=False)", message)

    # -- encoding ----------------------------------------------------------

    def _encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        vecs = self._get_model().encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        token_counts: Sequence[int] | None = None,
    ) -> np.ndarray:
        """
        Embed passages for indexing.

        Args:
            token_counts: Precomputed token counts, one per text. Only trusted
                when this model has no document prefix — otherwise the prefix
                would make the caller's counts wrong, and a wrong count here
                reintroduces exactly the silent truncation this guard exists
                to prevent.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        prefixed = [self.spec.document_prefix + t for t in texts]
        usable = (
            token_counts
            if token_counts is not None
            and not self.spec.document_prefix
            and len(token_counts) == len(texts)
            else None
        )
        self._check_budget(prefixed, "documents", counts=usable)

        out = np.zeros((len(prefixed), self.dimension), dtype=np.float32)

        if self.cache is None:
            return self._encode(prefixed, batch_size)

        keys = [EmbeddingCache.make_key(self.spec.name, t) for t in prefixed]
        hits = self.cache.get_many(keys)
        misses = [i for i, k in enumerate(keys) if k not in hits]

        for i, k in enumerate(keys):
            if k in hits:
                out[i] = hits[k]

        if misses:
            fresh = self._encode([prefixed[i] for i in misses], batch_size)
            for slot, i in enumerate(misses):
                out[i] = fresh[slot]
            self.cache.put_many(
                (keys[i], self.spec.name, out[i]) for i in misses
            )

        logger.debug(
            "embedded %d documents (%d cached, %d computed)",
            len(texts), len(texts) - len(misses), len(misses),
        )
        return out

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a query, applying the model's asymmetric instruction prefix.

        bge-*-en-v1.5 is trained with an instruction on the query side only;
        omitting it measurably degrades retrieval. Queries are short, so the
        budget check is advisory rather than strict.
        """
        prefixed = self.spec.query_prefix + text
        if self.count_tokens(prefixed) > self.spec.max_tokens:
            logger.warning("query exceeds %d tokens; truncating", self.spec.max_tokens)
        return self._encode([prefixed], batch_size=1)[0]


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder(
    model_name: str | None = None,
    *,
    cache_path: str | Path | None = None,
    strict: bool = True,
) -> Embedder:
    """
    Return the shared embedder, constructing it on first use.

    Double-checked locking: the previous code initialized module globals with
    no synchronization while a 4-worker thread pool called into them, which
    could load the model several times concurrently.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is None:
            from edupilot.core.config import EMBEDDING_CACHE_PATH, EMBEDDING_MODEL

            name = model_name or EMBEDDING_MODEL
            path = cache_path or EMBEDDING_CACHE_PATH
            _embedder = SentenceTransformerEmbedder(
                name, cache=EmbeddingCache(path) if path else None, strict=strict
            )
    return _embedder


def reset_embedder() -> None:
    """Drop the singleton. Test-support only."""
    global _embedder
    with _embedder_lock:
        _embedder = None
