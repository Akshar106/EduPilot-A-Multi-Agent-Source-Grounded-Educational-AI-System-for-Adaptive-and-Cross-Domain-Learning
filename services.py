"""
Composition root
================
Builds and holds the singletons: embedder, vector store, sparse encoder,
retrievers, indexer, pipeline, and user store.

Kept separate from `main.py` so the HTTP layer contains routing and nothing
else, and so tests can assemble the same object graph against
`InMemoryVectorStore` without starting a server.

Every singleton is created under a lock. The previous code initialized module
globals (`_pipeline`, `_retrievers`, `_embedding_model`, `_pinecone_index`)
with no synchronization while a four-worker thread pool called into them, so
two concurrent first-requests could each start loading the same model.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from config import (
    BASE_DIR,
    DOMAINS,
    EMBEDDING_CACHE_PATH,
    EMBEDDING_MODEL,
    INDEX_POINTER_PATH,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_BASE,
    PINECONE_METRIC,
    PINECONE_REGION,
    SPARSE_ENCODER_PATH,
    SQLITE_DB_PATH,
    STATE_DIR,
)
from ingestion.chunking import ChunkingConfig
from retrieval import (
    BM25Encoder,
    Embedder,
    EmbeddingCache,
    HybridRetriever,
    IndexPointer,
    IndexRegistry,
    Indexer,
    InMemoryVectorStore,
    PineconeVectorStore,
    SentenceTransformerEmbedder,
    VectorStore,
)
from security import UserStore

logger = logging.getLogger(__name__)


class Services:
    """
    Lazily-constructed application object graph.

    Nothing is built at import time — model loads and network calls happen on
    first use, so `import services` stays cheap and tests can substitute
    components before anything is instantiated.
    """

    def __init__(self, *, use_in_memory_store: bool = False) -> None:
        self._use_in_memory = use_in_memory_store
        # RLock, not Lock. The properties compose: `store` holds the lock and
        # then reads `embedder.dimension`, `indexer` reads `store`, `embedder`,
        # `registry` and `sparse_encoder`, and `_retriever_for` reads three of
        # them. With a plain Lock the first such access deadlocks the process
        # on itself, because a thread cannot re-acquire a non-reentrant lock it
        # already holds.
        self._lock = threading.RLock()

        self._embedder: Embedder | None = None
        self._store: VectorStore | None = None
        self._sparse: BM25Encoder | None = None
        self._registry: IndexRegistry | None = None
        self._indexer: Indexer | None = None
        self._users: UserStore | None = None
        self._pipeline = None
        self._retrievers: dict[str, HybridRetriever] = {}
        self._sparse_loaded = False

        STATE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core components
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    self._embedder = SentenceTransformerEmbedder(
                        EMBEDDING_MODEL,
                        cache=EmbeddingCache(EMBEDDING_CACHE_PATH),
                        strict=True,
                    )
        return self._embedder

    @property
    def index_pointer(self) -> IndexPointer:
        return IndexPointer(INDEX_POINTER_PATH, PINECONE_INDEX_BASE)

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self._store = self._build_store()
        return self._store

    def _build_store(self, index_name: str | None = None) -> VectorStore:
        if self._use_in_memory:
            logger.warning("using in-memory vector store — data is not persisted")
            return InMemoryVectorStore()
        return PineconeVectorStore(
            api_key=PINECONE_API_KEY,
            index_name=index_name,
            pointer=None if index_name else self.index_pointer,
            dimension=self.embedder.dimension,
            metric=PINECONE_METRIC,
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        )

    def store_for_index(self, index_name: str) -> VectorStore:
        """A store bound to a specific index version. Used during a rebuild."""
        return self._build_store(index_name)

    @property
    def sparse_encoder(self) -> BM25Encoder | None:
        """
        The fitted BM25 encoder, or None when no index has been built.

        Absence is not an error: without it, hybrid search degrades to
        dense-only, which still works. It is logged once rather than on every
        query.
        """
        if not self._sparse_loaded:
            with self._lock:
                if not self._sparse_loaded:
                    self._sparse = BM25Encoder.load(SPARSE_ENCODER_PATH)
                    self._sparse_loaded = True
                    if self._sparse is None:
                        logger.warning(
                            "no fitted BM25 encoder at %s — lexical matching is disabled "
                            "until you run `python reindex.py`",
                            SPARSE_ENCODER_PATH,
                        )
        return self._sparse

    def reload_sparse_encoder(self) -> None:
        """Re-read the encoder after a rebuild, without a restart."""
        with self._lock:
            self._sparse = BM25Encoder.load(SPARSE_ENCODER_PATH)
            self._sparse_loaded = True
            self._retrievers.clear()

    @property
    def registry(self) -> IndexRegistry:
        if self._registry is None:
            with self._lock:
                if self._registry is None:
                    self._registry = IndexRegistry(SQLITE_DB_PATH)
        return self._registry

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(
            model_name=self.embedder.model_name,
            model_window=self.embedder.max_tokens,
            max_tokens=min(448, self.embedder.max_tokens - 64),
        )

    @property
    def indexer(self) -> Indexer:
        if self._indexer is None:
            with self._lock:
                if self._indexer is None:
                    self._indexer = Indexer(
                        self.store,
                        self.embedder,
                        self.registry,
                        sparse_encoder=self.sparse_encoder,
                        chunking=self.chunking,
                    )
        return self._indexer

    @property
    def users(self) -> UserStore:
        if self._users is None:
            with self._lock:
                if self._users is None:
                    self._users = UserStore(SQLITE_DB_PATH)
        return self._users

    # ------------------------------------------------------------------
    # Retrievers
    # ------------------------------------------------------------------

    def retriever(self, domain: str) -> HybridRetriever:
        """Retriever for a course domain, cached per namespace."""
        namespace = DOMAINS[domain]["pinecone_namespace"]
        return self._retriever_for(namespace)

    def study_retriever(self, ss_session_id: str) -> HybridRetriever:
        """
        Retriever for one study session's uploads.

        The namespace derives from the session id, so one student's uploads
        are physically separated from another's rather than filtered apart at
        query time.
        """
        return self._retriever_for(f"ss_{ss_session_id.replace('-', '')}")

    def _retriever_for(self, namespace: str) -> HybridRetriever:
        cached = self._retrievers.get(namespace)
        if cached is not None:
            return cached
        with self._lock:
            if namespace not in self._retrievers:
                self._retrievers[namespace] = HybridRetriever(
                    self.store,
                    self.embedder,
                    namespace,
                    sparse_encoder=self.sparse_encoder,
                )
            return self._retrievers[namespace]

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    @property
    def pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    from agents import EduPilotPipeline
                    from llm import call_llm

                    self._pipeline = EduPilotPipeline(
                        llm=call_llm,
                        domains=DOMAINS,
                        retriever_factory=self.retriever,
                    )
        return self._pipeline

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """
        Report readiness of every dependency this app actually uses.

        The previous check validated GEMINI_API_KEY and PINECONE_API_KEY but
        not GROQ_API_KEY — and the default model is Groq, so it reported "ok"
        on a deployment where every chat request would fail.
        """
        import os

        from config import AVAILABLE_MODELS, DEFAULT_MODEL, GROQ_MODELS

        checks: dict[str, dict] = {}

        needs_groq = DEFAULT_MODEL in GROQ_MODELS or any(m in GROQ_MODELS for m in AVAILABLE_MODELS)
        needs_gemini = any(m.startswith("gemini") for m in AVAILABLE_MODELS)

        checks["groq_api_key"] = {
            "required": needs_groq,
            "present": bool(os.getenv("GROQ_API_KEY")),
        }
        checks["gemini_api_key"] = {
            "required": needs_gemini,
            "present": bool(os.getenv("GEMINI_API_KEY")),
        }
        checks["pinecone_api_key"] = {
            "required": not self._use_in_memory,
            "present": bool(PINECONE_API_KEY),
        }
        checks["jwt_secret"] = {
            "required": True,
            "present": bool(os.getenv("JWT_SECRET_KEY")),
        }
        checks["sparse_encoder"] = {
            "required": False,
            "present": Path(SPARSE_ENCODER_PATH).exists(),
        }
        checks["vector_index"] = {
            "required": True,
            "present": bool(self.index_pointer.active) or self._use_in_memory,
        }

        # A missing required dependency is a hard failure, not a warning.
        failing = [k for k, v in checks.items() if v["required"] and not v["present"]]
        return {
            "status": "ok" if not failing else "degraded",
            "failing": failing,
            "checks": checks,
            "domains": list(DOMAINS),
            "embedding_model": EMBEDDING_MODEL,
            "active_index": self.index_pointer.active,
        }


#: Application-wide instance.
services = Services()
