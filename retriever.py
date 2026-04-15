"""
EduPilot Retriever
==================
Isolated hybrid retrieval pipeline per domain.

Each DomainRetriever owns:
  - A Pinecone namespace      (semantic / vector search)
  - A BM25 index              (keyword search, rebuilt from SQLite on startup)

Vector storage  → Pinecone  (one index, one namespace per domain)
Text / metadata → SQLite     (document_chunks table via database.py)

Results from both sources are fused with Reciprocal Rank Fusion (RRF).

Design principle: NO domain shares state with another domain's retriever.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    BM25_WEIGHT,
    DOMAINS,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    SEMANTIC_WEIGHT,
)
from database import (
    chunk_count_by_domain,
    delete_chunks_by_domain,
    get_chunk_ids_by_domain,
    get_chunks_by_domain,
    save_chunks,
)
from utils import (
    DocumentChunk,
    RetrievedChunk,
    load_domain_documents,
    tokenize_simple,
)


# ---------------------------------------------------------------------------
# Embedding model — loaded once, shared across all retrievers
# ---------------------------------------------------------------------------
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


# ---------------------------------------------------------------------------
# Pinecone client — one shared client, one index
# ---------------------------------------------------------------------------
_pinecone_index = None


def _get_pinecone_index():
    """Return the shared Pinecone index (created lazily, once per process)."""
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    from pinecone import Pinecone, ServerlessSpec  # type: ignore

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait until the index is ready
        for _ in range(30):
            desc = pc.describe_index(PINECONE_INDEX_NAME)
            if desc.status.get("ready", False):
                break
            time.sleep(2)

    _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index


# ---------------------------------------------------------------------------
# Domain Retriever
# ---------------------------------------------------------------------------

class DomainRetriever:
    """
    Self-contained RAG retrieval pipeline for one domain.
    Call initialize() before use, then retrieve() for hybrid search.
    """

    def __init__(self, domain: str):
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain: {domain!r}. Must be one of {list(DOMAINS)}")

        self.domain = domain
        self.domain_cfg = DOMAINS[domain]
        self.kb_path = self.domain_cfg["knowledge_base_path"]
        self.namespace = self.domain_cfg["pinecone_namespace"]

        # BM25 parallel arrays (kept in sync with SQLite)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_texts: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metas: list[dict] = []

        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set up Pinecone namespace + BM25. Auto-indexes KB docs if needed."""
        # If no chunks in SQLite yet, index the knowledge base
        if chunk_count_by_domain(self.domain) == 0:
            self._index_knowledge_base()
        else:
            # Chunks already exist in SQLite — just rebuild BM25 from DB
            self._rebuild_bm25_from_db()

        self._initialized = True

    def _index_knowledge_base(self) -> int:
        """Load all documents from the domain KB folder and index them."""
        chunks = load_domain_documents(self.kb_path, self.domain)
        if not chunks:
            return 0
        self._add_chunks(chunks)
        return len(chunks)

    # ------------------------------------------------------------------
    # Document addition
    # ------------------------------------------------------------------

    def add_documents(self, file_paths: list[str]) -> int:
        """Index one or more new document files into this domain's pipeline."""
        from utils import load_and_chunk_file
        all_chunks: list[DocumentChunk] = []
        for fp in file_paths:
            all_chunks.extend(load_and_chunk_file(fp, self.domain))
        if all_chunks:
            self._add_chunks(all_chunks)
        return len(all_chunks)

    def _add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed chunks, upsert to Pinecone, and persist metadata to SQLite."""
        if not chunks:
            return

        # De-duplicate against existing chunk IDs in SQLite
        existing_ids = get_chunk_ids_by_domain(self.domain)
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            return

        model = get_embedding_model()
        new_texts = [c.text for c in new_chunks]
        embeddings = model.encode(
            new_texts, normalize_embeddings=True, show_progress_bar=False
        )

        index = _get_pinecone_index()

        # Upsert to Pinecone in batches of 100 (Pinecone recommended batch size)
        batch_size = 100
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            batch_emb = embeddings[i : i + batch_size]

            vectors = [
                {
                    "id": c.chunk_id,
                    "values": batch_emb[j].tolist(),
                    "metadata": {
                        "text": c.text,
                        "source_file": c.source_file,
                        "domain": c.domain,
                        "page_number": c.page_number or 0,
                        "chunk_index": c.metadata.get("chunk_index", 0),
                    },
                }
                for j, c in enumerate(batch)
            ]
            index.upsert(vectors=vectors, namespace=self.namespace)

        # Persist to SQLite (source of truth for BM25 and dedup)
        save_chunks([
            {
                "chunk_id": c.chunk_id,
                "domain": c.domain,
                "text": c.text,
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.metadata.get("chunk_index", 0),
            }
            for c in new_chunks
        ])

        # Rebuild BM25 to include new chunks
        self._rebuild_bm25_from_db()

    def _rebuild_bm25_from_db(self) -> None:
        """Sync BM25 index from SQLite document_chunks table."""
        rows = get_chunks_by_domain(self.domain)
        if not rows:
            self._bm25 = None
            self._bm25_texts = []
            self._bm25_ids = []
            self._bm25_metas = []
            return

        self._bm25_texts = [r["text"] for r in rows]
        self._bm25_ids = [r["chunk_id"] for r in rows]
        self._bm25_metas = [
            {
                "source_file": r["source_file"],
                "domain": self.domain,
                "page_number": str(r.get("page_number") or ""),
                "chunk_index": str(r.get("chunk_index") or 0),
            }
            for r in rows
        ]
        tokenized = [tokenize_simple(t) for t in self._bm25_texts]
        self._bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5, source_filter: list[str] | None = None) -> list[RetrievedChunk]:
        """
        Hybrid retrieval: Pinecone semantic search + BM25 keyword search,
        fused with Reciprocal Rank Fusion (RRF).

        If *source_filter* is provided (list of filenames), only chunks whose
        source_file path contains one of those filenames are returned.
        """
        if not self._initialized:
            raise RuntimeError("DomainRetriever not initialized. Call initialize() first.")

        if chunk_count_by_domain(self.domain) == 0:
            return []

        n_candidates = min(top_k * 3, chunk_count_by_domain(self.domain))

        # Fetch more candidates when filtering to ensure enough results
        fetch_k = n_candidates * 3 if source_filter else n_candidates
        fetch_k = min(fetch_k, chunk_count_by_domain(self.domain))

        sem_results = self._semantic_search(query, fetch_k)
        bm25_results = self._bm25_search(query, fetch_k)

        # Filter by source file if requested
        if source_filter:
            def _matches(meta: dict) -> bool:
                sf = meta.get("source_file", "")
                return any(fn in sf for fn in source_filter)

            sem_filtered = [r for r in sem_results if _matches(r["meta"])]
            bm25_filtered = [r for r in bm25_results if _matches(r["meta"])]

            # Fall back to unfiltered if no matches (filename mismatch)
            if sem_filtered or bm25_filtered:
                sem_results = sem_filtered
                bm25_results = bm25_filtered

        return self._reciprocal_rank_fusion(sem_results, bm25_results, top_k)

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
        """Query Pinecone for the top-k most similar chunks."""
        model = get_embedding_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

        index = _get_pinecone_index()
        results = index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        items = []
        for match in results.matches:
            meta = match.metadata or {}
            items.append({
                "id": match.id,
                "text": meta.get("text", ""),
                "meta": {
                    "source_file": meta.get("source_file", "unknown"),
                    "domain": meta.get("domain", self.domain),
                    "page_number": str(meta.get("page_number", "")),
                    "chunk_index": str(meta.get("chunk_index", 0)),
                },
                "score": float(match.score),
            })
        return items

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Return top-k results from the in-memory BM25 index."""
        if not self._bm25 or not self._bm25_texts:
            return []

        tokenized_query = tokenize_simple(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_n = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_n]

        items = []
        for i in top_indices:
            if scores[i] > 0:
                items.append({
                    "id": self._bm25_ids[i],
                    "text": self._bm25_texts[i],
                    "meta": self._bm25_metas[i],
                    "score": float(scores[i]),
                })
        return items

    def _reciprocal_rank_fusion(
        self,
        sem_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Combine semantic and BM25 rankings using Reciprocal Rank Fusion.
        score[id] += weight / (k + rank)
        """
        rrf_scores: dict[str, float] = {}
        sem_score_map: dict[str, float] = {}
        bm25_score_map: dict[str, float] = {}
        meta_map: dict[str, dict] = {}
        text_map: dict[str, str] = {}

        for rank, item in enumerate(sem_results, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + SEMANTIC_WEIGHT / (k + rank)
            sem_score_map[cid] = item["score"]
            meta_map[cid] = item["meta"]
            text_map[cid] = item["text"]

        max_bm25 = max((i["score"] for i in bm25_results), default=1.0) or 1.0
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + BM25_WEIGHT / (k + rank)
            bm25_score_map[cid] = item["score"] / max_bm25
            if cid not in meta_map:
                meta_map[cid] = item["meta"]
                text_map[cid] = item["text"]

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        chunks: list[RetrievedChunk] = []
        for cid in sorted_ids[:top_k]:
            meta = meta_map[cid]
            page_num = meta.get("page_number")
            chunks.append(RetrievedChunk(
                chunk_id=cid,
                text=text_map[cid],
                source_file=meta.get("source_file", "unknown"),
                domain=self.domain,
                page_number=int(page_num) if page_num else None,
                semantic_score=sem_score_map.get(cid, 0.0),
                bm25_score=bm25_score_map.get(cid, 0.0),
                rerank_score=rrf_scores[cid],
                metadata=meta,
            ))
        return chunks

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def document_count(self) -> int:
        """Number of chunks tracked in SQLite for this domain."""
        return chunk_count_by_domain(self.domain)

    def is_empty(self) -> bool:
        return self.document_count() == 0

    def reset(self) -> None:
        """Wipe Pinecone namespace and SQLite chunks (for testing / re-indexing)."""
        index = _get_pinecone_index()
        index.delete(delete_all=True, namespace=self.namespace)
        delete_chunks_by_domain(self.domain)
        self._bm25 = None
        self._bm25_texts = []
        self._bm25_ids = []
        self._bm25_metas = []


# ---------------------------------------------------------------------------
# Registry: one retriever per domain
# ---------------------------------------------------------------------------

_retrievers: dict[str, DomainRetriever] = {}


def get_retriever(domain: str) -> DomainRetriever:
    """Return the (lazily initialized) DomainRetriever for a given domain."""
    if domain not in _retrievers:
        r = DomainRetriever(domain)
        r.initialize()
        _retrievers[domain] = r
    return _retrievers[domain]


def initialize_all_retrievers() -> dict[str, int]:
    """Initialize retrievers for all domains and return chunk counts."""
    counts = {}
    for domain in DOMAINS:
        r = get_retriever(domain)
        counts[domain] = r.document_count()
    return counts
