"""
EduPilot — Self Study Retriever
================================
Session-scoped hybrid retrieval pipeline.

Each SelfStudyRetriever owns:
  - A Pinecone namespace  : ss_{uuid_hex}  (semantic / vector search)
  - A BM25 index          : rebuilt from self_study_chunks on demand

Vector storage  → Pinecone  (one namespace per study session)
Text / metadata → SQLite     (self_study_chunks table)

Isolation from domain retrievers is total — separate namespace, separate SQLite table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

import database as db
from retriever import _get_pinecone_index, get_embedding_model
from utils import RetrievedChunk, tokenize_simple

# Weight constants (same defaults as domain retriever)
_SEMANTIC_WEIGHT = 0.60
_BM25_WEIGHT = 0.40


class SelfStudyRetriever:
    """
    Hybrid RAG retrieval for a single self-study session.
    Call initialize() once after creation, then use retrieve() and add_documents().
    """

    def __init__(self, ss_session_id: str):
        self.ss_session_id = ss_session_id
        # Pinecone namespace: ss_ + uuid without dashes (32 hex chars → 35 chars total)
        self.namespace = f"ss_{ss_session_id.replace('-', '')}"

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_texts: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_metas: list[dict] = []
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self._rebuild_bm25_from_db()
        self._initialized = True

    # ------------------------------------------------------------------
    # Document addition
    # ------------------------------------------------------------------

    def add_documents(self, file_paths: list[str]) -> list[dict]:
        """
        Chunk, embed, and index one or more files into this session's namespace.
        Returns list of {filename, chunks_indexed}.
        """
        from utils import load_and_chunk_file

        results = []
        for fp in file_paths:
            chunks = load_and_chunk_file(fp, "SELF_STUDY")
            if not chunks:
                results.append({"filename": Path(fp).name, "chunks_indexed": 0})
                continue
            n = self._add_chunks(chunks, fp)
            results.append({"filename": Path(fp).name, "chunks_indexed": n})
        return results

    def _add_chunks(self, chunks, source_path: str) -> int:
        if not chunks:
            return 0

        existing_ids = db.get_ss_chunk_ids(self.ss_session_id)
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            return 0

        model = get_embedding_model()
        texts = [c.text for c in new_chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        index = _get_pinecone_index()
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
                        "domain": "SELF_STUDY",
                        "ss_session_id": self.ss_session_id,
                        "page_number": c.page_number or 0,
                        "chunk_index": c.metadata.get("chunk_index", 0),
                    },
                }
                for j, c in enumerate(batch)
            ]
            index.upsert(vectors=vectors, namespace=self.namespace)

        db.save_ss_chunks([
            {
                "chunk_id": c.chunk_id,
                "ss_session_id": self.ss_session_id,
                "text": c.text,
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_index": c.metadata.get("chunk_index", 0),
            }
            for c in new_chunks
        ])

        self._rebuild_bm25_from_db()
        return len(new_chunks)

    # ------------------------------------------------------------------
    # Document removal
    # ------------------------------------------------------------------

    def remove_document(self, filename: str) -> int:
        """Delete all chunks for a given filename from Pinecone and SQLite."""
        chunk_ids = db.get_ss_chunk_ids_by_source(self.ss_session_id, filename)
        if not chunk_ids:
            return 0

        index = _get_pinecone_index()
        batch_size = 100
        for i in range(0, len(chunk_ids), batch_size):
            index.delete(ids=chunk_ids[i : i + batch_size], namespace=self.namespace)

        deleted = db.delete_ss_chunks_by_source(self.ss_session_id, filename)
        self._rebuild_bm25_from_db()
        return deleted

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._initialized:
            raise RuntimeError("SelfStudyRetriever not initialized. Call initialize() first.")

        total = db.ss_chunk_count(self.ss_session_id)
        if total == 0:
            return []

        n_candidates = min(top_k * 3, total)
        # Fetch more candidates when filtering so we have enough after the filter
        fetch_k = min(n_candidates * 3 if source_filter else n_candidates, total)

        sem_results = self._semantic_search(query, fetch_k)
        bm25_results = self._bm25_search(query, fetch_k)

        if source_filter:
            def _matches(meta: dict) -> bool:
                sf = meta.get("source_file", "")
                return any(fn in sf for fn in source_filter)

            # Strict filtering — no fallback. The user explicitly chose these docs.
            sem_results  = [r for r in sem_results  if _matches(r["meta"])]
            bm25_results = [r for r in bm25_results if _matches(r["meta"])]

        return self._reciprocal_rank_fusion(sem_results, bm25_results, top_k)

    def _semantic_search(self, query: str, top_k: int) -> list[dict]:
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
                    "domain": "SELF_STUDY",
                    "page_number": str(meta.get("page_number", "")),
                    "chunk_index": str(meta.get("chunk_index", 0)),
                },
                "score": float(match.score),
            })
        return items

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
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
        rrf_scores: dict[str, float] = {}
        sem_score_map: dict[str, float] = {}
        bm25_score_map: dict[str, float] = {}
        meta_map: dict[str, dict] = {}
        text_map: dict[str, str] = {}

        for rank, item in enumerate(sem_results, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _SEMANTIC_WEIGHT / (k + rank)
            sem_score_map[cid] = item["score"]
            meta_map[cid] = item["meta"]
            text_map[cid] = item["text"]

        max_bm25 = max((i["score"] for i in bm25_results), default=1.0) or 1.0
        for rank, item in enumerate(bm25_results, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _BM25_WEIGHT / (k + rank)
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
                domain="SELF_STUDY",
                page_number=int(page_num) if page_num else None,
                semantic_score=sem_score_map.get(cid, 0.0),
                bm25_score=bm25_score_map.get(cid, 0.0),
                rerank_score=rrf_scores[cid],
                metadata=meta,
            ))
        return chunks

    # ------------------------------------------------------------------
    # BM25 rebuild
    # ------------------------------------------------------------------

    def _rebuild_bm25_from_db(self) -> None:
        rows = db.get_ss_chunks(self.ss_session_id)
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
                "domain": "SELF_STUDY",
                "page_number": str(r.get("page_number") or ""),
                "chunk_index": str(r.get("chunk_index") or 0),
            }
            for r in rows
        ]
        tokenized = [tokenize_simple(t) for t in self._bm25_texts]
        self._bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Wipe Pinecone namespace and all SQLite data for this session."""
        try:
            index = _get_pinecone_index()
            index.delete(delete_all=True, namespace=self.namespace)
        except Exception:
            pass
        self._bm25 = None
        self._bm25_texts = []
        self._bm25_ids = []
        self._bm25_metas = []

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def chunk_count(self) -> int:
        return db.ss_chunk_count(self.ss_session_id)

    def is_empty(self) -> bool:
        return self.chunk_count() == 0


# ---------------------------------------------------------------------------
# Registry — lazy per-session retrievers
# ---------------------------------------------------------------------------

_ss_retrievers: dict[str, SelfStudyRetriever] = {}


def get_ss_retriever(ss_session_id: str) -> SelfStudyRetriever:
    """Return the (lazily initialized) SelfStudyRetriever for a given session."""
    if ss_session_id not in _ss_retrievers:
        r = SelfStudyRetriever(ss_session_id)
        r.initialize()
        _ss_retrievers[ss_session_id] = r
    return _ss_retrievers[ss_session_id]


def evict_ss_retriever(ss_session_id: str) -> None:
    """Remove a retriever from the registry (called on session delete)."""
    _ss_retrievers.pop(ss_session_id, None)
