"""
Hybrid retriever
================
Search across one namespace: query transformation, engine-side sparse+dense
hybrid search, rank fusion across probes, cross-encoder reranking, and
small-to-big parent expansion.

Differences from the previous `DomainRetriever`:

  * **Filters are pushed down.** `source_filter` becomes a Pinecone metadata
    filter instead of a Python list comprehension over over-fetched results.
    The old code also fell back to *unfiltered* results when a filter matched
    nothing, so restricting a search to one attachment silently searched the
    whole corpus. A filter that matches nothing now returns nothing.
  * **Lexical search is engine-side.** BM25 term weights are stored as sparse
    vectors, so one query does both halves of the hybrid. No in-process index
    to rebuild, and it works with multiple workers.
  * **Reranking is on.** The Chat path reranked by token overlap; the
    cross-encoder now runs on every path.
  * **Empty means empty.** When nothing clears the relevance floor the
    retriever returns nothing instead of substituting the best of a bad set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from .embeddings import Embedder
from .query_transform import LLMCallable, TransformedQuery, transform_query
from .rerank import RerankedChunk, RerankMode, rerank, score_summary
from .sparse import BM25Encoder
from .vectorstore import SearchHit, VectorStore

logger = logging.getLogger(__name__)

#: RRF damping constant. 60 is the value from the original Cormack et al. paper.
RRF_K = 60

#: Relative trust in each probe type during fusion.
PROBE_WEIGHTS = {"primary": 1.0, "variant": 0.7, "hyde": 0.7}


@dataclass
class RetrievalConfig:
    """Knobs for one retrieval call."""

    top_k: int = 6
    """Chunks handed to the generator after reranking."""
    candidate_multiplier: int = 5
    """Candidates fetched per probe, as a multiple of top_k."""
    rerank_mode: RerankMode = "cross_encoder"
    min_relevance: float | None = None
    """
    Absolute relevance floor. None uses the reranker's calibrated default —
    score scales differ by model, so a hardcoded number is wrong for all but
    one of them. See retrieval/rerank.py.
    """
    relative_floor: float = 0.01
    """Drop candidates scoring below this fraction of the best candidate."""
    use_hybrid: bool = True
    """Attach sparse BM25 weights to the query for lexical matching."""
    use_multi_query: bool = False
    use_hyde: bool = False
    expand_to_parents: bool = True
    """Swap each matched child for its wider parent window before generating."""
    max_candidates: int = 120
    """Hard cap on candidates sent to the cross-encoder, for latency."""


@dataclass
class RetrievalResult:
    """Retrieved evidence plus the diagnostics the debug panel renders."""

    chunks: list[RerankedChunk] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            name = str(c.metadata.get("filename", ""))
            if name and name not in seen:
                seen.append(name)
        return seen


class HybridRetriever:
    """
    Retrieval over a single vector-store namespace.

    Args:
        store: Vector store backend.
        embedder: Dense embedder.
        namespace: Namespace to search (one per domain, or per study session).
        sparse_encoder: Fitted BM25 encoder. None disables the lexical half.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        namespace: str,
        *,
        sparse_encoder: BM25Encoder | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.namespace = namespace
        self.sparse_encoder = sparse_encoder

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    @staticmethod
    def build_filter(
        *,
        filenames: Sequence[str] | None = None,
        exclude_parents: bool = True,
        min_confidence: float | None = None,
        extra: dict | None = None,
    ) -> dict | None:
        """
        Build a Pinecone metadata filter.

        `exclude_parents` keeps parent windows out of search results: they are
        stored for context expansion but must never be matched directly, or a
        single parent would crowd out the children it contains.
        """
        clauses: list[dict] = []
        if filenames:
            clauses.append({"filename": {"$in": list(filenames)}})
        if exclude_parents:
            clauses.append({"is_parent": {"$ne": True}})
        if min_confidence is not None:
            clauses.append({"confidence": {"$gte": min_confidence}})
        if extra:
            clauses.append(extra)

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_one(
        self,
        probe: str,
        top_k: int,
        metadata_filter: dict | None,
        use_hybrid: bool,
    ) -> list[SearchHit]:
        """Run one probe as a single sparse+dense query."""
        dense = self.embedder.embed_query(probe)
        sparse = None
        if use_hybrid and self.sparse_encoder and self.sparse_encoder.fitted:
            sparse = self.sparse_encoder.encode_query(probe) or None

        return self.store.query(
            vector=dense.tolist(),
            top_k=top_k,
            namespace=self.namespace,
            metadata_filter=metadata_filter,
            sparse=sparse,
        )

    @staticmethod
    def _fuse(ranked_lists: list[tuple[str, list[SearchHit]]]) -> dict[str, tuple[float, SearchHit]]:
        """
        Reciprocal Rank Fusion across probes.

        RRF combines rankings without needing their scores to be on a common
        scale, which matters here because a HyDE probe and the literal query
        produce systematically different similarity magnitudes.
        """
        fused: dict[str, float] = {}
        best: dict[str, SearchHit] = {}

        for kind, hits in ranked_lists:
            weight = PROBE_WEIGHTS.get(kind, 0.7)
            for rank, hit in enumerate(hits, start=1):
                fused[hit.id] = fused.get(hit.id, 0.0) + weight / (RRF_K + rank)
                if hit.id not in best or hit.score > best[hit.id].score:
                    best[hit.id] = hit

        return {cid: (score, best[cid]) for cid, score in fused.items()}

    def retrieve(
        self,
        query: str,
        *,
        config: RetrievalConfig | None = None,
        filenames: Sequence[str] | None = None,
        metadata_filter: dict | None = None,
        llm: LLMCallable | None = None,
    ) -> RetrievalResult:
        """
        Retrieve evidence for a query.

        Returns a `RetrievalResult` whose `chunks` may legitimately be empty —
        that is the signal that the knowledge base does not cover the question,
        and the caller must refuse rather than answer unsupported.
        """
        cfg = config or RetrievalConfig()
        diagnostics: dict[str, Any] = {"namespace": self.namespace, "query": query}

        transformed: TransformedQuery = transform_query(
            query,
            llm=llm,
            use_multi_query=cfg.use_multi_query,
            use_hyde=cfg.use_hyde,
        )
        diagnostics["probes"] = transformed.probes
        diagnostics["acronyms_expanded"] = transformed.expanded != transformed.original

        combined_filter = self.build_filter(filenames=filenames, extra=metadata_filter)
        diagnostics["filter"] = combined_filter

        per_probe = min(cfg.top_k * cfg.candidate_multiplier, cfg.max_candidates)
        ranked_lists: list[tuple[str, list[SearchHit]]] = []

        for i, probe in enumerate(transformed.probes):
            if i == 0:
                kind = "primary"
            elif transformed.hyde and probe == transformed.hyde:
                kind = "hyde"
            else:
                kind = "variant"
            try:
                hits = self._search_one(probe, per_probe, combined_filter, cfg.use_hybrid)
            except Exception as exc:
                logger.warning("probe %r failed: %s", probe[:60], exc)
                continue
            ranked_lists.append((kind, hits))

        if not ranked_lists:
            diagnostics["error"] = "all probes failed"
            return RetrievalResult(chunks=[], diagnostics=diagnostics)

        fused = self._fuse(ranked_lists)
        diagnostics["candidates"] = len(fused)

        if not fused:
            diagnostics["reason"] = "no vectors matched the filter"
            return RetrievalResult(chunks=[], diagnostics=diagnostics)

        # Rerank the best candidates by fused rank.
        ordered = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)[: cfg.max_candidates]
        candidates = [
            (cid, hit.text, hit.metadata, score) for cid, (score, hit) in ordered if hit.text
        ]

        # When expanding to parents, several top children often share one
        # parent and collapse into a single context unit. Rerank a wider pool
        # so the caller still receives `top_k` *distinct* units afterwards.
        rerank_k = cfg.top_k * 3 if cfg.expand_to_parents else cfg.top_k

        reranked = rerank(
            # Rerank against the student's actual words, not the acronym-expanded
            # probe: expansion helps recall at the search stage but adds terms
            # the student never wrote, which skews cross-encoder relevance.
            query=query,
            candidates=candidates,
            top_k=rerank_k,
            mode=cfg.rerank_mode,
            min_relevance=cfg.min_relevance,
            relative_floor=cfg.relative_floor,
        )

        if not reranked:
            diagnostics["reason"] = "no candidate cleared the relevance floor"
            diagnostics["reranked"] = {"count": 0}
            return RetrievalResult(chunks=[], diagnostics=diagnostics)

        if cfg.expand_to_parents:
            expanded = self._expand_to_parents(reranked)
            diagnostics["children_matched"] = len(reranked)
            diagnostics["parents_after_dedup"] = len(expanded)
            reranked = expanded

        reranked = reranked[: cfg.top_k]
        diagnostics["reranked"] = score_summary(reranked)
        return RetrievalResult(chunks=reranked, diagnostics=diagnostics)

    # ------------------------------------------------------------------
    # Small-to-big
    # ------------------------------------------------------------------

    def _expand_to_parents(self, chunks: list[RerankedChunk]) -> list[RerankedChunk]:
        """
        Replace matched children with their parent windows.

        Retrieval matches a precise child; generation reads better with the
        surrounding context. Where several children share a parent, the parent
        is emitted once at the best child's relevance, which also removes the
        near-duplicate context the old pipeline fed the model.
        """
        parent_ids = [
            str(c.metadata["parent_id"]) for c in chunks if c.metadata.get("parent_id")
        ]
        if not parent_ids:
            return chunks

        try:
            parents = self.store.fetch(parent_ids, self.namespace)
        except Exception as exc:
            logger.warning("parent expansion failed, using children: %s", exc)
            return chunks

        out: list[RerankedChunk] = []
        used_parents: set[str] = set()

        for chunk in chunks:
            pid = str(chunk.metadata.get("parent_id") or "")
            parent = parents.get(pid)
            if not parent or not parent.metadata.get("text"):
                out.append(chunk)
                continue
            if pid in used_parents:
                continue  # a sibling already contributed this parent
            used_parents.add(pid)

            meta = dict(parent.metadata)
            meta["expanded_from_child"] = chunk.chunk_id
            out.append(
                RerankedChunk(
                    chunk_id=pid,
                    text=str(parent.metadata["text"]),
                    metadata=meta,
                    relevance=chunk.relevance,
                    retrieval_score=chunk.retrieval_score,
                )
            )
        return out
