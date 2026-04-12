"""
EduPilot Reranker
=================
Post-retrieval reranking to select the most relevant chunks.

Strategy:
  1. Score each chunk with a query–chunk relevance signal
  2. Combine with the existing hybrid retrieval score
  3. Return the top-k highest-scoring chunks

Two reranking modes are available:
  - "keyword"  : fast, rule-based keyword overlap scoring (default, no extra deps)
  - "cross_encoder" : uses a sentence-transformers cross-encoder if installed
"""

from __future__ import annotations

import re
from typing import Literal

from config import BM25_WEIGHT, DEFAULT_CONFIDENCE_THRESHOLD, SEMANTIC_WEIGHT
from utils import RetrievedChunk, tokenize_simple

RerankMode = Literal["keyword", "cross_encoder"]


# ---------------------------------------------------------------------------
# Keyword overlap reranker (fast, no heavy models)
# ---------------------------------------------------------------------------

def _keyword_overlap_score(query_tokens: set[str], chunk_text: str) -> float:
    """
    Jaccard-style overlap between query tokens and chunk tokens.
    Also gives a small bonus for exact phrase matches.
    """
    chunk_tokens = set(tokenize_simple(chunk_text))
    if not query_tokens or not chunk_tokens:
        return 0.0

    overlap = len(query_tokens & chunk_tokens)
    union = len(query_tokens | chunk_tokens)
    jaccard = overlap / union if union else 0.0

    # Bonus: consecutive bigram matches in the chunk
    query_text_lower = " ".join(sorted(query_tokens))
    bigrams_found = 0
    for token in query_tokens:
        if len(token) >= 4 and token in chunk_text.lower():
            bigrams_found += 1

    bigram_bonus = min(bigrams_found * 0.05, 0.20)
    return min(jaccard + bigram_bonus, 1.0)


def rerank_keyword(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[RetrievedChunk]:
    """
    Rerank using keyword overlap + existing hybrid score.
    Returns top_k chunks above the confidence threshold.
    """
    query_tokens = set(tokenize_simple(query))

    for chunk in chunks:
        kw_score = _keyword_overlap_score(query_tokens, chunk.text)
        # Blend: 60% hybrid retrieval score + 40% keyword overlap
        chunk.rerank_score = 0.60 * chunk.rerank_score + 0.40 * kw_score

    # Sort descending
    chunks.sort(key=lambda c: c.rerank_score, reverse=True)

    # Apply confidence threshold
    filtered = [c for c in chunks if c.rerank_score >= confidence_threshold]

    # If nothing passes threshold but we have chunks, keep the best one
    if not filtered and chunks:
        filtered = [chunks[0]]

    return filtered[:top_k]


# ---------------------------------------------------------------------------
# Cross-encoder reranker (better quality, requires sentence-transformers)
# ---------------------------------------------------------------------------

_cross_encoder_model = None
_CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder_model = CrossEncoder(_CROSS_ENCODER_NAME)
        except (ImportError, Exception):
            _cross_encoder_model = None
    return _cross_encoder_model


def rerank_cross_encoder(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[RetrievedChunk]:
    """
    Rerank using a cross-encoder model if available; falls back to keyword reranking.
    """
    model = _get_cross_encoder()
    if model is None:
        return rerank_keyword(query, chunks, top_k, confidence_threshold)

    pairs = [(query, c.text) for c in chunks]
    try:
        scores = model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
    except Exception:
        return rerank_keyword(query, chunks, top_k, confidence_threshold)

    chunks.sort(key=lambda c: c.rerank_score, reverse=True)
    filtered = [c for c in chunks if c.rerank_score >= confidence_threshold]
    if not filtered and chunks:
        filtered = [chunks[0]]
    return filtered[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    mode: RerankMode = "keyword",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[RetrievedChunk]:
    """
    Rerank a list of retrieved chunks for a given query.

    Args:
        query: The user's question or sub-question.
        chunks: Candidates from hybrid retrieval.
        top_k: Number of chunks to return.
        mode: "keyword" (default, fast) or "cross_encoder" (better quality).
        confidence_threshold: Minimum rerank score to include a chunk.

    Returns:
        Top-k chunks sorted by rerank score (highest first).
    """
    if not chunks:
        return []

    if mode == "cross_encoder":
        return rerank_cross_encoder(query, chunks, top_k, confidence_threshold)
    else:
        return rerank_keyword(query, chunks, top_k, confidence_threshold)


def score_summary(chunks: list[RetrievedChunk]) -> dict:
    """Return a debug-friendly summary of scores."""
    if not chunks:
        return {"count": 0}
    return {
        "count": len(chunks),
        "top_score": round(chunks[0].rerank_score, 4) if chunks else 0,
        "min_score": round(chunks[-1].rerank_score, 4) if chunks else 0,
        "avg_score": round(sum(c.rerank_score for c in chunks) / len(chunks), 4),
        "sources": [c.citation_label() for c in chunks],
    }
