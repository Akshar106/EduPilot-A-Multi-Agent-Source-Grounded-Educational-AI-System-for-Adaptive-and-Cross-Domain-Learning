"""
EduPilot retrieval
==================
Embeddings, vector storage, sparse encoding, query transformation, hybrid
search, and reranking.

    from edupilot.retrieval import HybridRetriever, RetrievalConfig, get_embedder

    result = retriever.retrieve("what is a p-value", config=RetrievalConfig())
    if result.is_empty:
        ...  # no supporting evidence — refuse rather than answer

`RetrievalResult.chunks` being empty is a meaningful outcome, not an error: it
means nothing in the knowledge base cleared the relevance floor.
"""

from .answer_cache import AnswerCache, CacheHit, normalize_question
from .embeddings import (
    Embedder,
    EmbeddingCache,
    SentenceTransformerEmbedder,
    TokenBudgetError,
    get_embedder,
    get_model_spec,
    reset_embedder,
)
from .hybrid import HybridRetriever, RetrievalConfig, RetrievalResult
from .indexer import Indexer, IndexRegistry, IndexResult, fit_sparse_encoder, rebuild_all
from .query_transform import TransformedQuery, expand_acronyms, transform_query
from .rerank import RerankedChunk, rerank, score_summary
from .sparse import BM25Encoder
from .vectorstore import (
    IndexPointer,
    InMemoryVectorStore,
    PineconeVectorStore,
    SearchHit,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)

__all__ = [
    "AnswerCache",
    "CacheHit",
    "normalize_question",
    "BM25Encoder",
    "Embedder",
    "EmbeddingCache",
    "HybridRetriever",
    "IndexPointer",
    "IndexRegistry",
    "IndexResult",
    "Indexer",
    "InMemoryVectorStore",
    "PineconeVectorStore",
    "RerankedChunk",
    "RetrievalConfig",
    "RetrievalResult",
    "SearchHit",
    "SentenceTransformerEmbedder",
    "TokenBudgetError",
    "TransformedQuery",
    "VectorRecord",
    "VectorStore",
    "VectorStoreError",
    "expand_acronyms",
    "fit_sparse_encoder",
    "get_embedder",
    "get_model_spec",
    "rebuild_all",
    "rerank",
    "reset_embedder",
    "score_summary",
    "transform_query",
]
