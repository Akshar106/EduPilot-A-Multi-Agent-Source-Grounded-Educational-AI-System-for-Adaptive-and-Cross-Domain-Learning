"""
EduPilot
========
A multi-agent, source-grounded educational RAG system.

Layout::

    edupilot.core         configuration, composition root, logging
    edupilot.db           SQLite persistence
    edupilot.llm          provider-agnostic LLM client with fallback
    edupilot.ingestion    PDF/DOCX/Markdown extraction and chunking
    edupilot.retrieval    embeddings, vector store, hybrid search, reranking
    edupilot.agents       the router → splitter → answerer → synthesizer pipeline
    edupilot.guardrails   citation, grounding, injection and output checks
    edupilot.security     auth, rate limiting, upload validation, error envelopes
    edupilot.evaluation   the offline test suite and its metrics
    edupilot.api          FastAPI application and routers

Nothing is imported eagerly here: pulling in `edupilot` must not load a model
or open a network connection.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
