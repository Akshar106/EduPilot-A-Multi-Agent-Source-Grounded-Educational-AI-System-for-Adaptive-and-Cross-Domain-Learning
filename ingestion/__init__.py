"""
EduPilot ingestion
==================
Document extraction and chunking.

    from ingestion import extract_document, chunk_document

    doc = extract_document("knowledge_base/llm/LLMs-04-attention.pdf")
    chunks = chunk_document(doc, domain="LLM")

`extract_document` returns a structured `ParsedDocument` (typed blocks grouped
into sections, with provenance and extraction telemetry). `chunk_document`
turns that into token-bounded `DocumentChunk`s that respect section and
sentence boundaries.
"""

from .chunking import ChunkingConfig, DocumentChunk, chunk_document, chunking_report
from .models import (
    Block,
    BlockKind,
    ExtractionStats,
    ParsedDocument,
    Section,
    file_content_hash,
    text_hash,
)
from .pipeline import (
    SUPPORTED_EXTENSIONS,
    build_sections,
    extract_document,
    extraction_report,
)

__all__ = [
    "Block",
    "BlockKind",
    "ChunkingConfig",
    "DocumentChunk",
    "ExtractionStats",
    "ParsedDocument",
    "SUPPORTED_EXTENSIONS",
    "Section",
    "build_sections",
    "chunk_document",
    "chunking_report",
    "extract_document",
    "extraction_report",
    "file_content_hash",
    "text_hash",
]
