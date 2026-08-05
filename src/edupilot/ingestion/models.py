"""
Ingestion data model
====================
The document representation that sits between extraction and chunking.

Extraction produces a `ParsedDocument`: an ordered list of typed `Block`s
grouped into `Section`s. Chunking consumes that structure instead of a flat
string, which is what lets chunks respect section and paragraph boundaries.

Every block carries provenance (page, bbox) and an extraction `confidence`,
so a downstream consumer can tell native text from OCR output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BlockKind(str, Enum):
    """Semantic role of a block of text within a document."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FORMULA = "formula"
    CAPTION = "caption"
    CODE = "code"


#: Kinds that carry standalone meaning and should never be dropped as noise.
STRUCTURAL_KINDS = frozenset({BlockKind.HEADING, BlockKind.TABLE, BlockKind.FORMULA})


@dataclass
class Block:
    """One contiguous, semantically-typed piece of document text."""

    text: str
    kind: BlockKind
    page: int
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    level: int | None = None
    """Heading depth (1 = top level). None for non-headings."""
    confidence: float = 1.0
    """1.0 for native text layer, lower for OCR-recovered text."""
    column: int = 0
    """Column index on a multi-column page, in reading order."""

    @property
    def is_heading(self) -> bool:
        return self.kind is BlockKind.HEADING

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Section:
    """A heading and every block beneath it, up to the next same-or-higher heading."""

    title: str
    level: int
    blocks: list[Block] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    parent_titles: list[str] = field(default_factory=list)
    """Titles of enclosing sections, outermost first."""

    @property
    def breadcrumb(self) -> str:
        """Human-readable section path, e.g. 'Attention > Scaled Dot-Product'."""
        parts = [t for t in (*self.parent_titles, self.title) if t]
        return " > ".join(parts)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


@dataclass
class ExtractionStats:
    """Per-document extraction telemetry, surfaced in logs and the KB admin UI."""

    page_count: int = 0
    block_count: int = 0
    ocr_pages: int = 0
    empty_pages: int = 0
    table_count: int = 0
    formula_count: int = 0
    heading_count: int = 0
    dropped_noise_lines: int = 0
    stripped_running_heads: int = 0
    dehyphenated: int = 0
    columns_detected: int = 1
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "page_count": self.page_count,
            "block_count": self.block_count,
            "ocr_pages": self.ocr_pages,
            "empty_pages": self.empty_pages,
            "table_count": self.table_count,
            "formula_count": self.formula_count,
            "heading_count": self.heading_count,
            "dropped_noise_lines": self.dropped_noise_lines,
            "stripped_running_heads": self.stripped_running_heads,
            "dehyphenated": self.dehyphenated,
            "columns_detected": self.columns_detected,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ParsedDocument:
    """Fully extracted document, ready for chunking."""

    source_path: str
    filename: str
    title: str
    content_hash: str
    """SHA-256 of the raw file bytes. Identity for idempotent re-ingest."""
    blocks: list[Block] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    stats: ExtractionStats = field(default_factory=ExtractionStats)
    extractor: str = ""
    """Name+version of the extractor, recorded so a pipeline change is traceable."""

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    @property
    def word_count(self) -> int:
        return sum(b.word_count for b in self.blocks)


def file_content_hash(path: str | Path) -> str:
    """SHA-256 of a file's bytes, streamed so large PDFs don't load into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text: str) -> str:
    """SHA-256 of a text string. Used for chunk identity and the embedding cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
