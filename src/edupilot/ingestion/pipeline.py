"""
Extraction pipeline
===================
Format dispatch plus section assembly.

`extract_document` picks the right extractor by file extension; `build_sections`
folds the flat block list into a heading hierarchy. That hierarchy is what
gives each chunk a breadcrumb ("Attention > Scaled Dot-Product Attention"),
which is prepended to the chunk text so an isolated chunk still says what it
is about — the single cheapest retrieval-quality win available.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .models import Block, ParsedDocument, Section
from .office import extract_docx, extract_markdown
from .pdf import extract_pdf

logger = logging.getLogger(__name__)

#: Extensions this pipeline can parse.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".docx"})

#: Blocks before the first heading are grouped under this pseudo-section.
PREAMBLE_TITLE = "(document start)"

#: A heading that is only a section number ("3.", "1.1.2") — the descriptive
#: text sits in a separate span, so the number alone is a useless breadcrumb.
_BARE_NUMBER_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?$")


def extract_document(
    path: str | Path,
    *,
    enable_ocr: bool = True,
    enable_tables: bool = True,
) -> ParsedDocument:
    """
    Extract any supported document into a `ParsedDocument` with sections built.

    Raises:
        ValueError: unsupported extension.
        FileNotFoundError: missing file.
        RuntimeError: the extractor could not read the file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such document: {p}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported extension {ext!r}; expected one of {sorted(SUPPORTED_EXTENSIONS)}")

    if ext == ".pdf":
        doc = extract_pdf(str(p), enable_ocr=enable_ocr, enable_tables=enable_tables)
    elif ext == ".docx":
        doc = extract_docx(str(p))
    else:
        doc = extract_markdown(str(p))

    doc.sections = build_sections(doc.blocks)
    logger.info(
        "extracted %s: %d blocks, %d sections, %d pages, %d tables, %d OCR pages in %dms",
        doc.filename,
        doc.stats.block_count,
        len(doc.sections),
        doc.stats.page_count,
        doc.stats.table_count,
        doc.stats.ocr_pages,
        doc.stats.duration_ms,
    )
    return doc


def build_sections(blocks: list[Block]) -> list[Section]:
    """
    Fold a flat block list into a heading hierarchy.

    A heading opens a new section. Its `parent_titles` come from the stack of
    open headings at strictly shallower levels, so a level-3 heading nested
    under a level-1 records that level-1 as its parent even when no level-2
    heading exists between them.

    Content appearing before any heading is kept in a preamble section rather
    than discarded.
    """
    sections: list[Section] = []
    # Stack of (level, title) for currently open ancestor headings.
    stack: list[tuple[int, str]] = []
    current: Section | None = None

    def close(sec: Section | None) -> None:
        if sec is not None and sec.blocks:
            sec.page_start = min(b.page for b in sec.blocks)
            sec.page_end = max(b.page for b in sec.blocks)
            sections.append(sec)

    for block in blocks:
        if block.is_heading:
            close(current)
            level = block.level or 3
            title = block.text.strip().split("\n")[0]

            while stack and stack[-1][0] >= level:
                stack.pop()

            current = Section(
                title=title,
                level=level,
                page_start=block.page,
                page_end=block.page,
                parent_titles=[t for _, t in stack],
            )
            stack.append((level, title))
            continue

        if current is None:
            current = Section(title=PREAMBLE_TITLE, level=0, page_start=block.page)
        current.blocks.append(block)

    close(current)

    for sec in sections:
        _expand_bare_number_title(sec)
    return sections


def _expand_bare_number_title(section: Section) -> None:
    """
    Give a number-only heading a descriptive title from its first line of body.

    Textbook PDFs frequently split "1.1.2  Estimating the Speed of Light" into
    two spans, so the heading block holds only "1.1.2". Left alone that
    produces the breadcrumb "Chapter 1 > Experiments > 1.1.2", which tells a
    retrieved chunk nothing about its own topic.
    """
    if not _BARE_NUMBER_HEADING.match(section.title.strip()):
        return
    for block in section.blocks:
        line = block.text.strip().split("\n")[0].strip()
        if 3 <= len(line) <= 90:
            section.title = f"{section.title.strip()} {line}"
            return


def extraction_report(doc: ParsedDocument) -> dict:
    """
    Summarize extraction quality for logging and the KB admin view.

    `low_confidence_ratio` is the share of text recovered by OCR rather than
    read from a native text layer — a useful signal that a source document is
    a scan and its answers should be treated more cautiously.
    """
    kinds: dict[str, int] = {}
    for b in doc.blocks:
        kinds[b.kind.value] = kinds.get(b.kind.value, 0) + 1

    total_words = doc.word_count or 1
    ocr_words = sum(b.word_count for b in doc.blocks if b.confidence < 1.0)

    return {
        "filename": doc.filename,
        "title": doc.title,
        "content_hash": doc.content_hash[:16],
        "extractor": doc.extractor,
        "sections": len(doc.sections),
        "words": doc.word_count,
        "block_kinds": kinds,
        "low_confidence_ratio": round(ocr_words / total_words, 3),
        **doc.stats.as_dict(),
    }
