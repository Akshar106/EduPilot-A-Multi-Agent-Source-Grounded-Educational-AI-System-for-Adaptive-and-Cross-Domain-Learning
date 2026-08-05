"""
Layout-aware PDF extraction
===========================
Replaces the previous one-liner (`page.get_text()`), which returned a flat
string with no structure, no tables, nothing for scanned pages, and math
mangled into runs like ``W Q i ∈Rdmodel×dk``.

What this module adds:

  reading order      two-column academic papers are emitted column-by-column,
                     not interleaved line-by-line across the gutter
  headings           detected from font size relative to the document body
                     size, giving chunking real section boundaries
  math               sub/superscript spans are reconstructed from PyMuPDF span
                     metadata, so ``W_Q^i`` survives instead of ``W Q i``
  tables             extracted with pdfplumber and rendered as markdown; the
                     overlapping text-layer blocks are suppressed so table
                     content is not indexed twice
  scanned pages      pages with no usable text layer are rasterized at 300 DPI
                     and OCR'd with Tesseract (6% of the EduPilot corpus)
  running heads      repeated chrome like "3.4. CONDITIONAL PROBABILITY 61"
                     is detected across pages and stripped

Everything degrades gracefully: no Tesseract binary disables OCR, no
pdfplumber disables tables, and extraction still succeeds.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF

from .models import Block, BlockKind, ExtractionStats, ParsedDocument, file_content_hash
from .normalize import (
    find_running_heads,
    normalize_block_text,
    strip_running_heads,
)

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "pymupdf-layout/1.0"

# --- Tunables ---------------------------------------------------------------

#: Below this many characters a page is treated as having no usable text layer.
MIN_NATIVE_CHARS = 60
#: Resolution for rasterizing a page before OCR. 300 DPI is the Tesseract sweet spot.
OCR_DPI = 300
#: Confidence recorded on OCR-derived blocks, so downstream can weight them lower.
OCR_CONFIDENCE = 0.60
#: A span this much smaller than its line's dominant size is a sub/superscript.
SCRIPT_SIZE_RATIO = 0.76
#: Vertical offset (as a fraction of font size) separating superscript from subscript.
SCRIPT_BASELINE_EPS = 0.12
#: A block must be this much larger than body text to be a heading.
HEADING_SIZE_RATIO = 1.18
#: Headings are short; longer blocks in a large font are display text, not headings.
HEADING_MAX_WORDS = 20
#: A block wider than this fraction of the page spans all columns.
FULL_WIDTH_RATIO = 0.70
#: Minimum blocks per side before a page is accepted as two-column.
MIN_BLOCKS_PER_COLUMN = 3

_MATH_CHARS = set("∈∑∏∫√≈≤≥≠±∞∂∇⊤⊗⊕×·αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ→←↔⇒⇔∀∃∅∪∩⊂⊆")
_SENTENCE_END = re.compile(r"[.!?:;]\s*$")
_LIST_START = re.compile(r"^\s*(?:[-*]|\d+[.)]|[a-z][.)])\s+")


# ---------------------------------------------------------------------------
# Span-level reconstruction
# ---------------------------------------------------------------------------


def _dominant_size(spans: list[dict]) -> float:
    """Font size covering the most characters in a line. Robust to stray glyphs."""
    weights: Counter[float] = Counter()
    for s in spans:
        weights[round(float(s.get("size", 0)), 1)] += len(s.get("text", ""))
    return weights.most_common(1)[0][0] if weights else 0.0


def _line_to_text(line: dict) -> tuple[str, bool]:
    """
    Render one line's spans to text, reconstructing sub/superscripts.

    PyMuPDF sets bit 0 of a span's ``flags`` for superscripts. Subscripts have
    no flag, so they are inferred: a span noticeably smaller than the line's
    dominant size whose baseline sits *below* the dominant baseline.

    Returns (text, contained_script) — the flag feeds formula classification.
    """
    spans = [s for s in line.get("spans", []) if s.get("text")]
    if not spans:
        return "", False

    dom_size = _dominant_size(spans)
    # Baseline of the dominant-size spans; y grows downward in PDF space.
    base_ys = [
        float(s.get("origin", (0, 0))[1])
        for s in spans
        if abs(round(float(s.get("size", 0)), 1) - dom_size) < 0.05
    ]
    dom_baseline = sum(base_ys) / len(base_ys) if base_ys else 0.0

    parts: list[str] = []
    saw_script = False
    for s in spans:
        text = s["text"]
        size = float(s.get("size", dom_size))
        flags = int(s.get("flags", 0))
        origin_y = float(s.get("origin", (0, dom_baseline))[1])

        is_small = dom_size > 0 and size < dom_size * SCRIPT_SIZE_RATIO
        if not is_small:
            parts.append(text)
            continue

        token = text.strip()
        if not token:
            parts.append(text)
            continue

        eps = max(dom_size * SCRIPT_BASELINE_EPS, 0.5)
        is_super = bool(flags & 1) or origin_y < dom_baseline - eps
        wrapped = token if len(token) == 1 else "{" + token + "}"
        parts.append(("^" if is_super else "_") + wrapped)
        saw_script = True

    return "".join(parts), saw_script


def _block_to_text(raw_block: dict) -> tuple[str, float, bool]:
    """
    Render a PyMuPDF dict block to text.

    Returns (text, dominant_font_size, contained_script).
    """
    lines_out: list[str] = []
    sizes: Counter[float] = Counter()
    saw_script = False

    for line in raw_block.get("lines", []):
        text, script = _line_to_text(line)
        saw_script = saw_script or script
        if text.strip():
            lines_out.append(text)
        for s in line.get("spans", []):
            sizes[round(float(s.get("size", 0)), 1)] += len(s.get("text", ""))

    dom = sizes.most_common(1)[0][0] if sizes else 0.0
    return "\n".join(lines_out), dom, saw_script


def _is_bold(raw_block: dict) -> bool:
    """True when most of a block's characters are bold (PyMuPDF flag bit 4)."""
    bold = plain = 0
    for line in raw_block.get("lines", []):
        for s in line.get("spans", []):
            n = len(s.get("text", ""))
            if int(s.get("flags", 0)) & 16:
                bold += n
            else:
                plain += n
    return bold > plain


# ---------------------------------------------------------------------------
# Column detection and reading order
# ---------------------------------------------------------------------------


def _detect_columns(blocks: list[dict], page_width: float) -> int:
    """
    Return 2 when the page is laid out in two columns, else 1.

    A page qualifies when most blocks are narrower than 60% of the page and
    they split into left and right groups with at least a few blocks each.
    This matches the arXiv papers in the corpus (GPT-4, PaLM 2, InstructBLIP)
    while leaving single-column slides and textbook pages alone.
    """
    if page_width <= 0 or len(blocks) < 2 * MIN_BLOCKS_PER_COLUMN:
        return 1

    mid = page_width / 2
    narrow = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) < page_width * 0.60]
    if len(narrow) < len(blocks) * 0.60:
        return 1

    left = [b for b in narrow if (b["bbox"][0] + b["bbox"][2]) / 2 < mid]
    right = [b for b in narrow if (b["bbox"][0] + b["bbox"][2]) / 2 >= mid]
    if len(left) >= MIN_BLOCKS_PER_COLUMN and len(right) >= MIN_BLOCKS_PER_COLUMN:
        return 2
    return 1


def _sort_reading_order(blocks: list[dict], page_width: float, n_columns: int) -> list[tuple[dict, int]]:
    """
    Order blocks the way a human reads them, and tag each with its column.

    Single column: plain top-to-bottom.

    Two columns: full-width blocks (titles, wide figures, footnote rules) act
    as horizontal dividers. Blocks are grouped into bands between those
    dividers, and each band is emitted left column first, then right. Sorting
    naively by ``(column, y)`` across the whole page would hoist a
    left-column footnote above a full-width section title.
    """
    if n_columns == 1:
        ordered = sorted(blocks, key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))
        return [(b, 0) for b in ordered]

    mid = page_width / 2
    by_y = sorted(blocks, key=lambda b: (round(b["bbox"][1], 1), b["bbox"][0]))

    out: list[tuple[dict, int]] = []
    band: list[dict] = []

    def flush() -> None:
        if not band:
            return
        keyed = [
            (b, 0 if (b["bbox"][0] + b["bbox"][2]) / 2 < mid else 1) for b in band
        ]
        keyed.sort(key=lambda kv: (kv[1], round(kv[0]["bbox"][1], 1), kv[0]["bbox"][0]))
        out.extend(keyed)
        band.clear()

    for b in by_y:
        width = b["bbox"][2] - b["bbox"][0]
        if width >= page_width * FULL_WIDTH_RATIO:
            flush()
            out.append((b, 0))
        else:
            band.append(b)
    flush()
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _math_density(text: str) -> float:
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return 0.0
    return sum(1 for c in stripped if c in _MATH_CHARS) / len(stripped)


def _classify(text: str, size: float, body_size: float, bold: bool, has_script: bool) -> tuple[BlockKind, int | None]:
    """Assign a semantic kind (and heading level) to a block of text."""
    stripped = text.strip()
    words = stripped.split()

    if not words:
        return BlockKind.PARAGRAPH, None

    # Formula: dense in math glyphs, or script-heavy and short.
    density = _math_density(stripped)
    if density > 0.12 or (has_script and len(words) <= 12 and density > 0.04):
        return BlockKind.FORMULA, None

    if _LIST_START.match(stripped):
        return BlockKind.LIST_ITEM, None

    if re.match(r"^\s*(Figure|Table|Fig\.|Eq\.|Algorithm)\s*\d+", stripped, re.IGNORECASE):
        return BlockKind.CAPTION, None

    # Heading: visually larger than body text, short, and not a full sentence.
    if body_size > 0 and len(words) <= HEADING_MAX_WORDS:
        ratio = size / body_size
        looks_titular = not _SENTENCE_END.search(stripped)
        if ratio >= HEADING_SIZE_RATIO and looks_titular:
            if ratio >= body_size and ratio >= 1.6:
                level = 1
            elif ratio >= 1.32:
                level = 2
            else:
                level = 3
            return BlockKind.HEADING, level
        if bold and ratio >= 1.0 and looks_titular and len(words) <= 10:
            return BlockKind.HEADING, 3

    return BlockKind.PARAGRAPH, None


def _document_body_size(doc: fitz.Document, sample_pages: int = 25) -> float:
    """
    Modal font size across the document, weighted by character count.

    Sampling evenly across the file keeps a 457-page textbook fast while
    staying representative.
    """
    sizes: Counter[float] = Counter()
    step = max(1, len(doc) // sample_pages)
    for i in range(0, len(doc), step):
        try:
            data = doc[i].get_text("dict")
        except Exception:  # pragma: no cover - corrupt page
            continue
        for blk in data.get("blocks", []):
            for line in blk.get("lines", []):
                for s in line.get("spans", []):
                    sizes[round(float(s.get("size", 0)), 1)] += len(s.get("text", ""))
    return sizes.most_common(1)[0][0] if sizes else 10.0


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


#: Table plausibility gates. Lecture slides are full of drawn rectangles that
#: pdfplumber's line-based detector happily reports as one-cell "tables"; these
#: thresholds reject them without losing real data tables.
TABLE_MIN_COLS = 2
TABLE_MIN_ROWS = 2
TABLE_MIN_FILL = 0.50
"""At least half the cells must be non-empty."""
TABLE_MAX_CELL_CHARS = 220
"""A cell holding a paragraph means the region is a text box, not a table."""
TABLE_MAX_CELL_SHARE = 0.55
"""No single cell may hold this share of the table's text."""


def _is_plausible_table(rows: list[list[str]]) -> bool:
    """
    Reject regions that are drawn boxes rather than genuine tables.

    Without this, a slide whose body sits inside a rounded rectangle becomes a
    single-cell "table" and the same text is indexed twice — once as prose and
    once as a bogus markdown table.
    """
    if len(rows) < TABLE_MIN_ROWS:
        return False
    n_cols = max(len(r) for r in rows)
    if n_cols < TABLE_MIN_COLS:
        return False

    cells = [c for r in rows for c in r]
    total_cells = len(rows) * n_cols
    filled = [c for c in cells if c.strip()]
    if not filled or len(filled) / total_cells < TABLE_MIN_FILL:
        return False

    lengths = [len(c) for c in filled]
    if max(lengths) > TABLE_MAX_CELL_CHARS:
        return False
    total_chars = sum(lengths)
    if total_chars and max(lengths) / total_chars > TABLE_MAX_CELL_SHARE:
        return False
    return True


def _table_to_markdown(rows: list[list[Any]]) -> str:
    """
    Render a pdfplumber table as markdown, or "" if it fails plausibility checks.
    """
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [(str(c).replace("\n", " ").strip() if c is not None else "") for c in row]
        if any(cells):
            cleaned.append(cells)
    if len(cleaned) < TABLE_MIN_ROWS:
        return ""

    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]

    if not _is_plausible_table(cleaned):
        return ""

    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(r) + " |" for r in body)
    return "\n".join(lines)


def _extract_tables(pdf_path: str, page_numbers: Iterable[int]) -> dict[int, list[tuple[str, tuple]]]:
    """
    Extract tables with pdfplumber, keyed by 0-based page index.

    Returns {page_index: [(markdown, bbox), ...]}. Returns {} when pdfplumber
    is unavailable, so table support is strictly additive.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.info("pdfplumber not installed — table extraction disabled")
        return {}

    # Require ruling lines on both axes. The default settings accept a region
    # bounded on one side only, which matches most slide graphics.
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 12,
    }

    wanted = set(page_numbers)
    found: dict[int, list[tuple[str, tuple]]] = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i not in wanted:
                    continue
                try:
                    tables = page.find_tables(table_settings=settings)
                except Exception:
                    continue
                for t in tables:
                    try:
                        md = _table_to_markdown(t.extract())
                    except Exception:
                        continue
                    if md:
                        found.setdefault(i, []).append((md, tuple(t.bbox)))
    except Exception as exc:
        logger.warning("pdfplumber failed on %s: %s", pdf_path, exc)
        return {}
    return found


def _pages_with_ruling_lines(doc: fitz.Document, min_lines: int = 6) -> list[int]:
    """
    Pages likely to contain a ruled table.

    pdfplumber's line-based detection only finds tables drawn with rules, and
    it is slow, so this pre-filter with PyMuPDF's cheap drawing count keeps a
    457-page textbook from taking minutes.
    """
    out: list[int] = []
    for i, page in enumerate(doc):
        try:
            if len(page.get_drawings()) >= min_lines:
                out.append(i)
        except Exception:
            continue
    return out


def _overlaps(inner: tuple, outer: tuple, tol: float = 4.0) -> bool:
    """True when `inner`'s centre falls inside `outer` (expanded by `tol`)."""
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return (outer[0] - tol) <= cx <= (outer[2] + tol) and (outer[1] - tol) <= cy <= (outer[3] + tol)


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

_ocr_available: bool | None = None


def _ocr_ready() -> bool:
    """Check once whether pytesseract and the Tesseract binary are both usable."""
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        _ocr_available = True
    except Exception as exc:
        logger.info("OCR unavailable (%s) — scanned pages will be skipped", exc)
        _ocr_available = False
    return _ocr_available


def _ocr_page(page: fitz.Page) -> str:
    """Rasterize a page at OCR_DPI and run Tesseract over it."""
    import io

    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_pdf(path: str, *, enable_ocr: bool = True, enable_tables: bool = True) -> ParsedDocument:
    """
    Extract a PDF into a structured ParsedDocument.

    Args:
        path: Path to the PDF.
        enable_ocr: Rasterize+OCR pages with no usable text layer.
        enable_tables: Extract ruled tables with pdfplumber.

    Raises:
        RuntimeError: if the file cannot be opened as a PDF.
    """
    started = time.perf_counter()
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot open PDF {path!r}: {exc}") from exc

    stats = ExtractionStats(page_count=len(doc))
    body_size = _document_body_size(doc)

    tables_by_page: dict[int, list[tuple[str, tuple]]] = {}
    if enable_tables:
        candidates = _pages_with_ruling_lines(doc)
        if candidates:
            tables_by_page = _extract_tables(path, candidates)
            stats.table_count = sum(len(v) for v in tables_by_page.values())

    # Pass 1 — raw per-page text, used only to learn the running-head signatures.
    raw_page_text = [p.get_text() for p in doc]
    running_heads = find_running_heads(raw_page_text)

    blocks: list[Block] = []
    max_columns = 1

    for pno, page in enumerate(doc):
        page_no = pno + 1
        page_width = float(page.rect.width)
        native_len = len(raw_page_text[pno].strip())

        # --- Scanned page: no usable text layer -----------------------------
        if native_len < MIN_NATIVE_CHARS:
            has_visual = bool(page.get_images()) or bool(page.get_drawings())
            if enable_ocr and has_visual and _ocr_ready():
                try:
                    ocr_text = _ocr_page(page)
                except Exception as exc:
                    logger.warning("OCR failed on %s p.%d: %s", path, page_no, exc)
                    ocr_text = ""
                clean, counters = normalize_block_text(ocr_text)
                if len(clean.strip()) >= MIN_NATIVE_CHARS:
                    stats.ocr_pages += 1
                    stats.dehyphenated += counters["dehyphenated"]
                    stats.dropped_noise_lines += counters["dropped_noise_lines"]
                    blocks.append(
                        Block(
                            text=clean,
                            kind=BlockKind.PARAGRAPH,
                            page=page_no,
                            bbox=tuple(page.rect),
                            confidence=OCR_CONFIDENCE,
                        )
                    )
                    continue
            if native_len == 0:
                stats.empty_pages += 1
                continue

        # --- Native text layer ----------------------------------------------
        try:
            data = page.get_text("dict")
        except Exception as exc:
            logger.warning("get_text failed on %s p.%d: %s", path, page_no, exc)
            continue

        text_blocks = [b for b in data.get("blocks", []) if b.get("type") == 0 and b.get("lines")]
        if not text_blocks:
            stats.empty_pages += 1
            continue

        n_cols = _detect_columns(text_blocks, page_width)
        max_columns = max(max_columns, n_cols)
        ordered = _sort_reading_order(text_blocks, page_width, n_cols)

        page_tables = tables_by_page.get(pno, [])

        for raw, column in ordered:
            # Suppress text blocks sitting inside an extracted table region.
            if any(_overlaps(raw["bbox"], tbbox) for _, tbbox in page_tables):
                continue

            text, size, has_script = _block_to_text(raw)
            if not text.strip():
                continue

            text, n_stripped = strip_running_heads(text, running_heads)
            stats.stripped_running_heads += n_stripped

            clean, counters = normalize_block_text(text)
            stats.dehyphenated += counters["dehyphenated"]
            stats.dropped_noise_lines += counters["dropped_noise_lines"]
            if not clean.strip():
                continue

            kind, level = _classify(clean, size, body_size, _is_bold(raw), has_script)
            if kind is BlockKind.HEADING:
                stats.heading_count += 1
            elif kind is BlockKind.FORMULA:
                stats.formula_count += 1

            blocks.append(
                Block(
                    text=clean,
                    kind=kind,
                    page=page_no,
                    bbox=tuple(float(v) for v in raw["bbox"]),
                    level=level,
                    column=column,
                )
            )

        # Emit tables after the page's prose so they land in the same section.
        for md, tbbox in page_tables:
            blocks.append(
                Block(text=md, kind=BlockKind.TABLE, page=page_no, bbox=tbbox)
            )

    doc.close()

    stats.block_count = len(blocks)
    stats.columns_detected = max_columns
    stats.duration_ms = int((time.perf_counter() - started) * 1000)

    return ParsedDocument(
        source_path=str(path),
        filename=Path(path).name,
        title=_infer_title(blocks, Path(path).stem),
        content_hash=file_content_hash(path),
        blocks=blocks,
        stats=stats,
        extractor=EXTRACTOR_VERSION,
    )


#: Headings that are structural markers rather than document titles.
_STRUCTURAL_HEADING = re.compile(
    r"^\s*(?:chapter|section|part|appendix|lecture|unit|module)?\s*\d+(?:\.\d+)*\.?\s*$",
    re.IGNORECASE,
)


def _infer_title(blocks: list[Block], fallback: str) -> str:
    """
    Use the first level-1/2 heading on page 1 as the title, else the filename.

    A heading like "Chapter 1" is a structural marker, not a title — using it
    would stamp every chunk in a 457-page textbook with the same misleading
    breadcrumb root, so those fall back to the filename.
    """
    for b in blocks[:12]:
        if b.is_heading and b.page == 1 and (b.level or 3) <= 2:
            title = b.text.strip().split("\n")[0]
            if 3 <= len(title) <= 160 and not _STRUCTURAL_HEADING.match(title):
                return title
    return fallback.replace("_", " ").replace("-", " ").strip()
