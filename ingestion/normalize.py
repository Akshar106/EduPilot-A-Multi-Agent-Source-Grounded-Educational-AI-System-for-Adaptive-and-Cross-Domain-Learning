"""
Text normalization
==================
Cleanup applied to raw extracted text before it becomes a Block.

Each function targets a defect measured on the EduPilot corpus (46 PDFs,
666 pages sampled):

  ligatures            14% of pages   "classiﬁcation" never matches "classification"
  orphaned bullets     12% of pages   PyMuPDF emits "●" on its own line
  hyphen line-breaks    5% of pages   "regu-\\nlarization" tokenizes as two words
  dot-leader artifacts  common in the textbook PDFs — ASCII figures and TOC leaders
  running heads         every textbook page repeats "3.4. CONDITIONAL PROBABILITY 61"

All functions are pure and independently testable.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Bullet glyphs PyMuPDF commonly emits as standalone lines.
BULLET_CHARS = "●○•▪▫◦‣⁃∙·"
_BULLET_ONLY = re.compile(rf"^\s*[{re.escape(BULLET_CHARS)}]\s*$")
_BULLET_LEAD = re.compile(rf"^\s*[{re.escape(BULLET_CHARS)}]\s*")

# A line that is mostly dots/periods is a leader or an ASCII-art figure, not prose.
_DOT_LEADER = re.compile(r"^[\s.·˙_—–-]*$")

# Hyphen at end of line joining two lowercase word halves.
_HYPHEN_BREAK = re.compile(r"([a-z])[-‐‑]\s*\n\s*([a-z])")

# Control characters, excluding \t \n \r.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Runs of 3+ identical punctuation marks used as visual rules.
_PUNCT_RUN = re.compile(r"([.·_=~*—–-])\1{3,}")

_WS_RUN = re.compile(r"[ \t ]{2,}")
_BLANK_RUN = re.compile(r"\n{3,}")

#: A line must be at least this fraction alphanumeric to count as prose.
_MIN_ALNUM_RATIO = 0.35


def normalize_unicode(text: str) -> str:
    """
    Apply NFKC so ligatures, fullwidth forms, and compatibility characters
    fold into their canonical equivalents (ﬁ → fi, ﬂ → fl, ％ → %).

    Math symbols (∈ ∑ √ ≤ α β) are unaffected — NFKC leaves them alone.
    """
    text = unicodedata.normalize("NFKC", text)
    # NFKC does not touch these; normalize them so quoting is consistent.
    return (
        text.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "—")
    )


def dehyphenate(text: str) -> tuple[str, int]:
    """
    Rejoin words split across a line break by a hyphen.

    Only fires between two lowercase letters, which leaves genuine compounds
    ("cross-validation", "F-1") and hyphenated line starts intact.

    Returns (text, number_of_joins).
    """
    joins = len(_HYPHEN_BREAK.findall(text))
    return _HYPHEN_BREAK.sub(r"\1\2", text), joins


def merge_orphan_bullets(text: str) -> str:
    """
    Attach a standalone bullet glyph to the text line that follows it.

    PyMuPDF renders slide bullets as their own line, which strands the marker
    and makes the following line look like an unrelated fragment:

        ●
        Absolute positional encodings add ...   →   - Absolute positional encodings add ...
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if _BULLET_ONLY.match(lines[i]):
            # Find the next non-blank line and prefix it.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                out.append("- " + _BULLET_LEAD.sub("", lines[j]).strip())
                i = j + 1
                continue
            i += 1
            continue
        if _BULLET_LEAD.match(lines[i]):
            out.append("- " + _BULLET_LEAD.sub("", lines[i]).strip())
        else:
            out.append(lines[i])
        i += 1
    return "\n".join(out)


def _alnum_ratio(line: str) -> float:
    stripped = line.strip()
    if not stripped:
        return 0.0
    alnum = sum(1 for c in stripped if c.isalnum())
    return alnum / len(stripped)


def drop_noise_lines(text: str) -> tuple[str, int]:
    """
    Remove lines that carry no retrievable content.

    Targets dot leaders and the ASCII-art figures in the statistics textbook,
    where a single page can yield dozens of lines like::

        •...............................................................
        ............................................. .......

    A line is noise when it is entirely leader characters, or when it is long
    and mostly non-alphanumeric. Short lines are kept — they are usually axis
    labels, equation fragments, or single-token slide text.

    Returns (text, number_of_lines_dropped).
    """
    kept: list[str] = []
    dropped = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if _DOT_LEADER.match(stripped):
            dropped += 1
            continue
        if len(stripped) >= 12 and _alnum_ratio(stripped) < _MIN_ALNUM_RATIO:
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept), dropped


def collapse_whitespace(text: str) -> str:
    """Normalize runs of spaces, blank lines, and decorative punctuation rules."""
    text = _CONTROL.sub(" ", text)
    text = _PUNCT_RUN.sub(r"\1\1\1", text)
    text = _WS_RUN.sub(" ", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def normalize_block_text(text: str) -> tuple[str, dict[str, int]]:
    """
    Run the full normalization chain over one block of extracted text.

    Returns (clean_text, counters) where counters feeds ExtractionStats.
    """
    text = normalize_unicode(text)
    text, joins = dehyphenate(text)
    text = merge_orphan_bullets(text)
    text, dropped = drop_noise_lines(text)
    text = collapse_whitespace(text)
    return text, {"dehyphenated": joins, "dropped_noise_lines": dropped}


# ---------------------------------------------------------------------------
# Running head / foot detection
# ---------------------------------------------------------------------------

#: A candidate must appear on at least this fraction of pages to be a running head.
_RUNNING_HEAD_MIN_RATIO = 0.30
#: ...and the document must have at least this many pages for the test to be meaningful.
_RUNNING_HEAD_MIN_PAGES = 6
#: Running heads are short. Anything longer is body text that happens to repeat.
_RUNNING_HEAD_MAX_CHARS = 90

_DIGITS = re.compile(r"\d+")


def _head_signature(line: str) -> str:
    """
    Canonical form of a header/footer line, with page numbers masked.

    'CHAPTER 3. PROBABILITY 61' and 'CHAPTER 3. PROBABILITY 62' share a
    signature, so the pair is recognised as one running head.
    """
    return _DIGITS.sub("#", line.strip().lower())


def find_running_heads(page_texts: list[str], band: int = 2) -> set[str]:
    """
    Identify header/footer signatures that repeat across a document.

    Looks at the first and last `band` non-blank lines of every page and
    returns the signatures common to at least 30% of them.

    Short documents (slide decks under 6 pages) are skipped — with few pages,
    legitimate repeated content would be misread as chrome.
    """
    if len(page_texts) < _RUNNING_HEAD_MIN_PAGES:
        return set()

    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        candidates = lines[:band] + lines[-band:]
        for line in set(candidates):
            if len(line.strip()) <= _RUNNING_HEAD_MAX_CHARS:
                counts[_head_signature(line)] += 1

    threshold = max(2, int(len(page_texts) * _RUNNING_HEAD_MIN_RATIO))
    return {sig for sig, n in counts.items() if n >= threshold and sig.strip("# .")}


def strip_running_heads(text: str, signatures: set[str], band: int = 2) -> tuple[str, int]:
    """
    Remove known running-head lines from the top and bottom of one page.

    Only the outer `band` lines are considered, so a section title that also
    appears mid-page as a real heading survives.

    Returns (text, number_of_lines_stripped).
    """
    if not signatures:
        return text, 0

    lines = text.split("\n")
    # Index positions of the non-blank lines eligible for stripping.
    non_blank = [i for i, ln in enumerate(lines) if ln.strip()]
    if not non_blank:
        return text, 0
    eligible = set(non_blank[:band]) | set(non_blank[-band:])

    kept: list[str] = []
    stripped = 0
    for i, line in enumerate(lines):
        if i in eligible and _head_signature(line) in signatures:
            stripped += 1
            continue
        kept.append(line)
    return "\n".join(kept), stripped
