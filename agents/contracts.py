"""
Prompt contracts
================
How untrusted document text is packaged into a prompt, and how agent output
is parsed back out.

The central mechanism is the **nonce fence**. Retrieved chunks are wrapped in
a delimiter containing a random token generated per request:

    <<<SOURCE-MATERIAL a3f9c1e2>>>
    [Source 1] ...
    <<<END-SOURCE-MATERIAL a3f9c1e2>>>

A static delimiter like the previous `--- COURSE SOURCE MATERIAL ---` can be
forged: a PDF containing that exact line followed by "--- END ---" and then
its own instructions appears to the model to be outside the data region. The
attacker cannot predict a per-request nonce, so the fence cannot be closed
early. Any occurrence of the nonce inside the content itself is stripped
before assembly, which closes the reflection case.

This is defence in depth, not a complete solution — it pairs with the explicit
instruction hierarchy in the system prompt and the injection scanner in
`guardrails/injection.py`.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from guardrails.refusal import REFUSAL_MARKER, is_refusal, strip_refusal_marker

#: Maximum characters of a single chunk placed in a prompt. Chunks are already
#: token-bounded by the chunker; this is a defensive cap for anything that
#: reaches the prompt by another path.
MAX_CHUNK_CHARS = 6000


@dataclass
class SourceFence:
    """A per-request delimiter pair for untrusted content."""

    nonce: str

    @property
    def open(self) -> str:
        return f"<<<SOURCE-MATERIAL {self.nonce}>>>"

    @property
    def close(self) -> str:
        return f"<<<END-SOURCE-MATERIAL {self.nonce}>>>"

    def scrub(self, text: str) -> str:
        """Remove any occurrence of this fence's nonce from untrusted text."""
        if self.nonce in text:
            return text.replace(self.nonce, "[redacted]")
        return text


def make_fence() -> SourceFence:
    """Generate a fresh fence. Call once per request, never reuse."""
    return SourceFence(nonce=secrets.token_hex(4))


# ---------------------------------------------------------------------------
# Citation labels
# ---------------------------------------------------------------------------

_ACRONYM = re.compile(r"^[A-Z][A-Z0-9]*s?$")


def citation_label(metadata: dict) -> str:
    """
    Human-readable source label, e.g. "LLMs 04 Attention Transformers, p.41".

    Preserves acronyms (ML, LLM, BCNF, SP26) that naive title-casing would
    mangle into "Ml" and "Bcnf".
    """
    title = str(metadata.get("doc_title") or "").strip()
    if not title:
        stem = Path(str(metadata.get("filename") or "source")).stem
        words = re.sub(r"[-_]+", " ", stem).split()
        title = " ".join(w if _ACRONYM.match(w) else w.capitalize() for w in words)

    page = metadata.get("page_number")
    start, end = metadata.get("page_start"), metadata.get("page_end")

    if start and end and start != end:
        return f"{title}, pp.{start}-{end}"
    if page:
        return f"{title}, p.{page}"
    return title


# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------


@dataclass
class EvidenceBlock:
    """The formatted evidence for one request, plus the mapping back to sources."""

    text: str
    fence: SourceFence
    labels: list[str] = field(default_factory=list)
    """`labels[i]` is the citation label for [Source i+1]."""
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.labels)


def build_evidence(
    chunks: Sequence,
    fence: SourceFence | None = None,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
) -> EvidenceBlock:
    """
    Format retrieved chunks as numbered, fenced source blocks.

    Args:
        chunks: Objects exposing `.text`, `.metadata`, and optionally
            `.chunk_id` / `.relevance` (a `RerankedChunk`).
        fence: Fence to use. A fresh one is generated when omitted.

    Returns:
        An `EvidenceBlock` whose `labels` list maps [Source N] back to a
        citation string — the mapping the citation validator checks against.
    """
    fence = fence or make_fence()
    if not chunks:
        return EvidenceBlock(text="(no source material retrieved)", fence=fence)

    parts: list[str] = []
    labels: list[str] = []
    ids: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        meta = dict(getattr(chunk, "metadata", {}) or {})
        label = citation_label(meta)
        body = fence.scrub(str(getattr(chunk, "text", ""))[:max_chars])

        labels.append(label)
        ids.append(str(getattr(chunk, "chunk_id", f"chunk-{i}")))

        header = f"[Source {i}] {label}"
        section = str(meta.get("section_path") or "")
        if section:
            header += f" — section: {section}"
        # Flag OCR-derived text so the model can hedge on garbled passages.
        if float(meta.get("confidence", 1.0)) < 1.0:
            header += " — (recovered by OCR; text may contain errors)"

        parts.append(f"{header}\n{body}")

    return EvidenceBlock(
        text="\n\n".join(parts), fence=fence, labels=labels, chunk_ids=ids
    )


def build_evidence_summary(chunks: Sequence, *, max_chars: int = 700) -> str:
    """Compact evidence rendering for the verifier, which needs breadth over depth."""
    if not chunks:
        return "(no evidence retrieved)"
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = dict(getattr(chunk, "metadata", {}) or {})
        text = str(getattr(chunk, "text", ""))
        clipped = text[:max_chars] + ("…" if len(text) > max_chars else "")
        lines.append(f"[Source {i}] {citation_label(meta)}\n{clipped}")
    return "\n\n".join(lines)


def format_history(history: Sequence[dict] | None, *, max_turns: int = 6) -> str:
    """
    Render recent conversation turns as a prompt block.

    History is prior model output and student text — not retrieved documents —
    so it sits outside the source fence, but it is still untrusted and is
    labelled as context rather than instruction.
    """
    if not history:
        return ""
    recent = list(history)[-max_turns:]
    lines = ["<conversation_context>"]
    for msg in recent:
        role = "Student" if msg.get("role") == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content[:1500]}")
    lines.append("</conversation_context>")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# `is_refusal` and `strip_refusal_marker` are re-exported from
# guardrails.refusal so callers can import everything prompt-related from one
# place without agents and guardrails importing each other.


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json_response(raw: str) -> dict:
    """
    Parse a JSON object from a model response.

    Handles markdown fences and leading prose. Returns {} on failure so
    callers can fall back rather than raise.
    """
    import json

    if not raw:
        return {}

    cleaned = _JSON_FENCE.sub("", raw).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost brace-balanced span.
    start = cleaned.find("{")
    if start == -1:
        return {}
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start : i + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}
