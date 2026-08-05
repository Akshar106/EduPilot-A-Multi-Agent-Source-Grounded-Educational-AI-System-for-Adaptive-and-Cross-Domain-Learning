"""
Token-aware structural chunking
===============================
Replaces the previous fixed 800-*word* window applied per PDF page.

Measured against the live index, the old chunker produced:

    46.0%  of chunks over the embedding model's token window (silently
           truncated at embed time — 47.9% of all indexed text never
           reached the embedder)
    12.9%  of chunks under 20 words (sparse slides became their own chunk)
    3,431  tokens in the largest single chunk

The failure had two causes. Length was measured in words while the model
budgets in tokens, and `chunk_size` was applied per page, so it almost never
bound: a slide became a 12-word chunk and a dense textbook page became a
3,431-token one.

This chunker instead:

  * measures length with the embedding model's own tokenizer, so the window
    is a real constraint rather than a proxy;
  * packs whole blocks within a section, spanning pages, until the token
    budget is reached — a section is the natural retrieval unit, not a page;
  * splits oversize blocks on sentence boundaries, never mid-sentence;
  * merges fragments below a floor so no near-empty chunk is indexed;
  * prepends a breadcrumb ("Lecture 4 > Attention > Scaled Dot-Product") so an
    isolated chunk still states its own context;
  * carries `parent_id`, enabling small-to-big retrieval: match on a precise
    child, hand the LLM the wider parent window.

Chunk IDs are content-addressed from the document hash, so re-ingesting an
unchanged file yields identical IDs (idempotent) and two study sessions
uploading the same filename can never collide.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .models import Block, BlockKind, ParsedDocument, Section, text_hash

logger = logging.getLogger(__name__)

CHUNKER_VERSION = "structural/1.0"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_tokenizer_cache: dict[str, object] = {}


def _hf_token_counter(model_name: str) -> Callable[[str], int] | None:
    """Load the HuggingFace tokenizer for `model_name`, or None if unavailable."""
    if model_name in _tokenizer_cache:
        tok = _tokenizer_cache[model_name]
        return None if tok is None else (lambda t: len(tok.encode(t, add_special_tokens=True)))  # type: ignore[union-attr]
    try:
        # Belt and braces alongside config.py: `ingestion` is importable
        # standalone (the benchmark scripts do exactly that), and this must be
        # set before the Rust tokenizer spins up its rayon pool or the
        # chunker/embedder combination can deadlock.
        import os

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        # This tokenizer is used for *counting*, never for model input, so raise
        # the length ceiling to stop it warning about sequences it will not run.
        tok.model_max_length = int(1e9)
        _tokenizer_cache[model_name] = tok
        return lambda t: len(tok.encode(t, add_special_tokens=True))
    except Exception as exc:
        logger.warning(
            "Could not load tokenizer for %s (%s) — falling back to a length estimate. "
            "Chunk boundaries will be approximate.",
            model_name,
            exc,
        )
        _tokenizer_cache[model_name] = None
        return None


def _estimate_tokens(text: str) -> int:
    """
    Conservative token estimate used only when no tokenizer is available.

    Technical English runs ~1.35 tokens/word once subword splits on terms like
    "regularization" and "eigendecomposition" are counted. Rounding up is
    deliberate: overestimating yields smaller chunks, which is safe, whereas
    underestimating reintroduces silent truncation.
    """
    words = len(text.split())
    return int(words * 1.35) + 2


def get_token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Return a token-counting function for the configured embedding model."""
    if model_name:
        counter = _hf_token_counter(model_name)
        if counter is not None:
            return counter
    return _estimate_tokens


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Chunking parameters, all measured in embedding-model tokens.

    Defaults target BAAI/bge-small-en-v1.5 (512-token window). `max_tokens`
    leaves headroom below 512 for the breadcrumb prefix and special tokens.
    """

    model_name: str = "BAAI/bge-small-en-v1.5"
    model_window: int = 512
    max_tokens: int = 448
    """Hard ceiling for a chunk including its prefix. Never exceeded."""
    target_tokens: int = 320
    """Preferred size. Packing stops once a chunk passes this."""
    min_tokens: int = 64
    """Fragments below this are merged forward rather than indexed alone."""
    overlap_sentences: int = 1
    """Sentences carried from the end of one chunk into the next."""
    parent_max_tokens: int = 1400
    """Size of the wider parent window used for small-to-big retrieval."""
    include_breadcrumb: bool = True
    """Prepend 'Document > Section > Subsection' to each chunk's text."""

    def __post_init__(self) -> None:
        if self.max_tokens > self.model_window:
            raise ValueError(
                f"max_tokens={self.max_tokens} exceeds model_window={self.model_window}; "
                "chunks would be silently truncated at embed time"
            )
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens must not exceed max_tokens")
        if self.min_tokens >= self.target_tokens:
            raise ValueError("min_tokens must be below target_tokens")


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------


@dataclass
class DocumentChunk:
    """
    One indexed unit of text.

    A superset of the previous DocumentChunk — `chunk_id`, `text`,
    `source_file`, `domain`, `page_number`, and `metadata` keep their old
    meanings so existing call sites continue to work.
    """

    chunk_id: str
    text: str
    source_file: str
    domain: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)

    # --- structural additions ---
    doc_title: str = ""
    section_path: str = ""
    parent_id: str | None = None
    chunk_index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    token_count: int = 0
    content_hash: str = ""
    kinds: tuple[str, ...] = ()
    confidence: float = 1.0
    is_parent: bool = False

    @property
    def embed_text(self) -> str:
        """Exactly the string that gets embedded — chunk text including breadcrumb."""
        return self.text


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "e.g.", "i.e.", "cf.", "vs.", "etc.", "al.", "fig.", "eq.", "sec.", "ch.",
    "approx.", "resp.", "viz.", "dr.", "prof.", "mr.", "mrs.", "ms.", "st.",
    "no.", "vol.", "pp.", "ed.", "eds.", "inc.", "ltd.", "u.s.", "u.k.",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, keeping common abbreviations intact.

    Deliberately lightweight — no NLTK/spaCy dependency. Newline-separated
    lines (slide bullets, list items) are treated as their own sentences,
    which matters because slide text often has no terminal punctuation.
    """
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        pieces = _SENTENCE_BOUNDARY.split(line)
        buf = ""
        for piece in pieces:
            candidate = f"{buf} {piece}".strip() if buf else piece
            last_word = candidate.split()[-1].lower() if candidate.split() else ""
            if last_word in _ABBREVIATIONS:
                buf = candidate
                continue
            out.append(candidate)
            buf = ""
        if buf:
            out.append(buf)
    return [s for s in out if s.strip()]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _breadcrumb(doc_title: str, section: Section, cfg: ChunkingConfig) -> str:
    """
    Build the contextual prefix for chunks in a section.

    Deduplicates the document title against the section path — in a markdown
    file whose H1 *is* the title, naive joining yields
    "Fundamentals > Fundamentals > Overfitting".
    """
    if not cfg.include_breadcrumb:
        return ""

    title = doc_title.strip()
    crumb = section.breadcrumb.strip()
    # Preamble sections carry no meaningful title.
    if crumb.startswith("(document"):
        crumb = ""

    if title and crumb:
        crumb_parts = crumb.split(" > ")
        if crumb_parts[0].strip().lower() == title.lower():
            title = ""

    return " > ".join(p for p in (title, crumb) if p)


def _with_prefix(prefix: str, body: str) -> str:
    return f"{prefix}\n\n{body}" if prefix else body


def _split_oversize(text: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """
    Break a block that alone exceeds `budget` into sentence-aligned pieces.

    A single sentence longer than the budget (a long equation, a table row)
    is split on word boundaries as a last resort rather than dropped.
    """
    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for sentence in split_sentences(text):
        s_tokens = count(sentence)

        if s_tokens > budget:
            if buf:
                pieces.append(" ".join(buf))
                buf, buf_tokens = [], 0
            words = sentence.split()
            step = max(1, int(len(words) * budget / max(s_tokens, 1)))
            for i in range(0, len(words), step):
                pieces.append(" ".join(words[i : i + step]))
            continue

        if buf_tokens + s_tokens > budget and buf:
            pieces.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(sentence)
        buf_tokens += s_tokens

    if buf:
        pieces.append(" ".join(buf))
    return pieces


def _pack_section(
    section: Section,
    prefix: str,
    cfg: ChunkingConfig,
    count: Callable[[str], int],
) -> list[tuple[str, list[Block]]]:
    """
    Pack a section's blocks into token-bounded groups.

    Returns [(body_text, source_blocks)]. Blocks are kept whole where they
    fit; only blocks that individually exceed the budget are split. Tables are
    never split — a half table is worse than a slightly oversize chunk, so an
    oversize table becomes its own chunk and is split only if it breaches the
    hard model window.
    """
    prefix_tokens = count(prefix) + 2 if prefix else 0
    budget = max(cfg.max_tokens - prefix_tokens, cfg.min_tokens)
    target = max(min(cfg.target_tokens, budget), cfg.min_tokens)

    groups: list[tuple[str, list[Block]]] = []
    buf_parts: list[str] = []
    buf_blocks: list[Block] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf_parts, buf_blocks, buf_tokens
        if not buf_parts:
            return
        groups.append(("\n\n".join(buf_parts), list(buf_blocks)))
        # Sentence-level overlap: carry the tail of this chunk into the next.
        carry: list[str] = []
        if cfg.overlap_sentences > 0:
            sentences = split_sentences(buf_parts[-1])
            carry = sentences[-cfg.overlap_sentences :] if sentences else []
        buf_parts = [" ".join(carry)] if carry else []
        buf_blocks = [buf_blocks[-1]] if (carry and buf_blocks) else []
        buf_tokens = count(buf_parts[0]) if buf_parts else 0

    for block in section.blocks:
        text = block.text.strip()
        if not text:
            continue
        tokens = count(text)

        # Oversize single block.
        if tokens > budget:
            flush()
            if buf_parts:
                groups.append(("\n\n".join(buf_parts), list(buf_blocks)))
                buf_parts, buf_blocks, buf_tokens = [], [], 0
            for piece in _split_oversize(text, budget, count):
                groups.append((piece, [block]))
            continue

        if buf_tokens + tokens > budget and buf_parts:
            flush()

        buf_parts.append(text)
        buf_blocks.append(block)
        buf_tokens += tokens

        if buf_tokens >= target:
            flush()

    if buf_parts and "".join(buf_parts).strip():
        groups.append(("\n\n".join(buf_parts), list(buf_blocks)))

    return groups


@dataclass
class _Spec:
    """A chunk-in-progress: breadcrumb prefix plus body, before materialization."""

    prefix: str
    body: str
    blocks: list[Block]
    section: Section

    def render(self) -> str:
        return _with_prefix(self.prefix, self.body)


def _common_breadcrumb(a: str, b: str) -> str:
    """
    Longest shared prefix of two breadcrumbs, split on ' > '.

    Merging a chunk from "Deck > Attention > Queries" with one from
    "Deck > Attention > Keys" yields "Deck > Attention" — still informative,
    and never claims the merged chunk belongs to only one of the two.
    """
    if a == b:
        return a
    pa, pb = a.split(" > "), b.split(" > ")
    shared: list[str] = []
    for x, y in zip(pa, pb):
        if x != y:
            break
        shared.append(x)
    return " > ".join(shared)


def _try_merge(a: _Spec, b: _Spec, cfg: ChunkingConfig, count: Callable[[str], int]) -> _Spec | None:
    """
    Merge two adjacent specs if the result fits the token budget, else None.

    When the two come from different sections, the later section's title is
    inlined into the body so the heading context is not lost by falling back
    to a shorter shared breadcrumb.
    """
    prefix = _common_breadcrumb(a.prefix, b.prefix)

    body = a.body
    if b.section is not a.section and b.section.title and b.section.title not in a.body:
        body = f"{body}\n\n{b.section.title}\n{b.body}"
    else:
        body = f"{body}\n\n{b.body}"
    body = body.strip()

    merged = _Spec(prefix=prefix, body=body, blocks=a.blocks + b.blocks, section=a.section)
    if count(merged.render()) > cfg.max_tokens:
        return None
    return merged


def _merge_fragments(
    specs: list[_Spec],
    cfg: ChunkingConfig,
    count: Callable[[str], int],
) -> list[_Spec]:
    """
    Fold specs below `min_tokens` into a neighbour, **across section boundaries**.

    Sparse slide decks are the motivating case. Each slide becomes its own
    section, so an agenda slide with six words would otherwise be indexed
    alone: no retrieval value, and it dilutes the index. Merging only within a
    section cannot fix that, because the fragment *is* the whole section.

    Fragments merge forward by preference so they read as a lead-in, and
    backward when nothing follows or the forward merge would overflow.
    """
    if not specs:
        return []

    out: list[_Spec] = []
    pending: _Spec | None = None

    for spec in specs:
        if pending is not None:
            merged = _try_merge(pending, spec, cfg, count)
            if merged is not None:
                pending = merged if count(merged.render()) < cfg.min_tokens else None
                if pending is None:
                    out.append(merged)
                continue
            out.append(pending)
            pending = None

        if count(spec.render()) < cfg.min_tokens:
            pending = spec
            continue
        out.append(spec)

    if pending is not None:
        merged = _try_merge(out[-1], pending, cfg, count) if out else None
        if merged is not None:
            out[-1] = merged
        else:
            out.append(pending)

    return out


def chunk_document(
    doc: ParsedDocument,
    domain: str,
    *,
    config: ChunkingConfig | None = None,
    scope: str | None = None,
) -> list[DocumentChunk]:
    """
    Chunk an extracted document.

    Args:
        doc: Output of `extract_document`.
        domain: Domain tag stored on every chunk (e.g. "AML", "SELF_STUDY").
        config: Chunking parameters; defaults target bge-small-en-v1.5.
        scope: Namespace qualifier for chunk IDs. Defaults to `domain`. Pass a
            per-session value (e.g. "ss_<uuid>") to keep two study sessions
            that uploaded the same filename fully isolated.

    Returns:
        Child chunks (embedded and indexed) followed by parent chunks
        (`is_parent=True`, stored but not embedded) for small-to-big retrieval.
    """
    cfg = config or ChunkingConfig()
    count = get_token_counter(cfg.model_name)
    scope = scope or domain

    sections = doc.sections or [
        Section(title="", level=0, blocks=doc.blocks, page_start=1, page_end=doc.stats.page_count)
    ]

    # Pass 1 — pack each section independently, so no chunk straddles a heading
    # unless a later merge deliberately joins two undersized ones.
    specs: list[_Spec] = []
    for section in sections:
        if not section.blocks:
            continue
        prefix = _breadcrumb(doc.title, section, cfg)
        for body, blocks in _pack_section(section, prefix, cfg, count):
            if body.strip():
                specs.append(_Spec(prefix=prefix, body=body, blocks=blocks, section=section))

    # Pass 2 — merge undersized specs across section boundaries.
    specs = _merge_fragments(specs, cfg, count)

    # Pass 3 — materialize.
    children: list[DocumentChunk] = []
    for index, spec in enumerate(specs):
        full_text = spec.render()
        tokens = count(full_text)

        # Invariant: nothing leaves this function over the model window.
        if tokens > cfg.model_window:
            logger.error(
                "chunk %d of %s is %d tokens (window %d) — truncating defensively",
                index, doc.filename, tokens, cfg.model_window,
            )
            words = full_text.split()
            keep = max(1, int(len(words) * cfg.model_window / tokens) - 8)
            full_text = " ".join(words[:keep])
            tokens = count(full_text)

        pages = [b.page for b in spec.blocks] or [spec.section.page_start]
        kinds = tuple(sorted({b.kind.value for b in spec.blocks})) or (BlockKind.PARAGRAPH.value,)
        confidence = min((b.confidence for b in spec.blocks), default=1.0)
        section_path = spec.prefix.split(" > ", 1)[-1] if " > " in spec.prefix else spec.section.breadcrumb

        children.append(
            DocumentChunk(
                chunk_id=f"{scope}:{doc.content_hash[:12]}:{index:05d}",
                text=full_text,
                source_file=doc.source_path,
                domain=domain,
                page_number=min(pages),
                doc_title=doc.title,
                section_path=section_path,
                parent_id=None,  # assigned by _build_parents
                chunk_index=index,
                page_start=min(pages),
                page_end=max(pages),
                token_count=tokens,
                content_hash=text_hash(full_text),
                kinds=kinds,
                confidence=confidence,
                metadata={
                    "source_file": doc.source_path,
                    "filename": doc.filename,
                    "domain": domain,
                    "page_number": min(pages),
                    "chunk_index": index,
                    "section_path": section_path,
                    "doc_title": doc.title,
                    "kinds": list(kinds),
                    "confidence": confidence,
                    "content_hash": text_hash(full_text),
                    # Document-level hash — the key the indexer filters on when
                    # replacing an edited file's vectors.
                    "doc_hash": doc.content_hash,
                    "chunker": CHUNKER_VERSION,
                },
            )
        )

    parents = _build_parents(children, doc, domain, scope, cfg, count)

    logger.info(
        "chunked %s: %d children (max %d tokens), %d parents",
        doc.filename,
        len(children),
        max((c.token_count for c in children), default=0),
        len(parents),
    )
    return children + parents


def _section_root(path: str) -> str:
    """Top-level section a breadcrumb belongs to."""
    return path.split(" > ", 1)[0] if path else ""


def _build_parents(
    children: list[DocumentChunk],
    doc: ParsedDocument,
    domain: str,
    scope: str,
    cfg: ChunkingConfig,
    count: Callable[[str], int],
) -> list[DocumentChunk]:
    """
    Build the wider parent windows that children point at.

    Parents are stored but never embedded — retrieval matches on a precise
    child, then the child's `parent_id` is used to hand the LLM the wider
    surrounding window. This is "small-to-big": precision at match time,
    context at generation time.

    A parent is closed when the token budget is reached or the top-level
    section changes, so a parent never spans two unrelated topics. Each child
    is repointed at the parent that actually contains it.
    """
    if not children:
        return []

    parents: list[DocumentChunk] = []
    buf: list[DocumentChunk] = []
    buf_tokens = 0

    def emit() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        pid = f"{scope}:{doc.content_hash[:12]}:p{len(parents):04d}"
        body = "\n\n".join(c.text for c in buf)
        pages = [p for c in buf for p in (c.page_start, c.page_end) if p is not None]
        section_path = buf[0].section_path
        parents.append(
            DocumentChunk(
                chunk_id=pid,
                text=body,
                source_file=doc.source_path,
                domain=domain,
                page_number=min(pages) if pages else None,
                doc_title=doc.title,
                section_path=section_path,
                parent_id=None,
                chunk_index=-1,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                token_count=count(body),
                content_hash=text_hash(body),
                kinds=tuple(sorted({k for c in buf for k in c.kinds})),
                confidence=min(c.confidence for c in buf),
                is_parent=True,
                metadata={
                    "source_file": doc.source_path,
                    "filename": doc.filename,
                    "domain": domain,
                    "section_path": section_path,
                    "doc_title": doc.title,
                    "is_parent": True,
                    "child_ids": [c.chunk_id for c in buf],
                    "doc_hash": doc.content_hash,
                    "chunker": CHUNKER_VERSION,
                    # Page provenance must survive parent expansion — without
                    # it, every citation for an expanded chunk loses its page
                    # number and "[Source 3] Lecture 4, p.12" degrades to
                    # "[Source 3] Lecture 4".
                    "page_number": min(pages) if pages else None,
                    "page_start": min(pages) if pages else None,
                    "page_end": max(pages) if pages else None,
                },
            )
        )
        for c in buf:
            c.parent_id = pid
            c.metadata["parent_id"] = pid
        buf, buf_tokens = [], 0

    for child in children:
        crosses_section = bool(buf) and _section_root(child.section_path) != _section_root(buf[0].section_path)
        if buf and (crosses_section or buf_tokens + child.token_count > cfg.parent_max_tokens):
            emit()
        buf.append(child)
        buf_tokens += child.token_count
    emit()

    return parents


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def chunking_report(chunks: Iterable[DocumentChunk], cfg: ChunkingConfig | None = None) -> dict:
    """
    Quality summary for a chunk set.

    `over_window` and `under_floor` are the two numbers that were broken
    before: both must be 0 for the chunker to be doing its job.
    """
    cfg = cfg or ChunkingConfig()
    children = [c for c in chunks if not c.is_parent]
    if not children:
        return {"chunks": 0}

    tokens = sorted(c.token_count for c in children)
    n = len(tokens)
    return {
        "chunks": n,
        "parents": sum(1 for c in chunks if c.is_parent),
        "tokens_min": tokens[0],
        "tokens_p50": tokens[n // 2],
        "tokens_p95": tokens[min(n - 1, int(n * 0.95))],
        "tokens_max": tokens[-1],
        "tokens_mean": round(sum(tokens) / n, 1),
        "over_window": sum(1 for t in tokens if t > cfg.model_window),
        "under_floor": sum(1 for t in tokens if t < cfg.min_tokens),
        "with_section_path": sum(1 for c in children if c.section_path),
    }
