"""
EduPilot Utilities
==================
Document loading, chunking, text preprocessing, and the shared LLM caller.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_MODEL,
    GEMINI_API_KEY,
    LLM_MAX_TOKENS_CLASSIFY,
    SUPPORTED_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single chunk extracted from a source document."""
    chunk_id: str
    text: str
    source_file: str
    domain: str
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk with retrieval scores attached."""
    chunk_id: str
    text: str
    source_file: str
    domain: str
    page_number: Optional[int]
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def citation_label(self) -> str:
        """Human-readable citation string, preserving acronyms like ML, LLMs, AML."""
        import re
        stem = Path(self.source_file).stem
        # Replace separators with spaces
        stem = re.sub(r"[-_]+", " ", stem)
        words = stem.split()
        formatted = []
        for w in words:
            # Preserve all-uppercase tokens (ML, AML, LLMs, SP26, BCNF…)
            if re.match(r'^[A-Z][A-Z0-9s]*$', w):
                formatted.append(w)
            else:
                formatted.append(w.capitalize())
        base = " ".join(formatted)
        if self.page_number:
            return f"{base}, p.{self.page_number}"
        return base


@dataclass
class DomainAnswer:
    """Grounded answer from one domain's RAG pipeline."""
    domain: str
    sub_question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    num_chunks_used: int = 0
    no_evidence: bool = False


@dataclass
class PipelineResult:
    """Full output of the EduPilot pipeline."""
    original_query: str
    intent_type: str                          # "single" | "multi"
    detected_domains: list[str]
    sub_questions: list[dict]                 # [{question, domain}]
    domain_answers: list[DomainAnswer]
    synthesized_answer: str
    final_answer: str
    is_satisfactory: bool
    quality_score: float
    verification_issues: list[str]
    is_course_related: bool
    needs_clarification: bool
    clarification_hint: str | None
    debug_info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def call_llm(
    messages: list[dict],
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = LLM_MAX_TOKENS_CLASSIFY,
) -> str:
    """
    Call the Gemini API and return the assistant text.
    Auto-retries on per-minute limits and falls back to next model on daily exhaustion.
    Raises on API errors so callers can handle gracefully.
    """
    import re as _re, sys as _sys, time as _time
    from config import AVAILABLE_MODELS

    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com and add it to your .env file."
        )

    client = genai.Client(api_key=api_key)

    # Build contents from OpenAI-style messages (user/assistant → user/model)
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=msg["content"])]))

    def _make_config(m: str) -> genai_types.GenerateContentConfig:
        kw: dict = dict(
            max_output_tokens=max_tokens,
            temperature=0.1,
            system_instruction=system,
            safety_settings=[
                genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
                genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
                genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        )
        if "2.5" in m:
            kw["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
        return genai_types.GenerateContentConfig(**kw)

    # Build model fallback list: requested model first, then remaining AVAILABLE_MODELS
    fallback_order = [model] + [m for m in AVAILABLE_MODELS if m != model]

    last_exc: Exception | None = None
    for current_model in fallback_order:
        config = _make_config(current_model)
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=current_model, contents=contents, config=config)
                if current_model != model:
                    print(f"[EduPilot] Fell back to {current_model} (original: {model})", file=_sys.stderr, flush=True)
                return response.text
            except Exception as exc:
                err = str(exc)
                last_exc = exc
                is_daily = "PerDay" in err or "PerDayPerProject" in err
                is_quota = "429" in err or "resource_exhausted" in err.lower()
                if is_daily:
                    # Daily quota exhausted for this model — try next model
                    print(f"[EduPilot] Daily quota exhausted for {current_model}, trying next model", file=_sys.stderr, flush=True)
                    break
                if is_quota and attempt < 2:
                    # Per-minute limit — wait and retry same model
                    delay_match = _re.search(r"retry in (\d+(?:\.\d+)?)s", err, _re.IGNORECASE)
                    wait = float(delay_match.group(1)) + 2 if delay_match else 15
                    print(f"[EduPilot] RPM limit on {current_model} — retrying in {wait:.0f}s", file=_sys.stderr, flush=True)
                    _time.sleep(wait)
                    continue
                raise  # non-quota error — propagate immediately

    raise last_exc  # all models exhausted


def parse_json_response(raw: str) -> dict:
    """
    Parse JSON from LLM response, stripping markdown fences if present.
    Returns empty dict on parse failure.
    """
    # Strip ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract a JSON object with regex
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def tokenize_simple(text: str) -> list[str]:
    """
    Fast whitespace + punctuation tokenizer for BM25.
    Avoids NLTK dependency.
    """
    return re.findall(r"\b\w+\b", text.lower())


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_text_file(path: str) -> str:
    """Load plain text or markdown file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(path: str) -> list[tuple[str, int]]:
    """
    Load PDF pages. Returns list of (page_text, page_number).
    Falls back to plain text read if PyMuPDF not installed.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages.append((text, i))
        return pages
    except ImportError:
        # Fallback: try reading as text
        try:
            text = load_text_file(path)
            return [(text, 1)]
        except Exception:
            return []


def load_docx_file(path: str) -> str:
    """Load .docx file. Returns empty string if python-docx not installed."""
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        return ""


def load_document(path: str) -> list[tuple[str, Optional[int]]]:
    """
    Load a document and return list of (text_segment, page_number).
    page_number is None for non-paginated formats.
    """
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return []

    if ext == ".pdf":
        return load_pdf_file(path)
    elif ext == ".docx":
        text = load_docx_file(path)
        return [(text, None)] if text else []
    else:  # .txt, .md
        text = load_text_file(path)
        return [(text, None)] if text else []


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source_file: str,
    domain: str,
    page_number: Optional[int] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    start_idx: int = 0,          # global offset so IDs are unique across pages
) -> list[DocumentChunk]:
    """
    Split text into overlapping word-based chunks.
    Returns DocumentChunk objects ready for indexing.
    """
    text = clean_text(text)
    words = text.split()
    if not words:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    idx = start_idx

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        if len(chunk_text_str.strip()) > 20:  # skip trivially short chunks
            chunk_id = f"{domain}_{Path(source_file).stem}_{idx:04d}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text_str,
                source_file=source_file,
                domain=domain,
                page_number=page_number,
                metadata={
                    "source_file": source_file,
                    "domain": domain,
                    "page_number": page_number,
                    "chunk_index": idx,
                },
            ))
            idx += 1

        if end >= len(words):
            break
        start = end - chunk_overlap

    return chunks


def load_and_chunk_file(
    file_path: str,
    domain: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Load a file and return all its chunks with globally unique IDs."""
    segments = load_document(file_path)
    all_chunks: list[DocumentChunk] = []
    global_idx = 0          # never resets — keeps IDs unique across all pages
    for text, page_num in segments:
        page_chunks = chunk_text(
            text=text,
            source_file=file_path,
            domain=domain,
            page_number=page_num,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_idx=global_idx,
        )
        global_idx += len(page_chunks)
        all_chunks.extend(page_chunks)
    return all_chunks


def load_domain_documents(domain_kb_path: str, domain: str) -> list[DocumentChunk]:
    """Load and chunk all supported documents in a domain folder."""
    kb_path = Path(domain_kb_path)
    if not kb_path.exists():
        return []

    all_chunks: list[DocumentChunk] = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in kb_path.glob(f"*{ext}"):
            chunks = load_and_chunk_file(str(file_path), domain)
            all_chunks.extend(chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Citation formatting helpers
# ---------------------------------------------------------------------------

def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into the prompt context string."""
    if not chunks:
        return "(No source material found.)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        citation = chunk.citation_label()
        block = textwrap.dedent(f"""\
            [Source {i}] — {citation}  (domain: {chunk.domain}, score: {chunk.rerank_score:.3f})
            {chunk.text}
        """)
        parts.append(block)

    return "\n---\n".join(parts)


def format_evidence_summary(domain_answers: list[DomainAnswer], max_chars_per_chunk: int = 500) -> str:
    """Compact evidence listing for the verifier prompt."""
    lines = []
    for da in domain_answers:
        lines.append(f"Domain {da.domain} — sub-question: '{da.sub_question}'")
        for i, chunk in enumerate(da.retrieved_chunks, 1):
            text = chunk.text
            truncated = text[:max_chars_per_chunk] + "…" if len(text) > max_chars_per_chunk else text
            lines.append(f"  [Src {i}] {chunk.citation_label()}: {truncated}")
    return "\n".join(lines) if lines else "(No evidence retrieved.)"


def build_sub_answers_block(domain_answers: list[DomainAnswer]) -> str:
    """Format domain answers for the synthesizer prompt."""
    parts = []
    for da in domain_answers:
        header = f"### [{da.domain}] Answer to: {da.sub_question}"
        parts.append(f"{header}\n\n{da.answer}")
    return "\n\n".join(parts)
