"""
Upload validation
=================
Everything applied to a file before it touches the filesystem.

The previous upload handlers wrote attacker-controlled filenames directly::

    dest = kb_path / uf.filename        # main.py:412
    dest = upload_dir / uf.filename     # main.py:784

`uf.filename` comes from the multipart request and was never sanitized.
Because `pathlib` lets an absolute path replace the base entirely,
``Path("/kb/aml") / "/etc/cron.d/job"`` evaluates to ``/etc/cron.d/job`` — an
arbitrary file write as the server user. A relative traversal
(``../../../.ssh/authorized_keys``) escapes just as easily. The *download*
route already sanitized with `Path(filename).name`; only the write paths were
missed.

Also enforced here, none of which existed before: a size cap, an extension
allowlist, content-sniffing so the declared extension must match the actual
bytes, and a scan of extracted text for prompt injection before the document
is indexed.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Extensions the ingestion pipeline can parse.
ALLOWED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".docx"})

#: Per-file cap. The largest legitimate document in the corpus is an 851-page
#: textbook at roughly 30 MB, so 64 MB is generous with room to spare.
MAX_FILE_BYTES = 64 * 1024 * 1024

#: Cap on a single upload request.
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_FILES_PER_REQUEST = 20

#: Magic bytes by extension. A `.pdf` whose bytes are a ZIP archive is either
#: a mistake or an attempt to smuggle content past the extension check.
MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".docx": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
}

#: Windows reserved device names — writing these can hang or misbehave.
_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ \-()\[\]]+")
_DOT_RUN = re.compile(r"\.{2,}")
_SPACE_RUN = re.compile(r"\s{2,}")

MAX_FILENAME_LENGTH = 120


class UploadRejected(ValueError):
    """An upload failed validation. The message is safe to show a user."""


@dataclass
class ValidatedUpload:
    """A file that passed every check, with the name it should be stored under."""

    original_name: str
    safe_name: str
    extension: str
    size_bytes: int
    content: bytes = field(repr=False, default=b"")
    warnings: list[str] = field(default_factory=list)


def safe_filename(raw: str) -> str:
    """
    Reduce an arbitrary filename to a single safe path component.

    Every traversal vector is neutralized:

    - `Path(raw).name` discards directories and defeats both `../` and an
      absolute path
    - NFKC folding prevents Unicode lookalikes from surviving the character
      filter
    - null bytes and control characters are stripped (a C-level path
      truncation trick)
    - leading dots are removed so nothing writes a hidden file
    - Windows reserved device names are prefixed

    Raises:
        UploadRejected: nothing usable remains after sanitization.
    """
    if not raw or not raw.strip():
        raise UploadRejected("Filename is empty.")

    # Strip null bytes and control characters before any path handling.
    cleaned = "".join(ch for ch in raw if ord(ch) >= 32 and ch != "\x7f")
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # Discard any directory component. Handles "../x", "/etc/x", "C:\\x".
    cleaned = cleaned.replace("\\", "/")
    cleaned = Path(cleaned).name

    cleaned = _UNSAFE_CHARS.sub("_", cleaned)
    cleaned = _DOT_RUN.sub(".", cleaned)
    cleaned = _SPACE_RUN.sub(" ", cleaned).strip(" .")

    if not cleaned:
        raise UploadRejected("Filename contains no usable characters.")

    stem, dot, ext = cleaned.rpartition(".")
    if not dot:
        stem, ext = cleaned, ""
    if stem.lower() in _RESERVED:
        stem = f"file_{stem}"

    if len(stem) > MAX_FILENAME_LENGTH:
        stem = stem[:MAX_FILENAME_LENGTH]

    result = f"{stem}.{ext}" if ext else stem
    if result != raw:
        logger.debug("sanitized filename %r -> %r", raw, result)
    return result


def resolve_within(base_dir: Path, filename: str) -> Path:
    """
    Resolve `filename` inside `base_dir`, refusing anything that escapes.

    Belt and braces: `safe_filename` should already make escape impossible,
    but this verifies the resolved path is genuinely under `base_dir` — which
    also catches escapes via a symlink inside the upload directory.

    Raises:
        UploadRejected: the resolved path lies outside `base_dir`.
    """
    base = base_dir.resolve()
    target = (base / safe_filename(filename)).resolve()
    if not target.is_relative_to(base):
        logger.error("path traversal blocked: %r resolved outside %s", filename, base)
        raise UploadRejected("Invalid filename.")
    return target


def _sniff_matches(extension: str, content: bytes) -> bool:
    """True when the file's magic bytes match its declared extension."""
    signatures = MAGIC_SIGNATURES.get(extension)
    if not signatures:
        return True  # text formats have no reliable signature
    return any(content.startswith(sig) for sig in signatures)


def validate_upload(
    filename: str,
    content: bytes,
    *,
    max_bytes: int = MAX_FILE_BYTES,
    allowed: frozenset[str] = ALLOWED_EXTENSIONS,
    scan_injection: bool = True,
) -> ValidatedUpload:
    """
    Validate one uploaded file.

    Args:
        filename: Client-supplied name. Treated as hostile.
        content: The file bytes.
        scan_injection: Scan a sample of the bytes for prompt-injection
            patterns and attach a warning. Never rejects on this alone — a
            legitimate lecture on prompt injection would trip it.

    Returns:
        A `ValidatedUpload` carrying the sanitized `safe_name`.

    Raises:
        UploadRejected: with a message safe to return to the client.
    """
    safe = safe_filename(filename)
    extension = Path(safe).suffix.lower()

    if extension not in allowed:
        raise UploadRejected(
            f"'{extension or 'no extension'}' files are not supported. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )

    size = len(content)
    if size == 0:
        raise UploadRejected("File is empty.")
    if size > max_bytes:
        raise UploadRejected(
            f"File is {size / 1e6:.1f} MB, over the {max_bytes / 1e6:.0f} MB limit."
        )

    if not _sniff_matches(extension, content):
        raise UploadRejected(
            f"File content does not match its '{extension}' extension. "
            "It may be corrupted or mislabelled."
        )

    warnings: list[str] = []
    if scan_injection:
        from guardrails.injection import scan

        # Sample the head; injections are placed where a model will read them,
        # and scanning 64 MB of PDF binary is pointless.
        sample = content[:200_000].decode("utf-8", errors="ignore")
        report = scan(sample, source=safe)
        if report.is_suspicious:
            warnings.append(
                "This document contains text resembling instructions to the AI. "
                "It has been indexed, but its content is treated strictly as data."
            )
            logger.warning(
                "upload %r flagged for injection patterns: %s",
                safe, [m["pattern"] for m in report.matches],
            )

    return ValidatedUpload(
        original_name=filename,
        safe_name=safe,
        extension=extension,
        size_bytes=size,
        content=content,
        warnings=warnings,
    )


def validate_batch(
    files: list[tuple[str, bytes]],
    *,
    max_files: int = MAX_FILES_PER_REQUEST,
    max_total: int = MAX_TOTAL_BYTES,
) -> list[ValidatedUpload]:
    """
    Validate a multi-file upload, enforcing per-request limits.

    Raises:
        UploadRejected: on the first failure, so a partially-valid batch is
            never half-written.
    """
    if not files:
        raise UploadRejected("No files provided.")
    if len(files) > max_files:
        raise UploadRejected(f"Too many files ({len(files)}); the limit is {max_files}.")

    total = sum(len(content) for _, content in files)
    if total > max_total:
        raise UploadRejected(
            f"Upload totals {total / 1e6:.1f} MB, over the {max_total / 1e6:.0f} MB limit."
        )

    validated: list[ValidatedUpload] = []
    seen: set[str] = set()
    for name, content in files:
        item = validate_upload(name, content)
        if item.safe_name in seen:
            raise UploadRejected(f"Duplicate filename in this upload: {item.safe_name}")
        seen.add(item.safe_name)
        validated.append(item)
    return validated
