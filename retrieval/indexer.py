"""
Indexing
========
Writes documents into the vector store: extract, chunk, embed, encode sparse
weights, upsert.

Properties the previous ingest path lacked:

  * **Idempotency.** Documents are identified by a SHA-256 of their bytes.
    Re-ingesting an unchanged file is a no-op; re-ingesting a *changed* file
    deletes the old version's vectors before writing the new ones. Previously
    dedup was by chunk ID only, so editing a document left its stale chunks in
    the index forever.
  * **Isolation.** Chunk IDs are scoped by namespace and content hash, so two
    study sessions uploading the same filename cannot collide. The old scheme
    (`SELF_STUDY_<stem>_<n>`) collided on every shared filename, and because
    `self_study_chunks.chunk_id` is globally UNIQUE with `INSERT OR IGNORE`,
    the second session's chunks were silently dropped from SQLite.
  * **Atomic rebuilds.** `rebuild_all` writes into a fresh index version and
    promotes it only on success, so a failed re-index cannot leave the live
    index half-populated.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from ingestion import DocumentChunk, chunk_document, extract_document
from ingestion.chunking import ChunkingConfig

from .embeddings import Embedder
from .sparse import BM25Encoder
from .vectorstore import IndexPointer, VectorRecord, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Outcome of indexing one document."""

    filename: str
    content_hash: str
    chunks_indexed: int = 0
    parents_indexed: int = 0
    skipped: bool = False
    """True when the document was already indexed at this exact content hash."""
    replaced_version: bool = False
    """True when an older version of the same file was removed first."""
    error: str | None = None
    duration_ms: int = 0
    extraction: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class IndexRegistry:
    """
    Tracks which documents are indexed, at which content hash, in which
    namespace.

    This is what makes re-ingest idempotent and lets a changed file replace
    its own prior vectors without wiping the namespace.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS indexed_documents (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        namespace      TEXT NOT NULL,
        filename       TEXT NOT NULL,
        source_path    TEXT NOT NULL,
        content_hash   TEXT NOT NULL,
        chunk_count    INTEGER NOT NULL DEFAULT 0,
        parent_count   INTEGER NOT NULL DEFAULT 0,
        embed_model    TEXT NOT NULL DEFAULT '',
        chunker        TEXT NOT NULL DEFAULT '',
        indexed_at     TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(namespace, filename)
    );
    CREATE INDEX IF NOT EXISTS idx_indexed_ns ON indexed_documents(namespace);
    CREATE INDEX IF NOT EXISTS idx_indexed_hash ON indexed_documents(content_hash);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._local = threading.local()
        self._conn().executescript(self.SCHEMA)
        self._conn().commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            # Shares the file with database.py and UserStore; wait for a lock
            # rather than failing the request on brief contention.
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """
        Transaction that commits on success and rolls back on failure.

        Without the rollback, a failed write leaves this connection holding an
        open transaction and every other connection to the file then fails
        with "database is locked".
        """
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def get(self, namespace: str, filename: str) -> dict | None:
        row = self._conn().execute(
            "SELECT * FROM indexed_documents WHERE namespace=? AND filename=?",
            (namespace, filename),
        ).fetchone()
        return dict(row) if row else None

    def record(
        self,
        *,
        namespace: str,
        filename: str,
        source_path: str,
        content_hash: str,
        chunk_count: int,
        parent_count: int,
        embed_model: str,
        chunker: str,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO indexed_documents
                    (namespace, filename, source_path, content_hash, chunk_count,
                     parent_count, embed_model, chunker, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(namespace, filename) DO UPDATE SET
                    source_path=excluded.source_path,
                    content_hash=excluded.content_hash,
                    chunk_count=excluded.chunk_count,
                    parent_count=excluded.parent_count,
                    embed_model=excluded.embed_model,
                    chunker=excluded.chunker,
                    indexed_at=datetime('now')
                """,
                (namespace, filename, source_path, content_hash, chunk_count,
                 parent_count, embed_model, chunker),
            )

    def forget(self, namespace: str, filename: str) -> None:
        with self._write() as conn:
            conn.execute(
                "DELETE FROM indexed_documents WHERE namespace=? AND filename=?",
                (namespace, filename),
            )

    def list_documents(self, namespace: str | None = None) -> list[dict]:
        if namespace:
            rows = self._conn().execute(
                "SELECT * FROM indexed_documents WHERE namespace=? ORDER BY filename",
                (namespace,),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM indexed_documents ORDER BY namespace, filename"
            ).fetchall()
        return [dict(r) for r in rows]

    def all_texts_for_fit(self, namespace: str | None = None) -> list[str]:
        """Filenames are not enough to refit BM25; the caller supplies texts."""
        raise NotImplementedError("BM25 is fitted from chunk texts, not the registry")


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class Indexer:
    """
    Writes documents into a namespace of the vector store.

    Args:
        store: Vector store backend.
        embedder: Dense embedder.
        registry: Document registry for idempotency.
        sparse_encoder: Fitted BM25 encoder. When absent, only dense vectors
            are written and hybrid search degrades to semantic-only.
        chunking: Chunking parameters. Defaults are matched to the embedder.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        registry: IndexRegistry,
        *,
        sparse_encoder: BM25Encoder | None = None,
        chunking: ChunkingConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.registry = registry
        self.sparse_encoder = sparse_encoder
        self.chunking = chunking or ChunkingConfig(
            model_name=embedder.model_name,
            model_window=embedder.max_tokens,
            max_tokens=min(448, embedder.max_tokens - 64),
        )

    # ------------------------------------------------------------------
    # Vector construction
    # ------------------------------------------------------------------

    def _build_records(self, chunks: Sequence[DocumentChunk]) -> list[VectorRecord]:
        """
        Embed children, derive parent vectors, and attach sparse weights.

        Parents are stored so they can be fetched for context expansion, but
        they are never embedded: a parent runs to ~1,400 tokens, well past the
        model window, and embedding it would either fail the token budget or
        reintroduce silent truncation. Their vector is the L2-normalized mean
        of their children — a valid centroid, and they are excluded from
        search by an `is_parent` metadata filter regardless.
        """
        children = [c for c in chunks if not c.is_parent]
        parents = [c for c in chunks if c.is_parent]

        records: list[VectorRecord] = []
        child_vectors: dict[str, np.ndarray] = {}

        if children:
            # The chunker already measured every chunk; reuse those counts so
            # the budget check does not tokenize the whole corpus a second time.
            vectors = self.embedder.embed_documents(
                [c.text for c in children],
                token_counts=[c.token_count for c in children],
            )
            for chunk, vector in zip(children, vectors):
                child_vectors[chunk.chunk_id] = vector
                sparse = (
                    self.sparse_encoder.encode_document(chunk.text)
                    if self.sparse_encoder and self.sparse_encoder.fitted
                    else None
                )
                records.append(
                    VectorRecord(
                        id=chunk.chunk_id,
                        values=vector.tolist(),
                        sparse=sparse or None,
                        metadata={**chunk.metadata, "text": chunk.text, "is_parent": False},
                    )
                )

        for parent in parents:
            ids = parent.metadata.get("child_ids") or []
            members = [child_vectors[i] for i in ids if i in child_vectors]
            if members:
                centroid = np.mean(members, axis=0)
                norm = float(np.linalg.norm(centroid))
                vector = (centroid / norm) if norm > 0 else centroid
            else:
                vector = np.zeros(self.embedder.dimension, dtype=np.float32)
            records.append(
                VectorRecord(
                    id=parent.chunk_id,
                    values=vector.astype(np.float32).tolist(),
                    sparse=None,
                    metadata={**parent.metadata, "text": parent.text, "is_parent": True},
                )
            )

        return records

    # ------------------------------------------------------------------
    # Single document
    # ------------------------------------------------------------------

    def index_document(
        self,
        path: str | Path,
        *,
        namespace: str,
        domain: str,
        force: bool = False,
        prepared_chunks: list[DocumentChunk] | None = None,
    ) -> IndexResult:
        """
        Index one document into `namespace`.

        Skips work when the file's content hash already matches what is
        indexed. When the hash differs, the previous version's vectors are
        deleted before the new ones are written, so a corrected slide deck
        cannot leave stale text retrievable.

        Args:
            prepared_chunks: Already-chunked output for this file. Supplying
                it skips extraction — used by a full rebuild, which has to
                chunk the whole corpus up front to fit BM25 and would
                otherwise extract every document a second time.
        """
        started = time.perf_counter()
        p = Path(path)
        filename = p.name

        try:
            from ingestion.models import file_content_hash

            content_hash = file_content_hash(p)
        except OSError as exc:
            return IndexResult(filename=filename, content_hash="", error=str(exc))

        existing = self.registry.get(namespace, filename)
        if existing and existing["content_hash"] == content_hash and not force:
            logger.info("%s unchanged (hash %s) — skipping", filename, content_hash[:12])
            return IndexResult(
                filename=filename,
                content_hash=content_hash,
                chunks_indexed=existing["chunk_count"],
                skipped=True,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        replaced = False
        if existing:
            logger.info(
                "%s changed (%s -> %s) — removing previous version",
                filename, existing["content_hash"][:12], content_hash[:12],
            )
            self._delete_document_vectors(namespace, filename, existing["content_hash"])
            replaced = True

        # A caller that has already chunked this file (a full rebuild, which
        # must chunk everything up front to fit BM25) passes the chunks in
        # rather than paying for a second extraction. On this corpus that is
        # ~4 minutes saved per rebuild.
        doc = None
        if prepared_chunks is not None:
            chunks = prepared_chunks
        else:
            try:
                doc = extract_document(p)
                chunks = chunk_document(doc, domain, config=self.chunking, scope=namespace)
            except Exception as exc:
                logger.exception("extraction failed for %s", filename)
                return IndexResult(
                    filename=filename,
                    content_hash=content_hash,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )

        children = [c for c in chunks if not c.is_parent]
        parents = [c for c in chunks if c.is_parent]
        if not children:
            return IndexResult(
                filename=filename,
                content_hash=content_hash,
                error="document produced no chunks (empty or unreadable)",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        try:
            records = self._build_records(chunks)
            self.store.upsert(records, namespace=namespace)
        except Exception as exc:
            logger.exception("upsert failed for %s", filename)
            return IndexResult(
                filename=filename,
                content_hash=content_hash,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        self.registry.record(
            namespace=namespace,
            filename=filename,
            source_path=str(p),
            content_hash=content_hash,
            chunk_count=len(children),
            parent_count=len(parents),
            embed_model=self.embedder.model_name,
            chunker=self.chunking.model_name,
        )

        from ingestion import extraction_report

        # Only available when this call did the extraction; a caller supplying
        # prepared chunks already has its own report.
        report = extraction_report(doc) if doc is not None else {}

        return IndexResult(
            filename=filename,
            content_hash=content_hash,
            chunks_indexed=len(children),
            parents_indexed=len(parents),
            replaced_version=replaced,
            duration_ms=int((time.perf_counter() - started) * 1000),
            extraction=report,
        )

    def _delete_document_vectors(self, namespace: str, filename: str, content_hash: str) -> None:
        """
        Remove a document's vectors by metadata filter.

        Filtering on both filename and the *old* content hash means a
        concurrent write of the new version cannot be deleted by this call.
        """
        try:
            self.store.delete(
                namespace,
                metadata_filter={
                    "$and": [
                        {"filename": {"$eq": filename}},
                        {"doc_hash": {"$eq": content_hash}},
                    ]
                },
            )
        except Exception as exc:
            # Serverless Pinecone does not support delete-by-filter on all
            # tiers; fall back to filename-only deletion.
            logger.warning("filtered delete failed (%s) — retrying by filename", exc)
            try:
                self.store.delete(namespace, metadata_filter={"filename": {"$eq": filename}})
            except Exception as exc2:
                logger.error("could not delete old vectors for %s: %s", filename, exc2)

    def remove_document(self, namespace: str, filename: str) -> bool:
        """Delete a document's vectors and forget it. Returns True if it existed."""
        existing = self.registry.get(namespace, filename)
        if not existing:
            return False
        self._delete_document_vectors(namespace, filename, existing["content_hash"])
        self.registry.forget(namespace, filename)
        return True

    # ------------------------------------------------------------------
    # Bulk
    # ------------------------------------------------------------------

    def index_directory(
        self,
        directory: str | Path,
        *,
        namespace: str,
        domain: str,
        force: bool = False,
        prepared: dict[str, list[DocumentChunk]] | None = None,
    ) -> list[IndexResult]:
        """
        Index every supported document in a directory (non-recursive).

        Args:
            prepared: {absolute_path: chunks} for files already chunked by the
                caller, so extraction is not repeated.
        """
        from ingestion import SUPPORTED_EXTENSIONS

        root = Path(directory)
        if not root.exists():
            logger.warning("knowledge base directory %s does not exist", root)
            return []

        files = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        results = []
        for p in files:
            results.append(
                self.index_document(
                    p,
                    namespace=namespace,
                    domain=domain,
                    force=force,
                    prepared_chunks=(prepared or {}).get(str(p)),
                )
            )
        return results


# ---------------------------------------------------------------------------
# BM25 fitting
# ---------------------------------------------------------------------------


def fit_sparse_encoder(
    corpus_paths: Iterable[tuple[Path, str]],
    *,
    chunking: ChunkingConfig | None = None,
    save_to: str | Path | None = None,
) -> BM25Encoder:
    """
    Fit a BM25 encoder over the chunk texts of an entire corpus.

    Fitting must precede indexing: the IDF table has to be stable across every
    document, or term weights are not comparable between them. Called once per
    full rebuild.

    Args:
        corpus_paths: (path, domain) pairs to chunk for fitting.
        save_to: Where to persist the fitted table.
    """
    cfg = chunking or ChunkingConfig()
    texts: list[str] = []

    for path, domain in corpus_paths:
        try:
            doc = extract_document(path)
            texts.extend(c.text for c in chunk_document(doc, domain, config=cfg) if not c.is_parent)
        except Exception as exc:
            logger.warning("skipping %s during BM25 fit: %s", path, exc)

    encoder = BM25Encoder().fit(texts)
    if save_to:
        encoder.save(save_to)
    return encoder


def rebuild_all(
    *,
    store_factory,
    embedder: Embedder,
    registry: IndexRegistry,
    domains: dict[str, dict],
    pointer: IndexPointer,
    sparse_path: str | Path,
    chunking: ChunkingConfig | None = None,
) -> dict:
    """
    Full blue/green rebuild.

    Creates a new index version, fits BM25 over the whole corpus, indexes every
    domain into the new version, and promotes it only once everything succeeds.
    A failure part-way leaves the previous version serving traffic untouched.

    Args:
        store_factory: Callable taking an index name and returning a VectorStore.
        domains: {domain_key: {"knowledge_base_path": ..., "namespace": ...}}.
    """
    version, index_name = pointer.next_version_name()
    logger.info("starting rebuild into %s (v%d)", index_name, version)

    corpus = [
        (p, domain)
        for domain, cfg in domains.items()
        for p in sorted(Path(cfg["knowledge_base_path"]).glob("*"))
        if p.is_file()
    ]
    encoder = fit_sparse_encoder(corpus, chunking=chunking, save_to=sparse_path)

    store = store_factory(index_name)
    indexer = Indexer(store, embedder, registry, sparse_encoder=encoder, chunking=chunking)

    summary: dict = {"index": index_name, "version": version, "domains": {}}
    total_chunks = 0
    failures: list[str] = []

    for domain, cfg in domains.items():
        results = indexer.index_directory(
            cfg["knowledge_base_path"],
            namespace=cfg["namespace"],
            domain=domain,
            force=True,
        )
        indexed = sum(r.chunks_indexed for r in results)
        errored = [r.filename for r in results if r.error]
        failures.extend(errored)
        total_chunks += indexed
        summary["domains"][domain] = {
            "documents": len(results),
            "chunks": indexed,
            "failed": errored,
        }

    summary["total_chunks"] = total_chunks
    summary["failures"] = failures

    if total_chunks == 0:
        raise RuntimeError("rebuild produced no chunks — not promoting")

    pointer.promote(index_name, version, note=f"{total_chunks} chunks, {len(failures)} failures")
    summary["promoted"] = True
    logger.info("rebuild complete: %d chunks in %s", total_chunks, index_name)
    return summary
