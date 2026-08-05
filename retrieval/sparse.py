"""
Sparse (lexical) encoding
=========================
BM25 term weights as sparse vectors, stored in Pinecone alongside the dense
vectors so lexical and semantic matching happen in one engine-side query.

Why this replaces the in-process `rank_bm25` index:

  * **It was rebuilt from SQLite on every add.** `_rebuild_bm25_from_db()` ran
    once per file during a bulk upload, re-tokenizing the entire corpus each
    time — O(N^2) on ingest.
  * **It lived only in memory, per process.** Running more than one uvicorn
    worker gave each worker a private BM25 index, and every restart paid a
    full rebuild before the first query could be served.
  * **It could not be filtered server-side.** Lexical hits were filtered in
    Python after the fact, the same defect as the dense path.

Term weights are now computed once at ingest and written to Pinecone. What
must persist is only the fitted IDF table — a small JSON file — not the
corpus. Queries are encoded with the same table so scores stay comparable.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

ENCODER_VERSION = "bm25/1.0"

_TOKEN = re.compile(r"\b\w+\b")

#: Extremely common words carry no discriminative signal and inflate every
#: sparse vector. Kept deliberately short — domain terms must never be cut.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those there their them then than but not
    can could should would may might do does did done being been if else when
    where which who whom how what why we you they he she i""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer, stopword-filtered, matching index and query sides."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def term_index(term: str, dimensions: int = 2**20) -> int:
    """
    Hash a term to a stable sparse index.

    Hashing avoids persisting a vocabulary and keeps the index stable across
    restarts and processes. Python's `hash()` is salted per process, so a
    fixed algorithm is required here.
    """
    import hashlib

    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


class BM25Encoder:
    """
    BM25 sparse encoder with a persistable IDF table.

    Document vectors use the BM25 term-frequency saturation component;
    query vectors use IDF weights. Their dot product reproduces the BM25
    score, which is what makes engine-side hybrid search equivalent to
    scoring locally.

    Args:
        k1: Term-frequency saturation. Higher means repeated terms keep adding.
        b: Length normalization strength. 0 disables it.
        dimensions: Size of the hashed term space.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, dimensions: int = 2**20) -> None:
        self.k1 = k1
        self.b = b
        self.dimensions = dimensions
        self.idf: dict[int, float] = {}
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0

    # -- fitting -----------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return bool(self.idf) and self.avg_doc_length > 0

    def fit(self, corpus: Iterable[str]) -> "BM25Encoder":
        """
        Learn IDF weights and average document length from a corpus.

        Fit once per index build over the same chunks that will be indexed.
        """
        doc_freq: Counter[int] = Counter()
        total_length = 0
        count = 0

        for text in corpus:
            tokens = tokenize(text)
            if not tokens:
                continue
            count += 1
            total_length += len(tokens)
            for index in {term_index(t, self.dimensions) for t in tokens}:
                doc_freq[index] += 1

        if count == 0:
            logger.warning("BM25 fit received an empty corpus")
            return self

        self.doc_count = count
        self.avg_doc_length = total_length / count
        # Robertson/Sparck-Jones IDF with the +1 smoothing that keeps it positive.
        self.idf = {
            index: math.log(1 + (count - freq + 0.5) / (freq + 0.5))
            for index, freq in doc_freq.items()
        }
        logger.info(
            "BM25 fitted on %d documents, %d distinct terms, avg length %.1f",
            count, len(self.idf), self.avg_doc_length,
        )
        return self

    # -- encoding ----------------------------------------------------------

    def encode_document(self, text: str) -> dict[int, float]:
        """
        Sparse BM25 term-frequency weights for a passage.

        Returns {} for empty text or an unfitted encoder, which callers treat
        as "dense-only" rather than an error — a document with no lexical
        vector is still retrievable semantically.
        """
        tokens = tokenize(text)
        if not tokens or not self.fitted:
            return {}

        length = len(tokens)
        norm = self.k1 * (1 - self.b + self.b * length / self.avg_doc_length)

        weights: dict[int, float] = {}
        for term, freq in Counter(tokens).items():
            index = term_index(term, self.dimensions)
            weights[index] = (freq * (self.k1 + 1)) / (freq + norm)
        return weights

    def encode_query(self, text: str) -> dict[int, float]:
        """
        Sparse IDF weights for a query.

        Terms unseen at fit time get the IDF a singleton document would have,
        so a rare query term still contributes instead of scoring zero.
        """
        tokens = tokenize(text)
        if not tokens or not self.fitted:
            return {}

        default_idf = math.log(1 + (self.doc_count + 0.5) / 1.5)
        weights: dict[int, float] = {}
        for term in set(tokens):
            index = term_index(term, self.dimensions)
            weights[index] = self.idf.get(index, default_idf)
        return weights

    def encode_documents(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return [self.encode_document(t) for t in texts]

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted table. Only IDF weights are stored, not the corpus."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "version": ENCODER_VERSION,
                    "k1": self.k1,
                    "b": self.b,
                    "dimensions": self.dimensions,
                    "avg_doc_length": self.avg_doc_length,
                    "doc_count": self.doc_count,
                    # JSON object keys must be strings.
                    "idf": {str(k): round(v, 6) for k, v in self.idf.items()},
                }
            )
        )
        logger.info("saved BM25 encoder (%d terms) to %s", len(self.idf), p)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Encoder | None":
        """Load a fitted encoder, or None when the file is missing or unreadable."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("could not read BM25 encoder at %s: %s", p, exc)
            return None

        if data.get("version") != ENCODER_VERSION:
            logger.warning(
                "BM25 encoder at %s is version %s, expected %s — refit required",
                p, data.get("version"), ENCODER_VERSION,
            )
            return None

        encoder = cls(k1=data["k1"], b=data["b"], dimensions=data["dimensions"])
        encoder.avg_doc_length = data["avg_doc_length"]
        encoder.doc_count = data["doc_count"]
        encoder.idf = {int(k): v for k, v in data["idf"].items()}
        logger.info("loaded BM25 encoder (%d terms) from %s", len(encoder.idf), p)
        return encoder
