"""
Frequency-gated semantic answer cache.

Uses a stub embedder so the tests are deterministic and need no model download.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from edupilot.retrieval.answer_cache import (
    MIN_GROUNDING_TO_CACHE,
    PROMOTE_AFTER,
    AnswerCache,
    normalize_question,
)


class StubEmbedder:
    """
    Maps a question to a unit vector by keyword overlap.

    Questions sharing keywords land close together, which is what the semantic
    match is supposed to exploit.
    """

    VOCAB = ["pvalue", "bias", "variance", "attention", "sql", "forest"]

    def embed_query(self, text: str) -> np.ndarray:
        t = normalize_question(text).replace("-", "").replace(" ", "")
        v = np.array([1.0 if w in t else 0.0 for w in self.VOCAB], dtype=np.float32)
        if not v.any():
            v = np.ones(len(self.VOCAB), dtype=np.float32)
        return v / np.linalg.norm(v)


def good_payload(answer: str = "An answer.") -> dict:
    return {
        "final_answer": answer,
        "refused": False,
        "grounding_score": 0.95,
        "sources": [{"source_num": 1, "citation_label": "AML p.1"}],
        "detected_domains": ["AML"],
    }


@pytest.fixture
def cache(tmp_path):
    return AnswerCache(str(tmp_path / "cache.db"), StubEmbedder(), index_version="edupilot-v1")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalization_collapses_incidental_differences():
    assert normalize_question("What is a P-VALUE?") == normalize_question("what is a p value")
    assert normalize_question("  Explain   bias!!  ") == "explain bias"


def test_normalization_keeps_distinct_questions_distinct():
    assert normalize_question("what is bias") != normalize_question("what is variance")


# ---------------------------------------------------------------------------
# The frequency gate — the whole point
# ---------------------------------------------------------------------------


def test_nothing_is_cached_before_the_threshold(cache):
    q = "what is a pvalue"
    for expected in range(1, PROMOTE_AFTER):
        count = cache.record_ask(q, domains=["STAT"], model="m")
        assert count == expected
        assert not cache.should_store(count, good_payload())
        assert cache.lookup(q, domains=["STAT"], model="m") is None


def test_the_threshold_ask_promotes_and_later_asks_hit(cache):
    q = "what is a pvalue"
    for _ in range(PROMOTE_AFTER):
        count = cache.record_ask(q, domains=["STAT"], model="m")

    assert count == PROMOTE_AFTER
    assert cache.should_store(count, good_payload("Cached answer."))
    assert cache.store(q, good_payload("Cached answer."), domains=["STAT"], model="m")

    hit = cache.lookup(q, domains=["STAT"], model="m")
    assert hit is not None
    assert hit.payload["final_answer"] == "Cached answer."


def test_one_off_questions_never_get_a_payload(cache):
    """The long tail is counted but never stored — that is the design."""
    for i in range(20):
        cache.record_ask(f"unique question about topic {i}", domains=None, model="m")
    stats = cache.stats()
    assert stats["tracked_questions"] == 20
    assert stats["cached_answers"] == 0


# ---------------------------------------------------------------------------
# What is eligible to be cached
# ---------------------------------------------------------------------------


def test_refusals_are_never_cached(cache):
    payload = good_payload()
    payload["refused"] = True
    assert not cache.should_store(PROMOTE_AFTER, payload)


def test_weakly_grounded_answers_are_never_cached(cache):
    payload = good_payload()
    payload["grounding_score"] = MIN_GROUNDING_TO_CACHE - 0.01
    assert not cache.should_store(PROMOTE_AFTER, payload)


def test_unmeasured_grounding_is_never_cached(cache):
    payload = good_payload()
    payload["grounding_score"] = None
    assert not cache.should_store(PROMOTE_AFTER, payload)


def test_empty_answers_are_never_cached(cache):
    assert not cache.should_store(PROMOTE_AFTER, good_payload("   "))


def test_stored_payload_excludes_the_diagnostics_blob(cache):
    q = "what is bias"
    for _ in range(PROMOTE_AFTER):
        cache.record_ask(q, domains=None, model="m")
    payload = good_payload()
    payload["debug"] = {"retrieval": {"chunks": ["lots of text"]}}
    cache.store(q, payload, domains=None, model="m")

    hit = cache.lookup(q, domains=None, model="m")
    assert "debug" not in hit.payload


# ---------------------------------------------------------------------------
# Semantic matching
# ---------------------------------------------------------------------------


def test_a_paraphrase_hits(cache):
    for _ in range(PROMOTE_AFTER):
        cache.record_ask("what is a pvalue", domains=None, model="m")
    cache.store("what is a pvalue", good_payload("P!"), domains=None, model="m")

    # Different wording, same keyword — the stub puts these at similarity 1.0.
    hit = cache.lookup("explain pvalue please", domains=None, model="m")
    assert hit is not None and hit.payload["final_answer"] == "P!"


def test_an_unrelated_question_does_not_hit(cache):
    for _ in range(PROMOTE_AFTER):
        cache.record_ask("what is a pvalue", domains=None, model="m")
    cache.store("what is a pvalue", good_payload("P!"), domains=None, model="m")

    assert cache.lookup("explain attention in transformers", domains=None, model="m") is None


# ---------------------------------------------------------------------------
# Scoping and invalidation
# ---------------------------------------------------------------------------


def test_a_different_model_does_not_share_cache(cache):
    for _ in range(PROMOTE_AFTER):
        cache.record_ask("what is bias", domains=None, model="model-a")
    cache.store("what is bias", good_payload("A"), domains=None, model="model-a")

    assert cache.lookup("what is bias", domains=None, model="model-b") is None


def test_a_different_domain_scope_does_not_share_cache(cache):
    for _ in range(PROMOTE_AFTER):
        cache.record_ask("what is bias", domains=["AML"], model="m")
    cache.store("what is bias", good_payload("A"), domains=["AML"], model="m")

    assert cache.lookup("what is bias", domains=["STAT"], model="m") is None


def test_a_rebuild_invalidates_cached_answers(tmp_path):
    """Entries from a retired index must never be served after a promotion."""
    path = str(tmp_path / "cache.db")
    old = AnswerCache(path, StubEmbedder(), index_version="edupilot-v1")
    for _ in range(PROMOTE_AFTER):
        old.record_ask("what is bias", domains=None, model="m")
    old.store("what is bias", good_payload("from v1"), domains=None, model="m")
    assert old.lookup("what is bias", domains=None, model="m") is not None

    # Same file, new index version — as after `edupilot-reindex --rebuild`.
    new = AnswerCache(path, StubEmbedder(), index_version="edupilot-v2")
    assert new.lookup("what is bias", domains=None, model="m") is None
    assert new.sweep() >= 1


def test_expired_entries_are_not_served(cache, monkeypatch):
    q = "what is bias"
    for _ in range(PROMOTE_AFTER):
        cache.record_ask(q, domains=None, model="m")
    cache.store(q, good_payload(), domains=None, model="m")
    assert cache.lookup(q, domains=None, model="m") is not None

    import edupilot.retrieval.answer_cache as mod

    # Capture the real clock first: patching `time.time` with a lambda that
    # itself calls `time.time()` recurses forever.
    real_now = time.time()
    monkeypatch.setattr(mod.time, "time", lambda: real_now + mod.TTL_SECONDS + 60)
    assert cache.lookup(q, domains=None, model="m") is None


def test_hit_counts_are_recorded(cache):
    q = "what is bias"
    for _ in range(PROMOTE_AFTER):
        cache.record_ask(q, domains=None, model="m")
    cache.store(q, good_payload(), domains=None, model="m")

    cache.lookup(q, domains=None, model="m")
    cache.lookup(q, domains=None, model="m")
    assert cache.stats()["total_hits"] == 2
