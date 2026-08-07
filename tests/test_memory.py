"""
Rolling conversation memory.

Runs against the throwaway database conftest pins DATA_DIR to, with a stub LLM
so no provider is called.
"""

from __future__ import annotations

import pytest

from edupilot import db
from edupilot.agents.memory import (
    KEEP_RECENT_TURNS,
    MAX_DIGEST_CHARS,
    SUMMARIZE_BATCH,
    ConversationMemory,
    format_memory,
)


class StubLLM:
    """Records its calls and returns a canned digest."""

    def __init__(self, reply: str = "Student asked about bias and variance."):
        self.reply = reply
        self.calls: list[dict] = []

    def __call__(self, *, messages, system, model, max_tokens):
        self.calls.append({"messages": messages, "system": system})
        return self.reply


class FailingLLM:
    def __call__(self, **kwargs):
        raise RuntimeError("provider down")


@pytest.fixture(autouse=True)
def _schema():
    db.init_db()


def _session(sid: str, turns: int) -> str:
    db.ensure_session(sid, "test-user")
    for i in range(turns):
        db.save_message(sid, "user", f"question {i}")
        db.save_message(sid, "assistant", f"answer {i}")
    return sid


# ---------------------------------------------------------------------------
# Short conversations pay nothing
# ---------------------------------------------------------------------------


def test_a_short_conversation_is_never_summarized():
    llm = StubLLM()
    sid = _session("mem-short", turns=2)

    digest, recent = ConversationMemory(llm).context_for(sid, model="m")

    assert digest == ""
    assert llm.calls == [], "summarization should not run on a short conversation"
    assert len(recent) == 4


def test_an_empty_session_returns_empty_context():
    db.ensure_session("mem-empty", "test-user")
    digest, recent = ConversationMemory(StubLLM()).context_for("mem-empty", model="m")
    assert digest == "" and recent == []


def test_no_session_id_is_handled():
    assert ConversationMemory(StubLLM()).context_for("", model="m") == ("", [])


# ---------------------------------------------------------------------------
# Rollover
# ---------------------------------------------------------------------------


def test_a_long_conversation_gets_a_digest_and_a_short_tail():
    llm = StubLLM()
    sid = _session("mem-long", turns=12)  # 24 messages

    digest, recent = ConversationMemory(llm).context_for(sid, model="m")

    assert digest == llm.reply
    assert len(llm.calls) == 1
    assert len(recent) == KEEP_RECENT_TURNS, "tail must be exactly the verbatim window"
    # The tail is the newest messages.
    assert recent[-1]["content"] == "answer 11"


def test_the_digest_persists_and_is_not_recomputed():
    llm = StubLLM()
    sid = _session("mem-persist", turns=12)
    mem = ConversationMemory(llm)

    mem.context_for(sid, model="m")
    assert len(llm.calls) == 1

    # Second call with no new messages: the digest already covers everything
    # older, so there is nothing left to fold.
    digest, recent = mem.context_for(sid, model="m")
    assert digest == llm.reply
    assert len(llm.calls) == 1, "digest must not be recomputed when nothing changed"


def test_summarization_is_incremental_across_rollovers():
    llm = StubLLM()
    sid = _session("mem-incremental", turns=12)
    mem = ConversationMemory(llm)
    mem.context_for(sid, model="m")

    # Enough new turns to trigger a second fold.
    for i in range(KEEP_RECENT_TURNS + SUMMARIZE_BATCH):
        db.save_message(sid, "user", f"later {i}")

    mem.context_for(sid, model="m")
    assert len(llm.calls) == 2

    # The second call must carry the previous digest forward, not start over.
    second = llm.calls[1]["messages"][0]["content"]
    assert "<existing_digest>" in second


def test_current_question_can_be_excluded_from_its_own_history():
    sid = _session("mem-exclude", turns=1)
    mid = db.save_message(sid, "user", "the question being answered right now")

    _, recent = ConversationMemory(StubLLM()).context_for(
        sid, model="m", exclude_message_ids={mid}
    )
    assert all(m["content"] != "the question being answered right now" for m in recent)


# ---------------------------------------------------------------------------
# Failure and truncation behaviour
# ---------------------------------------------------------------------------


def test_a_failing_summarizer_degrades_to_recent_turns():
    sid = _session("mem-fail", turns=12)
    digest, recent = ConversationMemory(FailingLLM()).context_for(sid, model="m")

    assert digest == "", "no digest is better than a wrong one"
    assert len(recent) == KEEP_RECENT_TURNS, "the conversation must still have context"


def test_a_failed_summary_is_retried_next_time():
    sid = _session("mem-retry", turns=12)
    ConversationMemory(FailingLLM()).context_for(sid, model="m")

    stored, through = db.get_session_memory(sid)
    assert stored == "" and through == 0, "a failed fold must not advance the pointer"

    llm = StubLLM()
    digest, _ = ConversationMemory(llm).context_for(sid, model="m")
    assert digest == llm.reply


def test_the_digest_is_length_capped():
    llm = StubLLM("x" * (MAX_DIGEST_CHARS * 3))
    sid = _session("mem-cap", turns=12)
    digest, _ = ConversationMemory(llm).context_for(sid, model="m")
    assert len(digest) <= MAX_DIGEST_CHARS


# ---------------------------------------------------------------------------
# Truncation must invalidate a digest that covered deleted turns
# ---------------------------------------------------------------------------


def test_editing_an_early_message_drops_a_digest_that_covered_it():
    llm = StubLLM()
    sid = _session("mem-truncate", turns=12)
    ConversationMemory(llm).context_for(sid, model="m")

    stored, through = db.get_session_memory(sid)
    assert stored and through > 0

    # Student edits a message the digest already folded in.
    db.delete_messages_from(sid, through - 1)

    stored_after, through_after = db.get_session_memory(sid)
    assert stored_after == "" and through_after == 0


def test_truncating_a_recent_message_keeps_the_digest():
    """Only a digest that actually covered the deleted turns is dropped."""
    llm = StubLLM()
    sid = _session("mem-truncate-late", turns=12)
    ConversationMemory(llm).context_for(sid, model="m")
    _, through = db.get_session_memory(sid)

    db.delete_messages_from(sid, through + 5)

    stored_after, _ = db.get_session_memory(sid)
    assert stored_after == llm.reply


# ---------------------------------------------------------------------------
# Prompt block
# ---------------------------------------------------------------------------


def test_empty_digest_renders_nothing():
    assert format_memory("") == ""
    assert format_memory("   ") == ""


def test_the_memory_block_denies_itself_evidentiary_weight():
    block = format_memory("We discussed bagging.")
    assert "<conversation_memory>" in block
    assert "NOT evidence" in block
    assert "We discussed bagging." in block
