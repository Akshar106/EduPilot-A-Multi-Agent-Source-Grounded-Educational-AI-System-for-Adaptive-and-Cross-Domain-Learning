"""
Persistence layer.

These run against the throwaway database conftest pins DATA_DIR to, so they
exercise the real schema and the real queries.
"""

from __future__ import annotations

import pytest

from edupilot import db


@pytest.fixture(scope="module", autouse=True)
def _schema():
    db.init_db()


def test_schema_is_idempotent():
    """init_db runs on every startup; a second call must be a no-op."""
    db.init_db()
    db.init_db()


def test_schema_version_is_recorded():
    version = db.get_conn().execute("PRAGMA user_version").fetchone()[0]
    assert version == db.SCHEMA_VERSION


def test_package_reexports_match_submodules():
    """`from edupilot import db` must expose everything __all__ promises."""
    for name in db.__all__:
        assert hasattr(db, name), f"db.__all__ lists {name}, which does not exist"


# ---------------------------------------------------------------------------
# Chat sessions — ownership is the whole point of this layer
# ---------------------------------------------------------------------------


def test_session_roundtrip():
    db.ensure_session("s-round", "user-1", title="hello")
    assert db.get_session_owner("s-round") == "user-1"

    mid = db.save_message("s-round", "user", "what is variance?")
    assert mid > 0

    messages = db.get_session_messages("s-round")
    assert [m["content"] for m in messages] == ["what is variance?"]


def test_ensure_session_requires_an_owner():
    """A session with no owner is unreachable and unattributable."""
    with pytest.raises(ValueError):
        db.ensure_session("s-noowner", "")


def test_list_sessions_is_scoped_to_one_user():
    db.ensure_session("s-alice", "alice")
    db.ensure_session("s-bob", "bob")

    alice_ids = {s["session_id"] for s in db.list_sessions("alice")}
    assert "s-alice" in alice_ids
    assert "s-bob" not in alice_ids


def test_delete_session_refuses_a_non_owner():
    """Ownership sits in the DELETE predicate, so a wrong user changes nothing."""
    db.ensure_session("s-owned", "owner-1")
    db.save_message("s-owned", "user", "keep me")

    assert db.delete_session("s-owned", "attacker") is False
    assert db.get_session_owner("s-owned") == "owner-1"
    assert len(db.get_session_messages("s-owned")) == 1

    assert db.delete_session("s-owned", "owner-1") is True
    assert db.get_session_owner("s-owned") is None


def test_message_metadata_survives_the_json_roundtrip():
    db.ensure_session("s-meta", "user-1")
    db.save_message(
        "s-meta",
        "assistant",
        "answer",
        intent_type="multi",
        detected_domains=["AML", "STAT"],
        quality_score=0.91,
        pipeline_meta={"retrieval": {"chunks": 5}},
    )
    msg = db.get_session_messages("s-meta")[0]
    assert msg["detected_domains"] == ["AML", "STAT"]
    assert msg["pipeline_meta"] == {"retrieval": {"chunks": 5}}
    assert msg["quality_score"] == pytest.approx(0.91)


def test_delete_messages_from_truncates_the_tail():
    db.ensure_session("s-trunc", "user-1")
    ids = [db.save_message("s-trunc", "user", f"m{i}") for i in range(4)]

    db.delete_messages_from("s-trunc", ids[2])
    remaining = [m["content"] for m in db.get_session_messages("s-trunc")]
    assert remaining == ["m0", "m1"]


# ---------------------------------------------------------------------------
# Self study
# ---------------------------------------------------------------------------


def test_study_session_roundtrip_and_scoping():
    db.create_ss_session("ss-1", "student-a", "Interview prep", "DSA + RAG")
    db.create_ss_session("ss-2", "student-b", "Other", None)

    assert db.get_ss_session_owner("ss-1") == "student-a"
    mine = {s["ss_session_id"] for s in db.list_ss_sessions("student-a")}
    assert mine == {"ss-1"}


def test_deleting_a_study_session_removes_its_children():
    db.create_ss_session("ss-cascade", "student-c", "Temp")
    db.save_ss_document("ss-cascade", "notes.pdf", ".pdf", 1024, 3)
    db.save_ss_message("ss-cascade", "user", "summarise")

    db.delete_ss_session("ss-cascade")

    assert db.get_ss_session("ss-cascade") is None
    assert db.list_ss_documents("ss-cascade") == []
    assert db.get_ss_messages("ss-cascade") == []


def test_document_chunk_accounting():
    rows = [
        {
            "chunk_id": f"c-{i}", "domain": "TESTDOM", "text": f"chunk {i}",
            "source_file": "lec1.pdf", "page_number": i, "chunk_index": i,
        }
        for i in range(3)
    ]
    db.save_chunks(rows)
    assert db.chunk_count_by_domain("TESTDOM") == 3
    assert db.get_chunk_ids_by_domain("TESTDOM") == {"c-0", "c-1", "c-2"}

    # save_chunks is INSERT OR IGNORE — re-running an ingest must not duplicate.
    db.save_chunks(rows)
    assert db.chunk_count_by_domain("TESTDOM") == 3

    db.delete_chunks_by_domain("TESTDOM")
    assert db.chunk_count_by_domain("TESTDOM") == 0


# ---------------------------------------------------------------------------
# Migration robustness
# ---------------------------------------------------------------------------


def test_migration_repairs_a_column_missing_despite_a_current_version():
    """
    A version-gated migration cannot repair itself.

    If `user_version` is advanced but the ALTER does not land — an interrupted
    startup, or a version bump committed ahead of its migration body — a
    counter-gated migration skips the column forever and every query touching
    it fails. Reconciling against the real schema fixes it on next start.
    """
    conn = db.get_conn()

    # Simulate the broken state: drop the column, leave the version current.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(chat_sessions)")}
    if "summary" in cols:
        conn.execute("ALTER TABLE chat_sessions DROP COLUMN summary")
    conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION}")
    conn.commit()

    assert "summary" not in {
        r["name"] for r in conn.execute("PRAGMA table_info(chat_sessions)")
    }

    db.init_db()

    assert "summary" in {
        r["name"] for r in conn.execute("PRAGMA table_info(chat_sessions)")
    }


def test_conversation_memory_roundtrip():
    db.ensure_session("s-memory", "user-1")
    assert db.get_session_memory("s-memory") == ("", 0)

    mid = db.save_message("s-memory", "user", "hello")
    db.set_session_memory("s-memory", "Student greeted the tutor.", mid)
    assert db.get_session_memory("s-memory") == ("Student greeted the tutor.", mid)


def test_get_messages_after_returns_only_newer_messages():
    db.ensure_session("s-after", "user-1")
    ids = [db.save_message("s-after", "user", f"m{i}") for i in range(4)]

    later = db.get_messages_after("s-after", ids[1])
    assert [m["content"] for m in later] == ["m2", "m3"]
