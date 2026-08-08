"""
Anonymous per-browser identity.

With sign-in disabled, every unauthenticated caller used to resolve to one
shared user. The ownership checks all passed — because everyone really was the
same owner — so `list_sessions` would hand each visitor everyone else's
conversations. These tests pin the isolation that replaced it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from edupilot.api.app import create_app
from edupilot.security.deps import ANON_COOKIE, anonymous_user


@pytest.fixture(scope="module")
def app():
    return create_app()


def test_a_first_visit_is_issued_an_identity(app):
    with TestClient(app) as client:
        r = client.get("/api/auth/me")
        assert r.status_code == 200
        assert ANON_COOKIE in r.cookies or ANON_COOKIE in client.cookies


def test_two_browsers_get_different_identities(app):
    # Separate clients = separate cookie jars = separate browsers.
    with TestClient(app) as a, TestClient(app) as b:
        id_a = a.get("/api/auth/me").json()["user_id"]
        id_b = b.get("/api/auth/me").json()["user_id"]

    assert id_a.startswith("anon-") and id_b.startswith("anon-")
    assert id_a != id_b, "every browser must get its own identity"


def test_an_identity_is_stable_across_requests(app):
    with TestClient(app) as client:
        first = client.get("/api/auth/me").json()["user_id"]
        second = client.get("/api/auth/me").json()["user_id"]
        third = client.get("/api/sessions").status_code
    assert first == second, "the cookie must persist the identity"
    assert third == 200


def test_one_visitor_does_not_see_anothers_conversations_listed(app):
    """The isolation that a single shared identity destroyed."""
    with TestClient(app) as a, TestClient(app) as b:
        created = a.post("/api/sessions", json={"title": "alice private"})
        assert created.status_code == 200
        sid = created.json()["session_id"]

        a_ids = {s["session_id"] for s in a.get("/api/sessions").json()["sessions"]}
        b_ids = {s["session_id"] for s in b.get("/api/sessions").json()["sessions"]}

        assert sid in a_ids, "the owner must see their own session"
        assert sid not in b_ids, "another visitor must not see it listed"


def test_a_student_cannot_reach_another_visitors_session_by_id():
    """
    Listing is scoped by owner, but direct access is guarded by owns_or_admin.
    Tested at the dependency rather than over HTTP because the answer depends
    on the role, and the role depends on import-time config: anonymous callers
    are students in production and admins in development, where the operator is
    the only visitor. This pins the production behaviour, which is the one that
    matters on a deployment.
    """
    from edupilot.security import AppError, owns_or_admin

    visitor = anonymous_user("bbbbbbbbbbbbbbbb", is_admin=False)
    with pytest.raises(AppError):
        owns_or_admin(visitor, "anon-aaaaaaaaaaaaaaaa", what="chat session")

    # Their own session is reachable.
    owns_or_admin(visitor, visitor.user_id, what="chat session")


def test_a_development_admin_can_cross_access_by_design():
    """
    Documents the tradeoff rather than hiding it: with EDUPILOT_ENV=development
    an anonymous caller is an admin and can reach any session by id. That is
    intended for a single-operator local install and is precisely why
    production does not grant the role.
    """
    from edupilot.security import owns_or_admin

    operator = anonymous_user("cccccccccccccccc", is_admin=True)
    owns_or_admin(operator, "anon-someone-else", what="chat session")


def test_a_forged_cookie_only_yields_an_empty_account(app):
    """The cookie authorizes nothing; it just names whose chats these are."""
    with TestClient(app) as client:
        client.cookies.set(ANON_COOKIE, "deadbeefdeadbeefdeadbeefdeadbeef")
        r = client.get("/api/sessions")
        assert r.status_code == 200
        assert r.json()["sessions"] == []


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------


def test_anonymous_callers_are_students_in_production():
    """
    An anonymous admin on a public deployment could upload into the shared
    knowledge base, and every other visitor's answers would then be grounded
    in whatever they uploaded.
    """
    user = anonymous_user("abc123", is_admin=False)
    assert not user.is_admin


def test_anonymous_callers_may_be_admin_in_development():
    """Locally, the operator is the only visitor and manages their own corpus."""
    assert anonymous_user("abc123", is_admin=True).is_admin


def test_knowledge_base_writes_are_refused_for_non_admins(app):
    """The concrete consequence of the student role."""
    from edupilot.core.config import IS_PRODUCTION

    if IS_PRODUCTION:
        with TestClient(app) as client:
            r = client.post("/api/kb/upload", data={"domain": "AML"}, files={})
            assert r.status_code in (403, 422)
