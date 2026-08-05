"""
HTTP surface.

Asserts the shape of the API — which routes exist, which are protected, and
what the request schemas accept — without starting a server or calling an LLM.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edupilot.api.app import create_app
from edupilot.api.routes import ROUTERS
from edupilot.api.schemas import ChatRequest, RegisterRequest, StudyChatRequest

#: Every route the frontend depends on. A rename here is a breaking change,
#: so it is pinned rather than derived.
EXPECTED_ROUTES = {
    ("GET", "/"),
    ("POST", "/api/auth/register"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/refresh"),
    ("POST", "/api/auth/logout"),
    ("GET", "/api/auth/me"),
    ("GET", "/api/health"),
    ("GET", "/api/config"),
    ("POST", "/api/chat"),
    ("GET", "/api/sessions"),
    ("POST", "/api/sessions"),
    ("GET", "/api/sessions/{session_id}"),
    ("DELETE", "/api/sessions/{session_id}"),
    ("DELETE", "/api/sessions/{session_id}/messages/{message_id}"),
    ("GET", "/api/kb/status"),
    ("GET", "/api/kb/documents"),
    ("POST", "/api/kb/upload"),
    ("DELETE", "/api/kb/{domain}/{filename}"),
    ("GET", "/api/documents/{domain}/{filename}"),
    ("POST", "/api/self-study/sessions"),
    ("GET", "/api/self-study/sessions"),
    ("GET", "/api/self-study/sessions/{ss_session_id}"),
    ("DELETE", "/api/self-study/sessions/{ss_session_id}"),
    ("POST", "/api/self-study/sessions/{ss_session_id}/upload"),
    ("DELETE", "/api/self-study/sessions/{ss_session_id}/documents/{doc_id}"),
    ("POST", "/api/self-study/chat"),
    ("GET", "/api/evaluate/cases"),
}


@pytest.fixture(scope="module")
def app():
    return create_app()


def _routes(app) -> set[tuple[str, str]]:
    found = set()
    for route in app.routes:
        for method in getattr(route, "methods", None) or ():
            if method not in ("HEAD", "OPTIONS"):
                found.add((method, route.path))
    return found


def test_every_expected_route_is_registered(app):
    missing = EXPECTED_ROUTES - _routes(app)
    assert not missing, f"routes disappeared: {sorted(missing)}"


def test_no_unexpected_api_routes(app):
    """Catches a route added without a corresponding test or doc update."""
    actual_api = {(m, p) for m, p in _routes(app) if p.startswith("/api/")}
    expected_api = {(m, p) for m, p in EXPECTED_ROUTES if p.startswith("/api/")}
    assert actual_api - expected_api == set()


def test_all_routers_are_registered():
    """A router left out of ROUTERS silently serves nothing."""
    from edupilot.api.routes import (
        auth,
        chat,
        evaluation,
        knowledge_base,
        self_study,
        sessions,
        system,
    )

    for module in (auth, chat, sessions, knowledge_base, self_study, evaluation, system):
        assert module.router in ROUTERS, f"{module.__name__} is not registered"


def test_openapi_schema_builds(app):
    """A duplicated operation id or bad annotation blows up here first."""
    schema = app.openapi()
    assert schema["info"]["title"] == "EduPilot API"
    assert schema["paths"]


# ---------------------------------------------------------------------------
# Schemas — the input bounds that keep prompt cost and abuse in check
# ---------------------------------------------------------------------------


def test_chat_query_cannot_be_empty():
    with pytest.raises(ValidationError):
        ChatRequest(query="")


def test_chat_query_is_length_bounded():
    from edupilot.core.config import MAX_QUERY_CHARS

    with pytest.raises(ValidationError):
        ChatRequest(query="x" * (MAX_QUERY_CHARS + 1))


def test_chat_top_k_is_bounded():
    with pytest.raises(ValidationError):
        ChatRequest(query="hi", top_k=999)
    with pytest.raises(ValidationError):
        ChatRequest(query="hi", rerank_top_k=0)


def test_chat_defaults_come_from_config():
    from edupilot.core.config import DEFAULT_MODEL, DEFAULT_RERANK_TOP_K, DEFAULT_TOP_K

    req = ChatRequest(query="what is variance?")
    assert req.model == DEFAULT_MODEL
    assert req.top_k == DEFAULT_TOP_K
    assert req.rerank_top_k == DEFAULT_RERANK_TOP_K
    assert req.session_id is None


def test_no_schema_accepts_a_caller_supplied_owner():
    """Ownership must come from the token; a body field would reintroduce the IDOR."""
    for model in (ChatRequest, StudyChatRequest, RegisterRequest):
        assert "user_id" not in model.model_fields
        assert "owner" not in model.model_fields


def test_register_password_has_a_floor_and_a_bcrypt_ceiling():
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.co", password="short")
    with pytest.raises(ValidationError):
        RegisterRequest(email="a@b.co", password="x" * 73)
    RegisterRequest(email="a@b.co", password="a-long-enough-password")
