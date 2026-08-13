"""
HTTP surface.

Asserts the shape of the API — which routes exist, which are protected, and
what the request schemas accept — without starting a server or calling an LLM.
"""

from __future__ import annotations

import re

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
    ("POST", "/api/evaluate/cases/{case_id}"),
    ("POST", "/api/evaluate/summary"),
}


@pytest.fixture(scope="module")
def app():
    return create_app()


def _routes(app) -> set[tuple[str, str]]:
    """
    Every (method, path) the app serves, including routes nested in routers.

    FastAPI 0.141 / Starlette 1.6 stopped flattening `include_router` into
    `app.routes`. It now inserts an `_IncludedRouter` wrapper whose own `path`
    is None and which exposes the router as `original_router` rather than
    `.routes`. A flat walk therefore sees only `/docs` and `/openapi.json` and
    reports every route as missing — while the app serves all of them.

    Handling both shapes keeps this working across versions rather than
    pinning one.
    """
    found: set[tuple[str, str]] = set()
    seen: set[int] = set()

    def walk(routes) -> None:
        for route in routes or ():
            if id(route) in seen:
                continue
            seen.add(id(route))

            path = getattr(route, "path", None)
            for method in getattr(route, "methods", None) or ():
                if method not in ("HEAD", "OPTIONS") and path:
                    found.add((method, path))

            # Mounts expose `.routes`; included routers expose the router they
            # wrapped. Router paths already carry their prefix.
            walk(getattr(route, "routes", None))
            wrapped = getattr(route, "original_router", None)
            if wrapped is not None:
                walk(getattr(wrapped, "routes", None))

    walk(app.routes)
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


# ---------------------------------------------------------------------------
# Static assets must revalidate
# ---------------------------------------------------------------------------


def test_frontend_assets_are_never_heuristically_cached(app):
    """
    Asset URLs are not content-hashed, so `app.js` after a deploy is the same
    URL with different bytes. Without an explicit Cache-Control a browser
    applies heuristic freshness and can serve the old file for minutes — which
    is indistinguishable from the change never having shipped.
    """
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        for path in ("/", "/static/app.js", "/static/style.css"):
            r = client.get(path)
            assert r.status_code == 200, path
            assert r.headers.get("cache-control") == "no-cache", (
                f"{path} would be heuristically cached by browsers"
            )


def test_unchanged_assets_revalidate_cheaply(app):
    """`no-cache` must still allow a 304 — otherwise it costs a full re-download."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        first = client.get("/static/app.js")
        etag = first.headers.get("etag")
        assert etag, "no ETag means every revalidation re-sends the body"

        second = client.get("/static/app.js", headers={"If-None-Match": etag})
        assert second.status_code == 304
        assert not second.content


def test_asset_urls_carry_a_content_derived_version(app):
    """
    Hand-written `?v=6` markers never get bumped, so a deploy reuses the same
    URL with different bytes and the browser keeps its cached copy. That
    produced a mixed frontend: new HTML plus stale JS, failing on elements the
    old script still expected.
    """
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        html = client.get("/").text

    stamps = re.findall(r"/static/(?:app\.js|style\.css)\?v=([a-f0-9]+)", html)
    assert len(stamps) >= 2, "both app.js and style.css must be version-stamped"
    assert len(set(stamps)) == 1, "one build stamp should cover the bundle"
    assert not re.search(r"\?v=\d+\b", html), "a hand-written version marker survived"


def test_the_version_stamp_tracks_the_files(app, tmp_path):
    """A changed asset must produce a different URL, or caches never refresh."""
    from edupilot.api.routes.system import _asset_version
    from edupilot.core.config import STATIC_DIR

    before = _asset_version()
    target = STATIC_DIR / "app.js"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n// touch\n")
        assert _asset_version() != before
    finally:
        target.write_bytes(original)
