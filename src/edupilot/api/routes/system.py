"""Unauthenticated surface: the SPA entry point, health, and client config."""

from __future__ import annotations

import hashlib
import re

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from edupilot.core.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_TOP_K,
    DOMAINS,
    GROQ_MODELS,
    MAX_QUERY_CHARS,
    STATIC_DIR,
)
from edupilot.core.services import services

router = APIRouter(tags=["system"])


#: Assets whose URLs get a content-derived version stamp.
_VERSIONED_ASSETS = ("app.js", "style.css")

_ASSET_REF = re.compile(
    r'(/static/(?:' + "|".join(a.replace(".", r"\.") for a in _VERSIONED_ASSETS) + r'))'
    r'(\?v=[^"\']*)?'
)


def _asset_version() -> str:
    """
    Cache-busting stamp derived from the asset files themselves.

    index.html previously carried hand-written `?v=6` / `?v=7` markers. Nothing
    bumps them when the files change, so the URL stays identical across a
    deploy and browsers correctly reuse the old copy — which produced a *mixed*
    frontend: new HTML with stale JS, failing on elements the old script still
    expected. Deriving the stamp from mtime and size means it cannot go stale.
    """
    parts = []
    for name in _VERSIONED_ASSETS:
        try:
            st = (STATIC_DIR / name).stat()
            parts.append(f"{name}:{int(st.st_mtime)}:{st.st_size}")
        except OSError:
            parts.append(name)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


@router.get("/", include_in_schema=False)
async def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"service": "EduPilot API", "docs": "/docs"})

    html = _ASSET_REF.sub(rf"\1?v={_asset_version()}", index.read_text(encoding="utf-8"))
    # The entry point must always revalidate: it is what carries the stamps
    # that point at every other asset.
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@router.get("/api/health")
async def health():
    return services.health()


@router.get("/api/config")
async def get_config():
    """Everything the frontend needs to render before its first real request."""
    return {
        "available_models": AVAILABLE_MODELS,
        "groq_models": GROQ_MODELS,
        "default_model": DEFAULT_MODEL,
        "domains": {
            k: {
                "name": v["name"],
                "abbr": v["abbr"],
                "color": v["color"],
                "description": v["description"],
            }
            for k, v in DOMAINS.items()
        },
        "defaults": {
            "top_k": DEFAULT_TOP_K,
            "rerank_top_k": DEFAULT_RERANK_TOP_K,
            "max_query_chars": MAX_QUERY_CHARS,
        },
    }
