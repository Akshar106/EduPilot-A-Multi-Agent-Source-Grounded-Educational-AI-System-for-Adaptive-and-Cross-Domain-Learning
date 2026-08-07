"""Unauthenticated surface: the SPA entry point, health, and client config."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

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


@router.get("/", include_in_schema=False)
async def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"service": "EduPilot API", "docs": "/docs"})
    # Always revalidate the entry point. It references app.js and style.css by
    # unversioned name, so a stale index.html pins the whole frontend to an old
    # build — the one failure that makes a shipped change look unshipped.
    return FileResponse(str(index), headers={"Cache-Control": "no-cache"})


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
