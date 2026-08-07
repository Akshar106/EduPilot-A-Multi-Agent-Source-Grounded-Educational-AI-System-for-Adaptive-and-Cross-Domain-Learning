"""
Application factory
===================
Assembles the FastAPI app: lifespan, middleware, error handling, static
assets, and router registration. Route logic lives in `routes/`; this module
contains none of it.

Run with::

    uvicorn edupilot.api.app:app --reload --port 8000

Required environment (see .env.example):
    GROQ_API_KEY, PINECONE_API_KEY, JWT_SECRET_KEY
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from edupilot import db
from edupilot.api.deps import shutdown_executor
from edupilot.api.routes import ROUTERS
from edupilot.core.config import (
    AUTH_REQUIRED,
    BOOTSTRAP_ADMIN_EMAIL,
    BOOTSTRAP_ADMIN_PASSWORD,
    CORS_ALLOWED_ORIGINS,
    IS_PRODUCTION,
    STATIC_DIR,
)
from edupilot.core.observability import configure_logging, request_id_var
from edupilot.core.services import services
from edupilot.security import AppError, AuthError, ErrorCode, Role

configure_logging()
logger = logging.getLogger("edupilot.api")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _bootstrap_admin() -> None:
    """
    Create the first admin account when the user table is empty.

    Without it there is no way to reach the admin-only knowledge-base routes
    on a fresh deployment.
    """
    if services.users.count_users() > 0:
        return
    if not (BOOTSTRAP_ADMIN_EMAIL and BOOTSTRAP_ADMIN_PASSWORD):
        if not AUTH_REQUIRED:
            # Single-user mode already grants admin, so there is nothing to
            # bootstrap and nothing to warn about.
            return
        logger.warning(
            "No users exist and BOOTSTRAP_ADMIN_EMAIL/PASSWORD are unset — "
            "knowledge-base administration is unreachable. Set both and restart."
        )
        return
    try:
        user = services.users.create_user(
            BOOTSTRAP_ADMIN_EMAIL, BOOTSTRAP_ADMIN_PASSWORD, role=Role.ADMIN
        )
        logger.info("bootstrapped admin account %s", user.email)
    except (AuthError, ValueError) as exc:
        logger.error("could not bootstrap admin: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    # Touching the property builds the UserStore, which creates the users
    # table. Doing it here means the first request never races the DDL.
    services.users  # noqa: B018 - property access with an intentional side effect

    orphans = db.orphaned_session_count()
    if any(orphans.values()):
        logger.warning(
            "%d chat and %d study sessions predate authentication and have no owner. "
            "They are hidden from all users. Run "
            "`edupilot-reindex --claim-sessions <email>` to assign them to a "
            "registered account.",
            orphans["chat_sessions"], orphans["self_study_sessions"],
        )

    if not AUTH_REQUIRED:
        from edupilot.security.deps import LOCAL_USER

        logger.warning(
            "single-user mode: authentication is OFF, every request runs as %s "
            "with admin rights. Set EDUPILOT_AUTH_REQUIRED=true to require sign-in.",
            LOCAL_USER.user_id,
        )

    _bootstrap_admin()

    health = services.health()
    if health["status"] != "ok":
        logger.error("startup health degraded — missing: %s", health["failing"])
    else:
        logger.info("startup OK — index=%s", health["active_index"])

    yield
    shutdown_executor()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build the application. Separate from the module-level `app` so tests
    can construct an isolated instance."""
    app = FastAPI(
        title="EduPilot API",
        description="Multi-agent, source-grounded educational RAG",
        version="2.0.0",
        lifespan=lifespan,
        docs_url=None if IS_PRODUCTION else "/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        # Previously allow_origins=["*"], which lets any site call this API
        # with the user's credentials attached.
        allow_origins=CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )

    if STATIC_DIR.exists():
        # Absolute path: a relative StaticFiles(directory="static") resolves
        # against the process CWD and breaks whenever the server is started
        # from anywhere but the project root.
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    _register_middleware(app)
    _register_error_handlers(app)

    for router in ROUTERS:
        app.include_router(router)

    return app


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id to every log line and response."""
        rid = uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = rid
        logger.info(
            "%s %s -> %d (%.0fms)",
            request.method, request.url.path, response.status_code, elapsed,
        )
        return response


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        exc.request_id = request_id_var.get() or exc.request_id
        exc.log()
        headers = {}
        if exc.code is ErrorCode.RATE_LIMITED and "retry_after" in exc.details:
            headers["Retry-After"] = str(exc.details["retry_after"])
        return JSONResponse(exc.to_response(), status_code=exc.status_code, headers=headers)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all that never leaks internals.

        The previous code returned `f"Pipeline error: {exc}"`, which could
        include index names, file paths, and provider error bodies.
        """
        error = AppError(code=ErrorCode.INTERNAL, internal=f"{type(exc).__name__}: {exc}")
        error.request_id = request_id_var.get() or error.request_id
        error.__cause__ = exc
        error.log()
        return JSONResponse(error.to_response(), status_code=500)


#: The ASGI application uvicorn serves.
app = create_app()
