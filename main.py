"""
EduPilot — FastAPI backend
==========================
Run with:
    uvicorn main:app --reload --port 8000

Required environment (see .env.example):
    GROQ_API_KEY, PINECONE_API_KEY, JWT_SECRET_KEY

Security posture, relative to the previous version:

  * every user-scoped route requires authentication, and ownership is checked
    against the token's user id — never against a value from the request body
  * both upload handlers sanitize filenames before touching the filesystem
    (they previously wrote `uf.filename` directly, allowing arbitrary writes)
  * knowledge-base mutation requires the admin role
  * CORS is restricted to configured origins
  * every route is rate limited
  * errors return a typed envelope; exception text stays in the log
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import database as db
from config import (
    AVAILABLE_MODELS,
    BASE_DIR,
    BOOTSTRAP_ADMIN_EMAIL,
    BOOTSTRAP_ADMIN_PASSWORD,
    CORS_ALLOWED_ORIGINS,
    DEFAULT_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_TOP_K,
    DOMAINS,
    ENABLE_HYDE,
    ENABLE_MULTI_QUERY,
    ENABLE_PARENT_EXPANSION,
    IS_PRODUCTION,
    MAX_QUERY_CHARS,
    VERIFY_MODEL,
)
from observability import RequestContext, configure_logging, request_id_var
from security import (
    AppError,
    AuthError,
    ErrorCode,
    Role,
    User,
    admin_user,
    create_access_token,
    create_refresh_token,
    current_user,
    owns_or_admin,
    rate_limited,
    resolve_within,
    validate_batch,
)
from security.uploads import UploadRejected
from services import services

configure_logging()
logger = logging.getLogger("edupilot.api")

STATIC_DIR = BASE_DIR / "static"
SELF_STUDY_DIR = BASE_DIR / "self_study_files"

_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="edupilot")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    services.users  # force the user table to exist before the first request

    orphans = db.orphaned_session_count()
    if any(orphans.values()):
        logger.warning(
            "%d chat and %d study sessions predate authentication and have no owner. "
            "They are hidden from all users. Run `python reindex.py --claim-sessions <email>` "
            "to assign them.",
            orphans["chat_sessions"], orphans["self_study_sessions"],
        )

    _bootstrap_admin()

    health = services.health()
    if health["status"] != "ok":
        logger.error("startup health degraded — missing: %s", health["failing"])
    else:
        logger.info("startup OK — index=%s", health["active_index"])

    yield
    _executor.shutdown(wait=False)


def _bootstrap_admin() -> None:
    """
    Create the first admin account when the user table is empty.

    Without it there is no way to reach the admin-only knowledge-base routes
    on a fresh deployment.
    """
    if services.users.count_users() > 0:
        return
    if not (BOOTSTRAP_ADMIN_EMAIL and BOOTSTRAP_ADMIN_PASSWORD):
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
    # Previously allow_origins=["*"], which lets any site call this API with
    # the user's credentials attached.
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

if STATIC_DIR.exists():
    # Absolute path: the previous StaticFiles(directory="static") resolved
    # against the process CWD and broke whenever the server was started from
    # anywhere but the project root.
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Middleware and error handling
# ---------------------------------------------------------------------------


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

    The previous code returned `f"Pipeline error: {exc}"`, which could include
    index names, file paths, and provider error bodies.
    """
    error = AppError(code=ErrorCode.INTERNAL, internal=f"{type(exc).__name__}: {exc}")
    error.request_id = request_id_var.get() or error.request_id
    error.__cause__ = exc
    error.log()
    return JSONResponse(error.to_response(), status_code=500)


async def run_blocking(fn, *args, **kwargs):
    """Run a synchronous call in the worker pool, preserving the request id."""
    rid = request_id_var.get()

    def wrapped():
        token = request_id_var.set(rid)
        try:
            return fn(*args, **kwargs)
        finally:
            request_id_var.reset(token)

    return await asyncio.get_running_loop().run_in_executor(_executor, wrapped)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=10, max_length=72)
    display_name: str = Field(default="", max_length=80)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=72)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=512)


class ChatRequest(BaseModel):
    # session_id is optional: a new session is created and owned by the caller
    # when omitted. Supplying someone else's id fails the ownership check.
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    session_id: str | None = None
    model: str = DEFAULT_MODEL
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    rerank_top_k: int = Field(default=DEFAULT_RERANK_TOP_K, ge=1, le=12)
    enable_verification: bool = True
    manual_domains: list[str] | None = None
    attached_filenames: list[str] | None = None
    chat_history: list[dict] | None = None


class NewSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class CreateStudySessionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)


class StudyChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    ss_session_id: str
    model: str = DEFAULT_MODEL
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    rerank_top_k: int = Field(default=DEFAULT_RERANK_TOP_K, ge=1, le=12)
    enable_verification: bool = True
    chat_history: list[dict] | None = None
    source_filter: list[str] | None = None


CurrentUser = Annotated[User, Depends(current_user)]
AdminUser = Annotated[User, Depends(admin_user)]


def _validate_model(name: str) -> str:
    """Reject unknown model names rather than passing them to a provider."""
    if name not in AVAILABLE_MODELS:
        raise AppError(
            code=ErrorCode.VALIDATION_FAILED,
            message=f"Unknown model '{name}'.",
            details={"available": AVAILABLE_MODELS},
        )
    return name


def _retrieval_config(top_k: int, rerank_top_k: int):
    from retrieval import RetrievalConfig

    return RetrievalConfig(
        top_k=rerank_top_k,
        candidate_multiplier=max(3, top_k // 2),
        use_multi_query=ENABLE_MULTI_QUERY,
        use_hyde=ENABLE_HYDE,
        expand_to_parents=ENABLE_PARENT_EXPANSION,
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _issue_tokens(user: User, response: Response) -> dict:
    access = create_access_token(user)
    refresh, digest, expires = create_refresh_token()
    services.users.store_refresh_token(user.user_id, digest, expires)

    response.set_cookie(
        "edupilot_access",
        access,
        httponly=True,
        secure=IS_PRODUCTION,
        samesite="lax",
        max_age=1800,
    )
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "role": str(user.role),
            "display_name": user.display_name,
        },
    }


@app.post("/api/auth/register", dependencies=[Depends(rate_limited("auth"))])
async def register(req: RegisterRequest, response: Response):
    try:
        user = services.users.create_user(
            req.email, req.password, display_name=req.display_name
        )
    except AuthError as exc:
        raise AppError(code=ErrorCode.VALIDATION_FAILED, message=exc.message) from exc
    except ValueError as exc:
        raise AppError(code=ErrorCode.VALIDATION_FAILED, message=str(exc)) from exc
    return _issue_tokens(user, response)


@app.post("/api/auth/login", dependencies=[Depends(rate_limited("auth"))])
async def login(req: LoginRequest, response: Response):
    try:
        user = await run_blocking(services.users.authenticate, req.email, req.password)
    except AuthError as exc:
        raise AppError(
            code=ErrorCode.UNAUTHENTICATED,
            message=exc.message,
            internal=f"failed login for {req.email}",
        ) from exc
    return _issue_tokens(user, response)


@app.post("/api/auth/refresh", dependencies=[Depends(rate_limited("auth"))])
async def refresh(req: RefreshRequest, response: Response):
    try:
        user = services.users.consume_refresh_token(req.refresh_token)
    except AuthError as exc:
        raise AppError(code=ErrorCode.UNAUTHENTICATED, message=exc.message) from exc
    return _issue_tokens(user, response)


@app.post("/api/auth/logout")
async def logout(user: CurrentUser, response: Response):
    revoked = services.users.revoke_all_refresh_tokens(user.user_id)
    response.delete_cookie("edupilot_access")
    return {"signed_out": True, "sessions_revoked": revoked}


@app.get("/api/auth/me")
async def me(user: CurrentUser):
    return {
        "user_id": user.user_id,
        "email": user.email,
        "role": str(user.role),
        "display_name": user.display_name,
    }


# ---------------------------------------------------------------------------
# Health and config
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"service": "EduPilot API", "docs": "/docs"})
    return FileResponse(str(index))


@app.get("/api/health")
async def health():
    return services.health()


@app.get("/api/config")
async def get_config():
    from config import GROQ_MODELS

    return {
        "available_models": AVAILABLE_MODELS,
        "groq_models": GROQ_MODELS,
        "default_model": DEFAULT_MODEL,
        "domains": {
            k: {"name": v["name"], "abbr": v["abbr"], "color": v["color"],
                "description": v["description"]}
            for k, v in DOMAINS.items()
        },
        "defaults": {
            "top_k": DEFAULT_TOP_K,
            "rerank_top_k": DEFAULT_RERANK_TOP_K,
            "max_query_chars": MAX_QUERY_CHARS,
        },
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def _run_chat(req: ChatRequest, session_id: str) -> dict:
    from agents import PipelineConfig
    from llm import get_usage, start_usage

    start_usage()
    cfg = PipelineConfig(
        model=req.model,
        verify_model=VERIFY_MODEL,
        retrieval=_retrieval_config(req.top_k, req.rerank_top_k),
        enable_verification=req.enable_verification,
    )
    result = services.pipeline.run(
        req.query,
        cfg,
        history=req.chat_history or [],
        manual_domains=req.manual_domains,
        filenames=req.attached_filenames,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics["usage"] = get_usage().as_dict()

    return {
        "session_id": session_id,
        "final_answer": result.final_answer,
        "intent_type": result.intent_type,
        "detected_domains": result.domains,
        "is_course_related": result.is_course_related,
        "needs_clarification": result.needs_clarification,
        "refused": result.refused,
        # None when grounding was not measured. Never defaulted, and never
        # floored — main.py:714 previously applied max(quality, 0.75).
        "grounding_score": result.grounding_score,
        "guardrail_action": result.verdict.action if result.verdict else None,
        "sources": result.sources,
        "debug": diagnostics,
    }


@app.post("/api/chat", dependencies=[Depends(rate_limited("chat"))])
async def chat(req: ChatRequest, user: CurrentUser):
    _validate_model(req.model)

    if req.session_id:
        owner = db.get_session_owner(req.session_id)
        if owner is None:
            # Unknown id — create it for this caller rather than 404ing, so a
            # client that generated an id locally still works.
            db.ensure_session(req.session_id, user.user_id)
        else:
            owns_or_admin(user, owner, what="chat session")
        session_id = req.session_id
    else:
        session_id = str(uuid.uuid4())
        db.ensure_session(session_id, user.user_id)

    user_msg_id = db.save_message(session_id, "user", req.query)
    if len(db.get_session_messages(session_id)) == 1:
        db.update_session_title(session_id, req.query[:60])

    result = await run_blocking(_run_chat, req, session_id)

    result["assistant_message_id"] = db.save_message(
        session_id=session_id,
        role="assistant",
        content=result["final_answer"],
        intent_type=result.get("intent_type"),
        detected_domains=result.get("detected_domains"),
        quality_score=result.get("grounding_score"),
        pipeline_meta=result.get("debug"),
    )
    result["user_message_id"] = user_msg_id
    return result


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@app.get("/api/sessions", dependencies=[Depends(rate_limited("read"))])
async def list_sessions(user: CurrentUser):
    return {"sessions": db.list_sessions(user.user_id, limit=50)}


@app.post("/api/sessions")
async def create_session(req: NewSessionRequest, user: CurrentUser):
    session_id = str(uuid.uuid4())
    db.ensure_session(session_id, user.user_id, title=req.title)
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}", dependencies=[Depends(rate_limited("read"))])
async def get_session(session_id: str, user: CurrentUser):
    owner = db.get_session_owner(session_id)
    if owner is None:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no session {session_id}")
    owns_or_admin(user, owner, what="chat session")
    return {"session_id": session_id, "messages": db.get_session_messages(session_id)}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user: CurrentUser):
    if not db.delete_session(session_id, user.user_id):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            internal=f"user {user.user_id} could not delete session {session_id}",
        )
    return {"deleted": session_id}


@app.delete("/api/sessions/{session_id}/messages/{message_id}")
async def truncate_from_message(session_id: str, message_id: int, user: CurrentUser):
    owner = db.get_session_owner(session_id)
    if owner is None:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no session {session_id}")
    owns_or_admin(user, owner, what="chat session")
    db.delete_messages_from(session_id, message_id)
    return {"truncated": True, "from_message_id": message_id}


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


@app.get("/api/kb/status", dependencies=[Depends(rate_limited("read"))])
async def kb_status(user: CurrentUser):
    stats = await run_blocking(services.store.stats)
    out = {}
    for domain, cfg in DOMAINS.items():
        namespace = cfg["pinecone_namespace"]
        out[domain] = {
            "name": cfg["name"],
            "color": cfg["color"],
            "chunk_count": stats.get("namespaces", {}).get(namespace, 0),
            "documents": services.registry.list_documents(namespace),
        }
    return out


@app.post("/api/kb/upload", dependencies=[Depends(rate_limited("upload"))])
async def upload_to_kb(
    admin: AdminUser,
    domain: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    Add documents to the shared course knowledge base. Admin only.

    Previously unauthenticated, so any visitor could inject documents that
    every student's answers would then be grounded in.
    """
    if domain not in DOMAINS:
        raise AppError(code=ErrorCode.VALIDATION_FAILED, message=f"Unknown domain '{domain}'.")

    raw = [(f.filename or "", await f.read()) for f in files]
    try:
        validated = validate_batch(raw)
    except UploadRejected as exc:
        raise AppError(code=ErrorCode.UPLOAD_REJECTED, message=str(exc)) from exc

    kb_path = Path(DOMAINS[domain]["knowledge_base_path"])
    kb_path.mkdir(parents=True, exist_ok=True)
    namespace = DOMAINS[domain]["pinecone_namespace"]

    results = []
    for item in validated:
        # resolve_within re-checks containment after sanitization. The old
        # code did `kb_path / uf.filename` with no check at all.
        dest = resolve_within(kb_path, item.safe_name)
        dest.write_bytes(item.content)

        outcome = await run_blocking(
            services.indexer.index_document, dest, namespace=namespace, domain=domain
        )
        db.save_uploaded_doc(
            filename=item.safe_name,
            domain=domain,
            file_type=item.extension,
            chunk_count=outcome.chunks_indexed,
            file_size_bytes=item.size_bytes,
        )
        results.append({
            "filename": item.safe_name,
            "original_name": item.original_name,
            "chunks_indexed": outcome.chunks_indexed,
            "skipped": outcome.skipped,
            "replaced_version": outcome.replaced_version,
            "error": outcome.error,
            "warnings": item.warnings,
        })

    return {"uploaded": results}


@app.get("/api/kb/documents", dependencies=[Depends(rate_limited("read"))])
async def list_kb_documents(user: CurrentUser):
    out = {}
    for domain, cfg in DOMAINS.items():
        out[domain] = {
            "name": cfg["name"],
            "color": cfg["color"],
            "documents": services.registry.list_documents(cfg["pinecone_namespace"]),
        }
    return out


@app.get("/api/documents/{domain}/{filename}", dependencies=[Depends(rate_limited("read"))])
async def serve_document(domain: str, filename: str, user: CurrentUser):
    if domain not in DOMAINS:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"unknown domain {domain}")
    kb_path = Path(DOMAINS[domain]["knowledge_base_path"])
    try:
        path = resolve_within(kb_path, filename)
    except UploadRejected as exc:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=str(exc)) from exc
    if not path.is_file():
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"missing {path}")
    return FileResponse(str(path), filename=path.name)


@app.delete("/api/kb/{domain}/{filename}")
async def delete_kb_document(domain: str, filename: str, admin: AdminUser):
    if domain not in DOMAINS:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"unknown domain {domain}")
    namespace = DOMAINS[domain]["pinecone_namespace"]
    safe = resolve_within(Path(DOMAINS[domain]["knowledge_base_path"]), filename)

    removed = await run_blocking(services.indexer.remove_document, namespace, safe.name)
    if safe.is_file():
        safe.unlink()
    return {"deleted": safe.name, "was_indexed": removed}


# ---------------------------------------------------------------------------
# Self study
# ---------------------------------------------------------------------------


async def _require_study_session(ss_session_id: str, user: User) -> dict:
    session = db.get_ss_session(ss_session_id)
    if not session:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no study session {ss_session_id}")
    owns_or_admin(user, db.get_ss_session_owner(ss_session_id) or "", what="study session")
    return session


@app.post("/api/self-study/sessions")
async def create_study_session(req: CreateStudySessionRequest, user: CurrentUser):
    ss_session_id = str(uuid.uuid4())
    db.create_ss_session(ss_session_id, user.user_id, req.name.strip(), req.description)
    return {"ss_session_id": ss_session_id, "name": req.name.strip()}


@app.get("/api/self-study/sessions", dependencies=[Depends(rate_limited("read"))])
async def list_study_sessions(user: CurrentUser):
    return {"sessions": db.list_ss_sessions(user.user_id)}


@app.get("/api/self-study/sessions/{ss_session_id}", dependencies=[Depends(rate_limited("read"))])
async def get_study_session(ss_session_id: str, user: CurrentUser):
    session = await _require_study_session(ss_session_id, user)
    return {
        "session": session,
        "documents": db.list_ss_documents(ss_session_id),
        "messages": db.get_ss_messages(ss_session_id),
    }


@app.delete("/api/self-study/sessions/{ss_session_id}")
async def delete_study_session(ss_session_id: str, user: CurrentUser):
    await _require_study_session(ss_session_id, user)
    namespace = f"ss_{ss_session_id.replace('-', '')}"

    await run_blocking(services.store.delete, namespace, delete_all=True)
    db.delete_ss_session(ss_session_id)

    upload_dir = SELF_STUDY_DIR / ss_session_id
    if upload_dir.is_dir():
        import shutil

        shutil.rmtree(upload_dir, ignore_errors=True)
    return {"deleted": ss_session_id}


@app.post(
    "/api/self-study/sessions/{ss_session_id}/upload",
    dependencies=[Depends(rate_limited("upload"))],
)
async def upload_study_documents(
    ss_session_id: str, user: CurrentUser, files: list[UploadFile] = File(...)
):
    await _require_study_session(ss_session_id, user)

    raw = [(f.filename or "", await f.read()) for f in files]
    try:
        validated = validate_batch(raw)
    except UploadRejected as exc:
        raise AppError(code=ErrorCode.UPLOAD_REJECTED, message=str(exc)) from exc

    upload_dir = SELF_STUDY_DIR / ss_session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    namespace = f"ss_{ss_session_id.replace('-', '')}"

    uploaded = []
    for item in validated:
        dest = resolve_within(upload_dir, item.safe_name)
        dest.write_bytes(item.content)

        outcome = await run_blocking(
            services.indexer.index_document,
            dest,
            namespace=namespace,
            domain="SELF_STUDY",
        )
        db.save_ss_document(
            ss_session_id=ss_session_id,
            filename=item.safe_name,
            file_type=item.extension,
            file_size_bytes=item.size_bytes,
            chunk_count=outcome.chunks_indexed,
        )
        db.touch_ss_session(ss_session_id)
        uploaded.append({
            "filename": item.safe_name,
            "chunks_indexed": outcome.chunks_indexed,
            "error": outcome.error,
            "warnings": item.warnings,
        })

    return {"uploaded": uploaded}


@app.delete("/api/self-study/sessions/{ss_session_id}/documents/{doc_id}")
async def delete_study_document(ss_session_id: str, doc_id: int, user: CurrentUser):
    await _require_study_session(ss_session_id, user)
    doc = db.get_ss_document(doc_id)
    if not doc or doc["ss_session_id"] != ss_session_id:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no document {doc_id}")

    namespace = f"ss_{ss_session_id.replace('-', '')}"
    await run_blocking(services.indexer.remove_document, namespace, doc["filename"])
    db.delete_ss_document_record(doc_id)
    db.touch_ss_session(ss_session_id)

    path = SELF_STUDY_DIR / ss_session_id / doc["filename"]
    if path.is_file():
        path.unlink()
    return {"deleted": doc_id, "filename": doc["filename"]}


def _run_study_chat(req: StudyChatRequest) -> dict:
    from agents import PipelineConfig
    from agents.pipeline import Answerer
    from guardrails.output import apply_output_guardrails
    from llm import call_llm, get_usage, start_usage

    start_usage()
    retriever = services.study_retriever(req.ss_session_id)
    retrieval = retriever.retrieve(
        req.query,
        config=_retrieval_config(req.top_k, req.rerank_top_k),
        filenames=req.source_filter,
    )

    answerer = Answerer(call_llm, DOMAINS)
    sub = answerer.answer(
        req.query,
        "Self Study",
        retrieval,
        model=req.model,
        max_tokens=2500,
        history=req.chat_history or [],
        self_study=True,
    )

    verdict = apply_output_guardrails(
        sub.answer,
        [c.text for c in retrieval.chunks],
        sub.evidence.labels,
        check_claims=bool(retrieval.chunks),
    )

    return {
        "final_answer": verdict.answer,
        "refused": verdict.is_refusal or sub.refused,
        "grounding_score": verdict.grounding_score,
        "guardrail_action": verdict.action,
        "sources": [
            {
                "citation_label": label,
                "filename": chunk.metadata.get("filename"),
                "page_number": chunk.metadata.get("page_number"),
                "relevance": round(chunk.relevance, 4),
            }
            for label, chunk in zip(sub.evidence.labels, retrieval.chunks)
        ],
        "debug": {
            "retrieval": retrieval.diagnostics,
            "guardrails": verdict.as_dict(),
            "usage": get_usage().as_dict(),
        },
    }


@app.post("/api/self-study/chat", dependencies=[Depends(rate_limited("chat"))])
async def study_chat(req: StudyChatRequest, user: CurrentUser):
    _validate_model(req.model)
    await _require_study_session(req.ss_session_id, user)

    db.save_ss_message(req.ss_session_id, "user", req.query)
    result = await run_blocking(_run_study_chat, req)
    db.save_ss_message(
        ss_session_id=req.ss_session_id,
        role="assistant",
        content=result["final_answer"],
        quality_score=result.get("grounding_score"),
    )
    return result


# ---------------------------------------------------------------------------
# Evaluation (admin only — each run is hundreds of LLM calls)
# ---------------------------------------------------------------------------


@app.get("/api/evaluate/cases", dependencies=[Depends(rate_limited("read"))])
async def list_test_cases(user: CurrentUser):
    from evaluation import TEST_CASES

    return {
        "test_cases": [
            {
                "id": tc.id, "name": tc.name, "query": tc.query,
                "expected_intent": tc.expected_intent,
                "expected_domains": tc.expected_domains,
                "expected_behavior": tc.expected_behavior,
                "category": tc.category,
            }
            for tc in TEST_CASES
        ]
    }
