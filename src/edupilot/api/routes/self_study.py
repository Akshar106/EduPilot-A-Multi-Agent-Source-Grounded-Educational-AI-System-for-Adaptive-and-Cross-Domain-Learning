"""
Self Study
==========
A student's private documents and the chat over them.

Each session gets its own vector namespace derived from its id, so one
student's uploads are physically separated from another's rather than filtered
apart at query time. Every route re-checks ownership.
"""

from __future__ import annotations

import shutil
import uuid

from fastapi import APIRouter, Depends, File, UploadFile

from edupilot import db
from edupilot.api.deps import CurrentUser, retrieval_config, run_blocking, validate_model
from edupilot.api.schemas import CreateStudySessionRequest, StudyChatRequest
from edupilot.core.config import DOMAINS, SELF_STUDY_DIR
from edupilot.core.services import services
from edupilot.security import (
    AppError,
    ErrorCode,
    User,
    owns_or_admin,
    rate_limited,
    resolve_within,
    validate_batch,
)
from edupilot.security.uploads import UploadRejected

router = APIRouter(prefix="/api/self-study", tags=["self-study"])


def _namespace(ss_session_id: str) -> str:
    return f"ss_{ss_session_id.replace('-', '')}"


async def _require_study_session(ss_session_id: str, user: User) -> dict:
    session = db.get_ss_session(ss_session_id)
    if not session:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no study session {ss_session_id}")
    owns_or_admin(user, db.get_ss_session_owner(ss_session_id) or "", what="study session")
    return session


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@router.post("/sessions")
async def create_study_session(req: CreateStudySessionRequest, user: CurrentUser):
    ss_session_id = str(uuid.uuid4())
    db.create_ss_session(ss_session_id, user.user_id, req.name.strip(), req.description)
    return {"ss_session_id": ss_session_id, "name": req.name.strip()}


@router.get("/sessions", dependencies=[Depends(rate_limited("read"))])
async def list_study_sessions(user: CurrentUser):
    return {"sessions": db.list_ss_sessions(user.user_id)}


@router.get("/sessions/{ss_session_id}", dependencies=[Depends(rate_limited("read"))])
async def get_study_session(ss_session_id: str, user: CurrentUser):
    session = await _require_study_session(ss_session_id, user)
    return {
        "session": session,
        "documents": db.list_ss_documents(ss_session_id),
        "messages": db.get_ss_messages(ss_session_id),
    }


@router.delete("/sessions/{ss_session_id}")
async def delete_study_session(ss_session_id: str, user: CurrentUser):
    await _require_study_session(ss_session_id, user)

    await run_blocking(services.store.delete, _namespace(ss_session_id), delete_all=True)
    db.delete_ss_session(ss_session_id)

    upload_dir = SELF_STUDY_DIR / ss_session_id
    if upload_dir.is_dir():
        shutil.rmtree(upload_dir, ignore_errors=True)
    return {"deleted": ss_session_id}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@router.post("/sessions/{ss_session_id}/upload", dependencies=[Depends(rate_limited("upload"))])
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
    namespace = _namespace(ss_session_id)

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


@router.delete("/sessions/{ss_session_id}/documents/{doc_id}")
async def delete_study_document(ss_session_id: str, doc_id: int, user: CurrentUser):
    await _require_study_session(ss_session_id, user)
    doc = db.get_ss_document(doc_id)
    if not doc or doc["ss_session_id"] != ss_session_id:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no document {doc_id}")

    await run_blocking(services.indexer.remove_document, _namespace(ss_session_id), doc["filename"])
    db.delete_ss_document_record(doc_id)
    db.touch_ss_session(ss_session_id)

    path = SELF_STUDY_DIR / ss_session_id / doc["filename"]
    if path.is_file():
        path.unlink()
    return {"deleted": doc_id, "filename": doc["filename"]}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def _run_study_chat(req: StudyChatRequest) -> dict:
    """
    Single-domain answer over the student's own uploads.

    Deliberately not the full pipeline: there is nothing to route or split
    across domains, so this goes straight from retrieval to answering, then
    through the same output guardrails as course chat.
    """
    from edupilot.agents.pipeline import Answerer
    from edupilot.guardrails.output import apply_output_guardrails
    from edupilot.llm import call_llm, get_usage, start_usage

    start_usage()
    retriever = services.study_retriever(req.ss_session_id)
    retrieval = retriever.retrieve(
        req.query,
        config=retrieval_config(req.top_k, req.rerank_top_k),
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


@router.post("/chat", dependencies=[Depends(rate_limited("chat"))])
async def study_chat(req: StudyChatRequest, user: CurrentUser):
    validate_model(req.model)
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
