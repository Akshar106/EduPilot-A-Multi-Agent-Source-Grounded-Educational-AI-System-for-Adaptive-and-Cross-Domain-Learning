"""The main course-chat endpoint: the full multi-agent pipeline."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from edupilot import db
from edupilot.api.deps import CurrentUser, retrieval_config, run_blocking, validate_model
from edupilot.api.schemas import ChatRequest
from edupilot.core.config import VERIFY_MODEL
from edupilot.core.services import services
from edupilot.security import owns_or_admin, rate_limited

router = APIRouter(prefix="/api", tags=["chat"])


def _run_chat(req: ChatRequest, session_id: str) -> dict:
    """
    Synchronous pipeline invocation. Runs in the worker pool, never inline.

    Imported lazily so that importing the router does not drag in the agent
    and retrieval stacks — which would make app startup load models.
    """
    from edupilot.agents import PipelineConfig
    from edupilot.llm import get_usage, start_usage

    start_usage()
    cfg = PipelineConfig(
        model=req.model,
        verify_model=VERIFY_MODEL,
        retrieval=retrieval_config(req.top_k, req.rerank_top_k),
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
        # floored — the previous version applied max(quality, 0.75), which
        # reported a passing score for answers that had not been checked.
        "grounding_score": result.grounding_score,
        "guardrail_action": result.verdict.action if result.verdict else None,
        "sources": result.sources,
        "debug": diagnostics,
    }


@router.post("/chat", dependencies=[Depends(rate_limited("chat"))])
async def chat(req: ChatRequest, user: CurrentUser):
    validate_model(req.model)

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
