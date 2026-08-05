"""Chat session listing, retrieval, deletion, and message truncation."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from edupilot import db
from edupilot.api.deps import CurrentUser
from edupilot.api.schemas import NewSessionRequest
from edupilot.security import AppError, ErrorCode, owns_or_admin, rate_limited

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", dependencies=[Depends(rate_limited("read"))])
async def list_sessions(user: CurrentUser):
    return {"sessions": db.list_sessions(user.user_id, limit=50)}


@router.post("")
async def create_session(req: NewSessionRequest, user: CurrentUser):
    session_id = str(uuid.uuid4())
    db.ensure_session(session_id, user.user_id, title=req.title)
    return {"session_id": session_id}


@router.get("/{session_id}", dependencies=[Depends(rate_limited("read"))])
async def get_session(session_id: str, user: CurrentUser):
    owner = db.get_session_owner(session_id)
    if owner is None:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no session {session_id}")
    owns_or_admin(user, owner, what="chat session")
    return {"session_id": session_id, "messages": db.get_session_messages(session_id)}


@router.delete("/{session_id}")
async def delete_session(session_id: str, user: CurrentUser):
    # Ownership is inside the DELETE predicate, so a miss here means either
    # "no such session" or "not yours" — and the caller learns neither.
    if not db.delete_session(session_id, user.user_id):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            internal=f"user {user.user_id} could not delete session {session_id}",
        )
    return {"deleted": session_id}


@router.delete("/{session_id}/messages/{message_id}")
async def truncate_from_message(session_id: str, message_id: int, user: CurrentUser):
    """Drop a message and everything after it, backing an edit-and-resend."""
    owner = db.get_session_owner(session_id)
    if owner is None:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"no session {session_id}")
    owns_or_admin(user, owner, what="chat session")
    db.delete_messages_from(session_id, message_id)
    return {"truncated": True, "from_message_id": message_id}
