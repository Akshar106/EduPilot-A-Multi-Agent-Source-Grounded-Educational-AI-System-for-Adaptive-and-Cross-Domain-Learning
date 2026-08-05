"""
Request bodies
==============
Pydantic models for every route that accepts JSON.

Bounds are set here rather than in the handlers, so an oversized or malformed
body is rejected before any pipeline work starts. Note what is *absent*: no
schema carries a `user_id` or an owner field. Ownership always comes from the
caller's verified token — accepting it from the body is what made the original
endpoints exploitable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from edupilot.core.config import DEFAULT_MODEL, DEFAULT_RERANK_TOP_K, DEFAULT_TOP_K, MAX_QUERY_CHARS


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    # 72 is bcrypt's hard input limit; anything longer is silently truncated
    # by the algorithm, which would make the extra characters security theatre.
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


__all__ = [
    "ChatRequest",
    "CreateStudySessionRequest",
    "LoginRequest",
    "NewSessionRequest",
    "RefreshRequest",
    "RegisterRequest",
    "StudyChatRequest",
]
