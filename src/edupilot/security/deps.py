"""
FastAPI security dependencies
=============================
Wires auth, roles, and rate limiting into route signatures.

The point of expressing these as dependencies is that access control becomes
part of the route's type signature rather than something a handler has to
remember to do. `main.py` previously trusted a client-supplied `session_id`
in the request body; here the caller's identity comes from `Depends(current_user)`
and cannot be set by the request.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Header, Request

from .auth import AuthError, Role, User, decode_access_token
from .errors import AppError, ErrorCode
from .ratelimit import RateLimitExceeded, limiter

logger = logging.getLogger(__name__)

#: The identity every request resolves to when AUTH_REQUIRED is false.
#:
#: A stable id, not a random one — sessions and study uploads are stored
#: against it, so it has to survive a restart or the operator's history
#: disappears. Admin, because a single-user install still needs to manage its
#: own knowledge base.
#:
#: config.py refuses to start with auth disabled in production, so this can
#: never become the identity of an anonymous internet caller.
LOCAL_USER = User(
    user_id="local-single-user",
    email="local@edupilot.invalid",
    role=Role.ADMIN,
    display_name="Local User",
)


def client_ip(request: Request) -> str:
    """
    Best-effort client address for unauthenticated rate limiting.

    `X-Forwarded-For` is honoured only when the app is explicitly configured
    behind a trusted proxy — the header is client-controlled, so trusting it
    unconditionally lets anyone forge a fresh identity per request and bypass
    the limiter entirely.
    """
    from edupilot.core.config import TRUST_PROXY_HEADERS

    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


async def current_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    """
    Resolve the authenticated user, or reject.

    Accepts a bearer token, falling back to the `edupilot_access` cookie so a
    browser session works without JavaScript holding the token.

    When AUTH_REQUIRED is false this never rejects: a valid token still wins,
    so a signed-in operator keeps their own identity, but anything else
    resolves to LOCAL_USER instead of a 401.
    """
    from edupilot.core.config import AUTH_REQUIRED

    token = _bearer_token(authorization) or request.cookies.get("edupilot_access")

    if token:
        try:
            return decode_access_token(token)
        except AuthError as exc:
            if AUTH_REQUIRED:
                raise AppError(
                    code=ErrorCode.UNAUTHENTICATED, message=exc.message, internal=str(exc)
                ) from exc
            # Expired or malformed token in single-user mode: fall through
            # rather than locking the operator out of their own install.
            logger.debug("ignoring unusable token in single-user mode: %s", exc)

    if not AUTH_REQUIRED:
        return LOCAL_USER

    raise AppError(code=ErrorCode.UNAUTHENTICATED, internal="no token presented")


async def optional_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> User | None:
    """Resolve the user if a valid token is present, else None."""
    try:
        return await current_user(request, authorization)
    except AppError:
        return None


async def admin_user(user: Annotated[User, Depends(current_user)]) -> User:
    """
    Require the admin role.

    Guards the shared course knowledge base. Previously any caller could
    upload into it, which meant any visitor could inject documents that every
    student's answers would then be grounded in.
    """
    if not user.is_admin:
        raise AppError(
            code=ErrorCode.FORBIDDEN,
            message="Managing the course knowledge base requires an instructor account.",
            internal=f"user {user.user_id} (role={user.role}) attempted an admin action",
        )
    return user


def rate_limited(scope: str, *, cost: float = 1.0) -> Callable:
    """
    Build a dependency enforcing the rate limit for `scope`.

    Authenticated callers are limited per user id; anonymous callers per IP,
    so one signed-in student cannot be throttled by another's traffic from the
    same network.

        @app.post("/api/chat", dependencies=[Depends(rate_limited("chat"))])
    """

    async def dependency(
        request: Request,
        user: Annotated[User | None, Depends(optional_user)] = None,
    ) -> None:
        identity = f"user:{user.user_id}" if user else f"ip:{client_ip(request)}"
        try:
            limiter.check(identity, scope, cost=cost)
        except RateLimitExceeded as exc:
            raise AppError(
                code=ErrorCode.RATE_LIMITED,
                message=f"Too many requests. Please wait {exc.retry_after}s and try again.",
                internal=f"{identity} exceeded '{scope}'",
                details={"retry_after": exc.retry_after, "scope": scope},
            ) from exc

    return dependency


def owns_or_admin(user: User, owner_id: str, *, what: str = "resource") -> None:
    """
    Assert the caller owns a resource, or is an admin.

    Called at the top of every handler that reads or mutates user-scoped data.
    This is the check whose absence made every chat session and uploaded
    document readable by any visitor.

    Raises:
        AppError: NOT_FOUND rather than FORBIDDEN when the caller is not the
            owner. Returning 403 would confirm the resource exists, turning
            the endpoint into an enumeration oracle.
    """
    if user.user_id != owner_id and not user.is_admin:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            internal=(
                f"user {user.user_id} attempted to access {what} owned by {owner_id}"
            ),
        )
