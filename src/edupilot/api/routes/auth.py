"""Registration, login, refresh, logout, and identity."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from edupilot.api.deps import CurrentUser
from edupilot.api.schemas import LoginRequest, RefreshRequest, RegisterRequest
from edupilot.core.config import IS_PRODUCTION
from edupilot.core.services import services
from edupilot.security import (
    AppError,
    AuthError,
    ErrorCode,
    User,
    create_access_token,
    create_refresh_token,
    rate_limited,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _issue_tokens(user: User, response: Response) -> dict:
    """Mint an access/refresh pair and set the access cookie."""
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


@router.post("/register", dependencies=[Depends(rate_limited("auth"))])
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


@router.post("/login", dependencies=[Depends(rate_limited("auth"))])
async def login(req: LoginRequest, response: Response):
    from edupilot.api.deps import run_blocking

    try:
        # bcrypt verification is deliberately slow; off the event loop it goes.
        user = await run_blocking(services.users.authenticate, req.email, req.password)
    except AuthError as exc:
        raise AppError(
            code=ErrorCode.UNAUTHENTICATED,
            message=exc.message,
            internal=f"failed login for {req.email}",
        ) from exc
    return _issue_tokens(user, response)


@router.post("/refresh", dependencies=[Depends(rate_limited("auth"))])
async def refresh(req: RefreshRequest, response: Response):
    try:
        user = services.users.consume_refresh_token(req.refresh_token)
    except AuthError as exc:
        raise AppError(code=ErrorCode.UNAUTHENTICATED, message=exc.message) from exc
    return _issue_tokens(user, response)


@router.post("/logout")
async def logout(user: CurrentUser, response: Response):
    revoked = services.users.revoke_all_refresh_tokens(user.user_id)
    response.delete_cookie("edupilot_access")
    return {"signed_out": True, "sessions_revoked": revoked}


@router.get("/me")
async def me(user: CurrentUser):
    return {
        "user_id": user.user_id,
        "email": user.email,
        "role": str(user.role),
        "display_name": user.display_name,
    }
