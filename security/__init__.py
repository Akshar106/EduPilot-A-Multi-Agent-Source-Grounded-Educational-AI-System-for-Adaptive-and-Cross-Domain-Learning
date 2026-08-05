"""
EduPilot security
=================
Authentication, authorization, upload validation, rate limiting, and the
error envelope.

    from security import current_user, admin_user, rate_limited, AppError

    @app.post("/api/chat", dependencies=[Depends(rate_limited("chat"))])
    async def chat(req: ChatRequest, user: User = Depends(current_user)):
        ...

Every user-scoped handler must call `owns_or_admin(user, owner_id)` before
reading or mutating a record. That check is what closes the IDOR that made
all chat sessions and uploaded documents publicly readable.
"""

from .auth import (
    ACCESS_TOKEN_TTL,
    REFRESH_TOKEN_TTL,
    AuthError,
    Role,
    User,
    UserStore,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from .deps import (
    admin_user,
    client_ip,
    current_user,
    optional_user,
    owns_or_admin,
    rate_limited,
)
from .errors import AppError, ErrorCode, classify_upstream
from .ratelimit import LIMITS, RateLimit, RateLimitExceeded, RateLimiter, limiter
from .uploads import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_BYTES,
    UploadRejected,
    ValidatedUpload,
    resolve_within,
    safe_filename,
    validate_batch,
    validate_upload,
)

__all__ = [
    "ACCESS_TOKEN_TTL",
    "ALLOWED_EXTENSIONS",
    "AppError",
    "AuthError",
    "ErrorCode",
    "LIMITS",
    "MAX_FILE_BYTES",
    "REFRESH_TOKEN_TTL",
    "RateLimit",
    "RateLimitExceeded",
    "RateLimiter",
    "Role",
    "UploadRejected",
    "User",
    "UserStore",
    "ValidatedUpload",
    "admin_user",
    "classify_upstream",
    "client_ip",
    "create_access_token",
    "create_refresh_token",
    "current_user",
    "decode_access_token",
    "hash_password",
    "limiter",
    "optional_user",
    "owns_or_admin",
    "rate_limited",
    "resolve_within",
    "safe_filename",
    "validate_batch",
    "validate_upload",
    "verify_password",
]
