"""
Error envelope
==============
A typed error response that never leaks internals to the client.

The previous handlers did this::

    raise HTTPException(500, f"Pipeline error: {exc}")     # main.py:494, :846

which returns the raw exception string. For this system that can include the
Pinecone index name and host, SQLite paths, provider error bodies (which
sometimes echo request metadata), and stack-adjacent details of which
component failed — a free map of the internals, and occasionally credential
fragments embedded in a provider's own error text.

Here the client gets a stable machine-readable `code`, a message written for
a human, and a `request_id`. The real exception goes to the server log under
that same id, so support can still trace it without publishing it.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """
    Stable error identifiers the frontend can branch on.

    `str, Enum` rather than `StrEnum` for Python 3.10 compatibility.
    """

    def __str__(self) -> str:
        return self.value

    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    VALIDATION_FAILED = "validation_failed"
    UPLOAD_REJECTED = "upload_rejected"
    RATE_LIMITED = "rate_limited"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    UPSTREAM_QUOTA = "upstream_quota"
    RETRIEVAL_FAILED = "retrieval_failed"
    INTERNAL = "internal_error"


#: Client-safe message per code. Deliberately generic: the specific cause is
#: logged, not returned.
_SAFE_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.UNAUTHENTICATED: "Please sign in to continue.",
    ErrorCode.FORBIDDEN: "You do not have permission to do that.",
    ErrorCode.NOT_FOUND: "The requested item was not found.",
    ErrorCode.VALIDATION_FAILED: "The request was not valid.",
    ErrorCode.UPLOAD_REJECTED: "That file could not be accepted.",
    ErrorCode.RATE_LIMITED: "Too many requests. Please wait a moment and try again.",
    ErrorCode.UPSTREAM_UNAVAILABLE: (
        "The AI service is temporarily unavailable. Please try again shortly."
    ),
    ErrorCode.UPSTREAM_QUOTA: (
        "The AI service quota has been reached. Try a different model in settings, "
        "or wait for the quota to reset."
    ),
    ErrorCode.RETRIEVAL_FAILED: "Could not search the knowledge base right now.",
    ErrorCode.INTERNAL: "Something went wrong on our end.",
}

_STATUS: dict[ErrorCode, int] = {
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.VALIDATION_FAILED: 400,
    ErrorCode.UPLOAD_REJECTED: 400,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.UPSTREAM_UNAVAILABLE: 503,
    ErrorCode.UPSTREAM_QUOTA: 429,
    ErrorCode.RETRIEVAL_FAILED: 503,
    ErrorCode.INTERNAL: 500,
}


@dataclass
class AppError(Exception):
    """
    An error safe to serialize to a client.

    Args:
        code: Machine-readable identifier.
        message: Overrides the default client-facing text. Must not contain
            internal detail — put that in `internal`.
        internal: Detail for the log only. Never serialized.
        details: Structured, client-safe extras (e.g. which field failed).
    """

    code: ErrorCode = ErrorCode.INTERNAL
    message: str | None = None
    internal: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        super().__init__(self.client_message)

    @property
    def status_code(self) -> int:
        return _STATUS.get(self.code, 500)

    @property
    def client_message(self) -> str:
        return self.message or _SAFE_MESSAGES.get(self.code, _SAFE_MESSAGES[ErrorCode.INTERNAL])

    def to_response(self) -> dict:
        """The JSON body sent to the client."""
        body: dict[str, Any] = {
            "error": {
                "code": str(self.code),
                "message": self.client_message,
                "request_id": self.request_id,
            }
        }
        if self.details:
            body["error"]["details"] = self.details
        return body

    def log(self) -> None:
        """
        Record the detail server-side, keyed by request_id.

        Severity follows the status code, and a traceback is attached only for
        5xx. A wrong password or a rate-limit rejection is expected traffic,
        not a fault: logging those at ERROR with a full stack trace buries the
        genuine failures under routine noise and makes the log useless for
        alerting.
        """
        status = self.status_code
        detail = self.internal or self.client_message

        if status >= 500:
            logger.error(
                "[%s] %s: %s", self.request_id, self.code, detail,
                exc_info=self.__cause__ is not None,
            )
        elif status == 429:
            logger.info("[%s] %s: %s", self.request_id, self.code, detail)
        else:
            logger.warning("[%s] %s: %s", self.request_id, self.code, detail)


def classify_upstream(exc: Exception) -> AppError:
    """
    Map a provider exception to a safe AppError.

    The provider's own message can echo request content and internal
    identifiers, so it is inspected for classification and then discarded.
    """
    text = str(exc).lower()

    if any(t in text for t in ("quota", "429", "rate limit", "resource_exhausted", "tokens per day")):
        return AppError(
            code=ErrorCode.UPSTREAM_QUOTA,
            internal=f"{type(exc).__name__}: {exc}",
        )
    if any(t in text for t in ("503", "502", "504", "unavailable", "overloaded", "timeout")):
        return AppError(
            code=ErrorCode.UPSTREAM_UNAVAILABLE,
            internal=f"{type(exc).__name__}: {exc}",
        )
    if any(t in text for t in ("api key", "api_key", "401", "authentication", "unauthorized")):
        # A misconfigured server key is an internal fault, not the user's.
        return AppError(
            code=ErrorCode.INTERNAL,
            internal=f"provider auth failure — check server API keys: {type(exc).__name__}",
        )
    return AppError(code=ErrorCode.INTERNAL, internal=f"{type(exc).__name__}: {exc}")
