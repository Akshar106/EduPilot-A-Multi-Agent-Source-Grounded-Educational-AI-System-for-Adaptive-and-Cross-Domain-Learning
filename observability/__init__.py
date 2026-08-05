"""
EduPilot observability
======================
Structured logging with request correlation.

The previous system logged with bare `print(..., file=sys.stderr)` scattered
through `utils.py` and `synthesizer.py`. Those lines had no level, no
timestamp, no module, and no way to associate them with the request that
produced them — so with two concurrent requests the interleaved output was
unreadable, and nothing could be filtered or shipped anywhere.

A request id is generated per HTTP request, stored in a `ContextVar`, and
attached to every log record emitted while handling it, including records
from worker threads (`main.run_blocking` propagates the value across the
thread boundary).

Set `LOG_FORMAT=json` for machine-readable output; the default is
human-readable for local development.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

#: Correlates every log line emitted while handling one request.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    """Attach the current request id to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Anything attached via `extra=` on the call site.
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Compact, aligned output for a terminal."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s [%(request_id)s] %(name)-22s %(message)s",
            datefmt="%H:%M:%S",
        )


_configured = False


def configure_logging(level: str | None = None) -> None:
    """
    Install handlers and formatters. Idempotent.

    Third-party loggers that are noisy at INFO are raised to WARNING so the
    application's own lines stay visible.
    """
    global _configured
    if _configured:
        return

    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    use_json = os.getenv("LOG_FORMAT", "human").lower() == "json"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if use_json else HumanFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

    for noisy, noise_level in {
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "urllib3": logging.WARNING,
        "sentence_transformers": logging.WARNING,
        "transformers": logging.ERROR,
        "pdfminer": logging.ERROR,
        "PIL": logging.WARNING,
        "huggingface_hub": logging.WARNING,
    }.items():
        logging.getLogger(noisy).setLevel(noise_level)

    _configured = True


@contextmanager
def RequestContext(request_id: str) -> Iterator[str]:
    """Bind a request id for the duration of a block."""
    token = request_id_var.set(request_id)
    try:
        yield request_id
    finally:
        request_id_var.reset(token)


@contextmanager
def timed(logger: logging.Logger, operation: str, **fields: Any) -> Iterator[dict]:
    """
    Log how long a block took, and let it record extra fields.

        with timed(logger, "retrieval", domain="AML") as span:
            span["chunks"] = len(result.chunks)

    The duration is emitted even when the block raises, so a slow failure is
    as visible as a slow success.
    """
    span: dict[str, Any] = dict(fields)
    started = time.perf_counter()
    try:
        yield span
    finally:
        span["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "%s %s",
            operation,
            " ".join(f"{k}={v}" for k, v in span.items()),
            extra={f"ctx_{k}": v for k, v in span.items()},
        )


__all__ = [
    "HumanFormatter",
    "JsonFormatter",
    "RequestContext",
    "RequestIdFilter",
    "configure_logging",
    "request_id_var",
    "timed",
]
