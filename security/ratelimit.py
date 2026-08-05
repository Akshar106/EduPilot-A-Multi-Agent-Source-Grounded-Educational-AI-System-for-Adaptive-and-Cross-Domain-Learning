"""
Rate limiting
=============
Token-bucket limiter keyed by user (or IP for unauthenticated routes).

Nothing throttled the previous API. Two endpoints made that expensive rather
than merely rude:

  POST /api/chat      every call runs router + planner + N generations +
                      verifier against a metered LLM API
  POST /api/evaluate  runs the entire 50-case suite — hundreds of LLM calls —
                      unauthenticated

A single scripted client could exhaust the daily quota in under a minute,
denying service to everyone and, on a paid key, running up the bill.

Buckets are per-process and in-memory. That is honest about the deployment:
one uvicorn process today. Behind multiple workers the effective limit
multiplies by the worker count, so the docstring on `RateLimiter` names Redis
as the swap-in — the interface is designed for it.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimit:
    """A bucket's capacity and refill rate."""

    requests: int
    """Burst capacity."""
    per_seconds: float
    """Window over which the bucket fully refills."""

    @property
    def refill_per_second(self) -> float:
        return self.requests / self.per_seconds

    def __str__(self) -> str:
        return f"{self.requests}/{self.per_seconds:g}s"


#: Per-endpoint-class limits. Chat and evaluation are throttled hardest
#: because each request fans out into many upstream LLM calls.
LIMITS: dict[str, RateLimit] = {
    "chat": RateLimit(requests=20, per_seconds=60),
    "upload": RateLimit(requests=10, per_seconds=300),
    "evaluate": RateLimit(requests=2, per_seconds=3600),
    "auth": RateLimit(requests=8, per_seconds=300),
    "read": RateLimit(requests=120, per_seconds=60),
    "default": RateLimit(requests=60, per_seconds=60),
}


class RateLimitExceeded(Exception):
    """Caller exceeded their allowance."""

    def __init__(self, scope: str, retry_after: float) -> None:
        self.scope = scope
        self.retry_after = max(1, int(retry_after + 0.999))
        super().__init__(
            f"Rate limit exceeded for '{scope}'. Retry in {self.retry_after}s."
        )


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """
    In-memory token-bucket limiter.

    Token buckets are used rather than fixed windows because they permit a
    short burst — a student sending three questions quickly — while still
    bounding the sustained rate. A fixed window would reject the third
    question and also allow a 2x burst across a window boundary.

    Swapping in Redis means reimplementing `_take` against an atomic
    INCR/EXPIRE or a Lua token bucket; nothing else changes.
    """

    def __init__(self, limits: dict[str, RateLimit] | None = None) -> None:
        self._limits = limits or LIMITS
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def limit_for(self, scope: str) -> RateLimit:
        return self._limits.get(scope, self._limits["default"])

    def _sweep(self, now: float) -> None:
        """Drop buckets that have been full and idle. Caller holds the lock."""
        if now - self._last_sweep < 300:
            return
        stale = [
            key for key, bucket in self._buckets.items()
            if now - bucket.last_refill > 3600
        ]
        for key in stale:
            del self._buckets[key]
        self._last_sweep = now
        if stale:
            logger.debug("swept %d idle rate-limit buckets", len(stale))

    def check(self, identity: str, scope: str = "default", *, cost: float = 1.0) -> None:
        """
        Consume `cost` tokens for `identity` in `scope`.

        Raises:
            RateLimitExceeded: not enough tokens are available.
        """
        limit = self.limit_for(scope)
        key = (identity, scope)
        now = time.monotonic()

        with self._lock:
            self._sweep(now)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(limit.requests), last_refill=now)
                self._buckets[key] = bucket

            elapsed = now - bucket.last_refill
            bucket.tokens = min(
                float(limit.requests), bucket.tokens + elapsed * limit.refill_per_second
            )
            bucket.last_refill = now

            if bucket.tokens < cost:
                deficit = cost - bucket.tokens
                retry_after = deficit / limit.refill_per_second
                logger.info(
                    "rate limit hit: identity=%s scope=%s limit=%s retry_after=%.1fs",
                    identity[:24], scope, limit, retry_after,
                )
                raise RateLimitExceeded(scope, retry_after)

            bucket.tokens -= cost

    def remaining(self, identity: str, scope: str = "default") -> int:
        """Tokens currently available. For the X-RateLimit-Remaining header."""
        limit = self.limit_for(scope)
        with self._lock:
            bucket = self._buckets.get((identity, scope))
            if bucket is None:
                return limit.requests
            elapsed = time.monotonic() - bucket.last_refill
            return int(
                min(float(limit.requests), bucket.tokens + elapsed * limit.refill_per_second)
            )

    def reset(self, identity: str | None = None) -> None:
        """Clear buckets. Test support, and for lifting a limit manually."""
        with self._lock:
            if identity is None:
                self._buckets.clear()
            else:
                for key in [k for k in self._buckets if k[0] == identity]:
                    del self._buckets[key]


#: Process-wide limiter.
limiter = RateLimiter()
