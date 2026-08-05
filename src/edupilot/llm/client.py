"""
LLM client
==========
One entry point for every model call, with provider detection, bounded
retries, and token accounting.

Replaces the caller in `utils.py`. Changes that matter:

  * **Bounded blocking.** The old Gemini path called `time.sleep(15)` inside
    the request thread on a 429. With a four-worker pool, three rate-limited
    requests could stall the entire API. Backoff is now capped and the total
    wait per call is bounded.
  * **Fallback is explicit.** The old chain tried every Gemini model, then
    Groq, with the logic spread across two functions and a bare `raise` that
    could surface either provider's exception. The chain is now a list, tried
    in order, with the reason for each hop logged.
  * **Usage is recorded.** Nothing tracked tokens before, so there was no way
    to know what a request cost or which stage was expensive.
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field

from edupilot.core.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    GROQ_MODELS,
    LLM_MAX_TOKENS_CLASSIFY,
)

logger = logging.getLogger(__name__)

#: Total seconds any single `call_llm` will spend sleeping on retries.
MAX_TOTAL_BACKOFF = 12.0
MAX_ATTEMPTS_PER_MODEL = 3

_GROQ_PREFIXES = ("llama", "mixtral", "gemma", "qwen", "deepseek", "kimi", "moonshot")


def is_groq_model(model: str) -> bool:
    return model in GROQ_MODELS or model.lower().startswith(_GROQ_PREFIXES)


def is_gemini_model(model: str) -> bool:
    return model.lower().startswith("gemini")


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    """Token and call counts for one request, aggregated across stages."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    fallbacks: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens
        self.by_model[model] = self.by_model.get(model, 0) + 1

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "fallbacks": self.fallbacks,
            "by_model": self.by_model,
        }


#: Per-thread usage, so a request's accounting is not polluted by concurrent
#: requests sharing the worker pool.
_usage = threading.local()


def start_usage() -> Usage:
    """Begin accounting for the current thread. Call once per request."""
    _usage.current = Usage()
    return _usage.current


def get_usage() -> Usage:
    current = getattr(_usage, "current", None)
    if current is None:
        current = Usage()
        _usage.current = current
    return current


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Every model in the fallback chain failed."""


def _is_quota(message: str) -> bool:
    m = message.lower()
    return any(t in m for t in ("429", "quota", "rate limit", "resource_exhausted", "too many"))


def _is_transient(message: str) -> bool:
    m = message.lower()
    return any(t in m for t in ("500", "502", "503", "504", "unavailable", "overloaded", "timeout"))


def _is_daily_quota(message: str) -> bool:
    m = message.lower()
    return any(t in m for t in ("perday", "per_day", "daily", "tokens per day", "tpd"))


def _retry_delay(message: str, attempt: int) -> float:
    """
    Delay before the next attempt.

    Honours a server-supplied "retry in Ns" hint when present, otherwise
    exponential backoff with jitter. Capped so one slow provider cannot pin a
    worker thread for the length of its suggested wait.
    """
    hinted = re.search(r"retry (?:in|after) (\d+(?:\.\d+)?)\s*s", message, re.IGNORECASE)
    if hinted:
        return min(float(hinted.group(1)) + 0.5, 8.0)
    return min(0.75 * (2**attempt) + random.uniform(0, 0.5), 8.0)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _call_groq(messages: list[dict], system: str | None, model: str, max_tokens: int) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise LLMError("GROQ_API_KEY is not set")

    payload = ([{"role": "system", "content": system}] if system else []) + list(messages)
    response = Groq(api_key=api_key).chat.completions.create(
        model=model, messages=payload, max_tokens=max_tokens, temperature=0.1
    )

    usage = getattr(response, "usage", None)
    get_usage().record(
        model,
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )
    return response.choices[0].message.content or ""


def _call_gemini(messages: list[dict], system: str | None, model: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set")

    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]

    kwargs: dict = {
        "max_output_tokens": max_tokens,
        "temperature": 0.1,
        "system_instruction": system,
    }
    # Gemini 2.5 reasons by default; this pipeline wants deterministic
    # extraction and grading, not exploration, and thinking tokens count
    # against the output budget.
    if "2.5" in model:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    response = genai.Client(api_key=api_key).models.generate_content(
        model=model, contents=contents, config=types.GenerateContentConfig(**kwargs)
    )

    meta = getattr(response, "usage_metadata", None)
    get_usage().record(
        model,
        getattr(meta, "prompt_token_count", 0) or 0,
        getattr(meta, "candidates_token_count", 0) or 0,
    )
    return response.text or ""


def _dispatch(messages: list[dict], system: str | None, model: str, max_tokens: int) -> str:
    if is_groq_model(model):
        return _call_groq(messages, system, model, max_tokens)
    if is_gemini_model(model):
        return _call_gemini(messages, system, model, max_tokens)
    raise LLMError(f"Unknown model provider for {model!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_fallback_chain(model: str) -> list[str]:
    """
    Models to try, in order, starting with the requested one.

    Same-provider alternatives come first (a quota hit is usually per-model),
    then the other provider, since a provider-wide outage needs a different
    vendor rather than a different model.
    """
    chain = [model]
    same = [m for m in AVAILABLE_MODELS if m != model and is_groq_model(m) == is_groq_model(model)]
    other = [m for m in AVAILABLE_MODELS if m != model and is_groq_model(m) != is_groq_model(model)]
    for candidate in same + other:
        if candidate not in chain:
            chain.append(candidate)
    return chain


def call_llm(
    messages: list[dict],
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = LLM_MAX_TOKENS_CLASSIFY,
) -> str:
    """
    Call a model, retrying and falling back as needed.

    Args:
        messages: Chat messages, each `{"role": ..., "content": ...}`.
        system: System instruction.
        model: Preferred model. Provider is inferred from the name.
        max_tokens: Output cap.

    Returns:
        The completion text.

    Raises:
        LLMError: every model in the chain failed. The message names the last
            failure; per-model reasons are in the log.
    """
    chain = build_fallback_chain(model)
    usage = get_usage()
    budget = MAX_TOTAL_BACKOFF
    last_error: Exception | None = None

    for position, candidate in enumerate(chain):
        for attempt in range(MAX_ATTEMPTS_PER_MODEL):
            try:
                text = _dispatch(messages, system, candidate, max_tokens)
                if position > 0:
                    usage.fallbacks += 1
                    logger.info("fell back from %s to %s", model, candidate)
                return text
            except Exception as exc:
                last_error = exc
                message = str(exc)

                # A missing key or a bad request will not improve on retry.
                if not (_is_quota(message) or _is_transient(message)):
                    logger.warning("%s failed permanently: %s", candidate, message[:200])
                    break

                # A daily cap will not clear within this request.
                if _is_daily_quota(message):
                    logger.info("%s daily quota exhausted — trying next model", candidate)
                    break

                if attempt == MAX_ATTEMPTS_PER_MODEL - 1:
                    break

                delay = _retry_delay(message, attempt)
                if delay > budget:
                    logger.info(
                        "%s needs %.1fs backoff, only %.1fs of budget left — next model",
                        candidate, delay, budget,
                    )
                    break

                budget -= delay
                usage.retries += 1
                logger.info(
                    "%s attempt %d failed (%s) — retrying in %.1fs",
                    candidate, attempt + 1, message[:120], delay,
                )
                time.sleep(delay)

    raise LLMError(f"All {len(chain)} models failed. Last error: {last_error}") from last_error


def parse_json_response(raw: str) -> dict:
    """Re-exported from agents.contracts so callers need only one import."""
    from edupilot.agents.contracts import parse_json_response as _parse

    return _parse(raw)
