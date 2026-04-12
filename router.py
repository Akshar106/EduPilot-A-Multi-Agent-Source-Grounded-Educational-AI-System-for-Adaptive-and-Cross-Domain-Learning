"""
EduPilot Router
===============
Classifies incoming student queries by:
  1. Intent type  — "single" | "multi"
  2. Domain(s)    — ["AML"], ["ADT"], ["STAT"], or multi-domain list
  3. Flags        — is_course_related, needs_clarification

Uses Claude to make these decisions, with a lightweight keyword-based
fallback so the system degrades gracefully even without API access.
"""

from __future__ import annotations

from config import DEFAULT_MODEL, DOMAINS, LLM_MAX_TOKENS_CLASSIFY
from prompts import (
    CLARIFICATION_RESPONSE,
    OUT_OF_DOMAIN_RESPONSE,
    ROUTER_SYSTEM,
    ROUTER_USER,
)
from utils import call_llm, parse_json_response


# ---------------------------------------------------------------------------
# Router result dataclass
# ---------------------------------------------------------------------------

class RouterResult:
    __slots__ = (
        "intent_type",
        "domains",
        "is_course_related",
        "needs_clarification",
        "clarification_hint",
        "reasoning",
        "raw_response",
    )

    def __init__(
        self,
        intent_type: str = "single",
        domains: list[str] | None = None,
        is_course_related: bool = True,
        needs_clarification: bool = False,
        clarification_hint: str | None = None,
        reasoning: str = "",
        raw_response: str = "",
    ):
        self.intent_type = intent_type
        self.domains = domains or []
        self.is_course_related = is_course_related
        self.needs_clarification = needs_clarification
        self.clarification_hint = clarification_hint
        self.reasoning = reasoning
        self.raw_response = raw_response

    def __repr__(self) -> str:
        return (
            f"RouterResult(intent={self.intent_type!r}, domains={self.domains}, "
            f"course_related={self.is_course_related}, clarify={self.needs_clarification})"
        )


# ---------------------------------------------------------------------------
# Keyword fallback (used when API unavailable or as double-check)
# ---------------------------------------------------------------------------

def _keyword_classify(query: str) -> RouterResult:
    """
    Lightweight keyword-based domain classification.
    Used as a fallback when the LLM is unavailable.
    """
    q_lower = query.lower()
    detected: list[str] = []

    for domain, cfg in DOMAINS.items():
        for kw in cfg["keywords"]:
            if kw.lower() in q_lower:
                detected.append(domain)
                break

    intent = "multi" if len(detected) > 1 else "single"
    is_related = bool(detected)
    needs_clarification = len(q_lower.split()) < 4 and not detected

    return RouterResult(
        intent_type=intent,
        domains=list(dict.fromkeys(detected)),  # preserve order, dedupe
        is_course_related=is_related,
        needs_clarification=needs_clarification,
        clarification_hint="Please provide more context." if needs_clarification else None,
        reasoning="Keyword-based fallback classification.",
    )


# ---------------------------------------------------------------------------
# LLM-based router
# ---------------------------------------------------------------------------

def classify_query(
    query: str,
    model: str = DEFAULT_MODEL,
    chat_history: list[dict] | None = None,
) -> RouterResult:
    """
    Classify the student query using an LLM.

    Args:
        query: The student's raw question.
        model: Claude model to use.
        chat_history: Previous conversation turns for context.

    Returns:
        RouterResult with intent, domains, and flags.
    """
    # Build context-aware query if chat history is available
    effective_query = query
    if chat_history:
        recent = chat_history[-3:]  # last 3 turns
        ctx_lines = []
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if isinstance(content, str):
                ctx_lines.append(f"{role.capitalize()}: {content[:200]}")
        if ctx_lines:
            effective_query = (
                "[Previous conversation context]\n"
                + "\n".join(ctx_lines)
                + f"\n\n[Current question]\n{query}"
            )

    user_prompt = ROUTER_USER.format(query=effective_query)

    try:
        raw = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            system=ROUTER_SYSTEM,
            model=model,
            max_tokens=LLM_MAX_TOKENS_CLASSIFY,
        )
        data = parse_json_response(raw)
    except Exception as exc:
        # Graceful degradation to keyword fallback
        result = _keyword_classify(query)
        result.reasoning = f"API error ({exc}); used keyword fallback."
        result.raw_response = str(exc)
        return result

    if not data:
        result = _keyword_classify(query)
        result.reasoning = "JSON parse failed; used keyword fallback."
        result.raw_response = raw
        return result

    # Validate and coerce fields
    intent_type = data.get("intent_type", "single")
    if intent_type not in ("single", "multi"):
        intent_type = "single"

    raw_domains = data.get("domains", [])
    domains = [d for d in raw_domains if d in DOMAINS]

    # If LLM returned no domains but query seems related, use keyword fallback domains
    if not domains and data.get("is_course_related", True):
        fallback = _keyword_classify(query)
        domains = fallback.domains

    return RouterResult(
        intent_type=intent_type,
        domains=domains,
        is_course_related=bool(data.get("is_course_related", True)),
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_hint=data.get("clarification_hint"),
        reasoning=data.get("reasoning", ""),
        raw_response=raw if "raw" in locals() else "",
    )


# ---------------------------------------------------------------------------
# Quick helpers for the app layer
# ---------------------------------------------------------------------------

def should_ask_for_clarification(result: RouterResult, query: str = "") -> bool:
    # Hard override: if keyword fallback detects domains, never ask for clarification.
    # This prevents the LLM from being overly cautious on clear questions.
    if result.domains:
        return False
    if query:
        kw_result = _keyword_classify(query)
        if kw_result.domains:
            return False
    # Only ask if truly no domains detected AND query is very short/vague
    return result.needs_clarification or (
        result.is_course_related and not result.domains
        and len(query.split()) < 5
    )


def get_clarification_message(result: RouterResult) -> str:
    if result.clarification_hint:
        return f"{CLARIFICATION_RESPONSE}\n\n*Hint: {result.clarification_hint}*"
    return CLARIFICATION_RESPONSE


def get_out_of_domain_message() -> str:
    return OUT_OF_DOMAIN_RESPONSE
