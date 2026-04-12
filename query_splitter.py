"""
EduPilot Query Splitter
=======================
Decomposes multi-intent student queries into self-contained sub-questions,
each mapped to exactly one domain.

For single-intent queries, it returns the original query as a single sub-question
(no LLM call needed).
"""

from __future__ import annotations

from config import DEFAULT_MODEL, DOMAINS, LLM_MAX_TOKENS_CLASSIFY
from prompts import SPLITTER_SYSTEM, SPLITTER_USER
from utils import call_llm, parse_json_response


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_query(
    query: str,
    intent_type: str,
    detected_domains: list[str],
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """
    Split a query into sub-questions with domain assignments.

    For single-intent queries: returns [{question, domain}] without an LLM call.
    For multi-intent queries: calls the LLM to decompose the query.

    Args:
        query: Original student question.
        intent_type: "single" or "multi" from the router.
        detected_domains: Domain list from the router.
        model: Claude model to use.

    Returns:
        List of dicts: [{question: str, domain: str, reasoning: str}]
    """
    if intent_type == "single":
        domain = detected_domains[0] if detected_domains else _guess_domain(query)
        return [{"question": query, "domain": domain, "reasoning": "Single-intent query."}]

    # Multi-intent: ask LLM to decompose
    return _llm_split(query, detected_domains, model)


def _llm_split(
    query: str,
    detected_domains: list[str],
    model: str,
) -> list[dict]:
    """Use an LLM to split a multi-intent query."""
    domains_str = ", ".join(detected_domains) if detected_domains else "AML, ADT, STAT"
    user_prompt = SPLITTER_USER.format(query=query, domains=domains_str)

    try:
        raw = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            system=SPLITTER_SYSTEM,
            model=model,
            max_tokens=LLM_MAX_TOKENS_CLASSIFY,
        )
        data = parse_json_response(raw)
    except Exception:
        return _fallback_split(query, detected_domains)

    if not data or "sub_questions" not in data:
        return _fallback_split(query, detected_domains)

    sub_questions: list[dict] = []
    for sq in data["sub_questions"]:
        question = sq.get("question", "").strip()
        domain = sq.get("domain", "").upper()
        reasoning = sq.get("reasoning", "")

        if not question:
            continue
        if domain not in DOMAINS:
            domain = _guess_domain(question)
        sub_questions.append({
            "question": question,
            "domain": domain,
            "reasoning": reasoning,
        })

    if not sub_questions:
        return _fallback_split(query, detected_domains)

    return sub_questions


def _fallback_split(query: str, domains: list[str]) -> list[dict]:
    """
    Fallback: split on sentence boundaries and assign domains in order.
    Used when LLM decomposition fails.
    """
    import re

    # Split on sentence-ending punctuation
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", query) if s.strip()]
    if len(sentences) <= 1 or not domains:
        domain = domains[0] if domains else _guess_domain(query)
        return [{"question": query, "domain": domain, "reasoning": "Fallback: single part."}]

    # Pair sentences with domains (cycle through domains if more sentences than domains)
    result = []
    for i, sentence in enumerate(sentences):
        domain = domains[i % len(domains)]
        result.append({
            "question": sentence,
            "domain": domain,
            "reasoning": f"Fallback split, assigned to {domain}.",
        })
    return result


def _guess_domain(text: str) -> str:
    """
    Simple keyword heuristic to guess domain when the router didn't provide one.
    Returns the most likely domain, defaulting to "AML".
    """
    from config import DOMAINS as _DOMAINS

    text_lower = text.lower()
    scores: dict[str, int] = {d: 0 for d in _DOMAINS}

    for domain, cfg in _DOMAINS.items():
        for kw in cfg["keywords"]:
            if kw.lower() in text_lower:
                scores[domain] += 1

    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "AML"
