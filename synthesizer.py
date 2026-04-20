"""
EduPilot Synthesizer
====================
Two responsibilities:
  1. Domain Agent  — generate a grounded answer for one domain's sub-question
  2. Cross-Domain Synthesizer — merge multiple domain answers into one response

Both use Claude and stay strictly within the retrieved evidence.
"""

from __future__ import annotations

from config import DEFAULT_MODEL, DOMAINS, LLM_MAX_TOKENS_GENERATE, LLM_MAX_TOKENS_SS, LLM_MAX_TOKENS_SYNTH


def _friendly_error(exc: Exception) -> str:
    """Convert raw API exceptions into clean user-facing messages."""
    import sys
    print(f"\n[EduPilot ERROR] {type(exc).__name__}: {exc}\n", file=sys.stderr, flush=True)
    msg = str(exc).lower()
    if "rate limit" in msg or "429" in msg or "quota" in msg or "tokens per day" in msg or "tpd" in msg:
        is_daily = "perday" in msg or "per_day" in msg or "daily" in msg
        if is_daily:
            return (
                "⚠️ **Daily quota exhausted** — you've used all free requests for today on this model.\n\n"
                "**What you can do:**\n"
                "- Switch to **`gemini-2.0-flash-lite`** in the sidebar (separate daily bucket, 1500 req/day)\n"
                "- Wait until **midnight UTC** for the daily quota to reset\n"
                "- Get a fresh API key at [Google AI Studio](https://aistudio.google.com)"
            )
        return (
            "⚠️ **Per-minute rate limit hit** — too many requests in the last 60 seconds.\n\n"
            "Please wait ~15 seconds and try again. EduPilot auto-retries on per-minute limits."
        )
    if "authentication" in msg or "401" in msg or "api key" in msg or "api_key" in msg:
        return "⚠️ **API authentication error** — please check your `GEMINI_API_KEY` in the `.env` file."
    if "gemini_api_key" in msg or "not set" in msg:
        return "⚠️ **GEMINI_API_KEY not configured** — add your key to the `.env` file and restart the server. Get a free key at [aistudio.google.com](https://aistudio.google.com)."
    if "timeout" in msg or "connection" in msg:
        return "⚠️ **Connection timeout** — the AI service is temporarily unavailable. Please try again in a moment."
    return f"⚠️ **Error generating answer** — {exc}"
from prompts import (
    DOMAIN_AGENT_SYSTEM,
    DOMAIN_AGENT_USER,
    DOMAIN_AGENT_USER_NO_CONTEXT,
    NO_EVIDENCE_RESPONSE,
    SS_AGENT_SYSTEM,
    SS_AGENT_USER,
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_USER,
)
from utils import (
    DomainAnswer,
    RetrievedChunk,
    build_sub_answers_block,
    call_llm,
    format_chunks_for_prompt,
)


# ---------------------------------------------------------------------------
# Domain agent — one per sub-question
# ---------------------------------------------------------------------------

def _format_chat_history_block(chat_history: list[dict]) -> str:
    """Format recent conversation turns into a prompt block, or return empty string."""
    if not chat_history:
        return ""
    # Keep last 6 messages (3 pairs) to stay within token budget
    recent = chat_history[-6:]
    lines = ["\n--- CONVERSATION HISTORY (for context) ---"]
    for msg in recent:
        role = "Student" if msg.get("role") == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append("--- END HISTORY ---\n")
    return "\n".join(lines)


def generate_domain_answer(
    sub_question: str,
    domain: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str = DEFAULT_MODEL,
    chat_history: list[dict] | None = None,
) -> DomainAnswer:
    """
    Generate an answer for a single sub-question, blending retrieved course
    evidence with the model's own knowledge.

    Args:
        sub_question: The student's sub-question for this domain.
        domain: Domain identifier (e.g., "AML").
        retrieved_chunks: Reranked chunks from the domain retriever.
        model: LLM model to use.
        chat_history: Previous conversation turns for context.

    Returns:
        DomainAnswer with answer text, citations, and metadata.
    """
    domain_cfg = DOMAINS.get(domain, {})
    domain_name = domain_cfg.get("name", domain)
    domain_abbr = domain_cfg.get("abbr", domain)

    history_block = _format_chat_history_block(chat_history or [])

    system_prompt = DOMAIN_AGENT_SYSTEM.format(
        domain_name=domain_name,
        domain_abbr=domain_abbr,
    )

    if not retrieved_chunks:
        # No course material found — answer from model's own knowledge
        user_prompt = DOMAIN_AGENT_USER_NO_CONTEXT.format(
            question=sub_question,
            chat_history_block=history_block,
        )
        try:
            answer_text = call_llm(
                messages=[{"role": "user", "content": user_prompt}],
                system=system_prompt,
                model=model,
                max_tokens=LLM_MAX_TOKENS_GENERATE,
            )
        except Exception as exc:
            answer_text = _friendly_error(exc)

        return DomainAnswer(
            domain=domain,
            sub_question=sub_question,
            answer=answer_text,
            citations=[],
            retrieved_chunks=[],
            num_chunks_used=0,
            no_evidence=True,
        )

    chunks_text = format_chunks_for_prompt(retrieved_chunks)

    user_prompt = DOMAIN_AGENT_USER.format(
        question=sub_question,
        chat_history_block=history_block,
        retrieved_chunks=chunks_text,
    )

    try:
        answer_text = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            system=system_prompt,
            model=model,
            max_tokens=LLM_MAX_TOKENS_GENERATE,
        )
    except Exception as exc:
        answer_text = _friendly_error(exc)

    # Extract citation labels from chunks
    citations = [c.citation_label() for c in retrieved_chunks]

    return DomainAnswer(
        domain=domain,
        sub_question=sub_question,
        answer=answer_text,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        num_chunks_used=len(retrieved_chunks),
        no_evidence=False,
    )


# ---------------------------------------------------------------------------
# Cross-domain synthesizer
# ---------------------------------------------------------------------------

def synthesize_answers(
    original_query: str,
    domain_answers: list[DomainAnswer],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Merge multiple domain answers into one coherent educational response.

    If only one domain answer exists, returns it directly (no synthesis needed).

    Args:
        original_query: The student's original question.
        domain_answers: Per-domain answers from generate_domain_answer().
        model: Claude model to use.

    Returns:
        Synthesized answer string.
    """
    if not domain_answers:
        return NO_EVIDENCE_RESPONSE

    # Single domain — no synthesis needed
    if len(domain_answers) == 1:
        da = domain_answers[0]
        if da.no_evidence:
            return NO_EVIDENCE_RESPONSE
        return da.answer  # prompt already includes ## References section

    # Filter out completely empty answers
    valid_answers = [da for da in domain_answers if not da.no_evidence]
    if not valid_answers:
        return NO_EVIDENCE_RESPONSE

    # Build the synthesis prompt
    sub_answers_block = build_sub_answers_block(valid_answers)
    unique_domains = list(dict.fromkeys(da.domain for da in valid_answers))

    user_prompt = SYNTHESIZER_USER.format(
        original_query=original_query,
        num_parts=len(valid_answers),
        num_domains=len(unique_domains),
        sub_answers=sub_answers_block,
    )

    try:
        synthesized = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            system=SYNTHESIZER_SYSTEM,
            model=model,
            max_tokens=LLM_MAX_TOKENS_SYNTH,
        )
    except Exception as exc:
        # Rate limit or API error — show friendly message + raw domain answers as fallback
        err_msg = _friendly_error(exc)
        parts = [err_msg, "\n\n---\n\n*Partial answers retrieved before error:*"]
        for da in valid_answers:
            domain_name = DOMAINS.get(da.domain, {}).get("name", da.domain)
            parts.append(f"## {domain_name}\n\n{da.answer}")
        synthesized = "\n\n".join(parts)

    return synthesized


def generate_ss_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    model: str = DEFAULT_MODEL,
    chat_history: list[dict] | None = None,
) -> DomainAnswer:
    """
    Generate a strictly grounded answer for Self Study mode.
    Never supplements with general knowledge — returns a refusal if context is insufficient.
    """
    history_block = _format_chat_history_block(chat_history or [])

    if not retrieved_chunks:
        return DomainAnswer(
            domain="Self Study",
            sub_question=question,
            answer="I cannot find information about this in the selected document(s). Please upload relevant documents or remove the filter to search across all files.",
            citations=[],
            retrieved_chunks=[],
            num_chunks_used=0,
            no_evidence=True,
        )

    chunks_text = format_chunks_for_prompt(retrieved_chunks)

    user_prompt = SS_AGENT_USER.format(
        question=question,
        chat_history_block=history_block,
        retrieved_chunks=chunks_text,
    )

    try:
        answer_text = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            system=SS_AGENT_SYSTEM,
            model=model,
            max_tokens=LLM_MAX_TOKENS_SS,
        )
    except Exception as exc:
        answer_text = f"Error generating answer: {exc}"

    citations = [c.citation_label() for c in retrieved_chunks]
    da = DomainAnswer(
        domain="Self Study",
        sub_question=question,
        answer=answer_text,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        num_chunks_used=len(retrieved_chunks),
        no_evidence=False,
    )
    da.answer = _add_reference_list(da.answer, da)
    return da


def _add_reference_list(answer_text: str, da: DomainAnswer) -> str:
    """Append a formatted reference list to a single-domain answer."""
    if not da.citations:
        return answer_text

    refs = "\n\n---\n\n**References:**\n"
    for i, cit in enumerate(da.citations, 1):
        refs += f"- [Source {i}] {cit}\n"

    return answer_text + refs
