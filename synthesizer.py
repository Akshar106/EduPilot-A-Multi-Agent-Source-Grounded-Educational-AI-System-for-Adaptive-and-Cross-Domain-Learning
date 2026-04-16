"""
EduPilot Synthesizer
====================
Two responsibilities:
  1. Domain Agent  — generate a grounded answer for one domain's sub-question
  2. Cross-Domain Synthesizer — merge multiple domain answers into one response

Both use Claude and stay strictly within the retrieved evidence.
"""

from __future__ import annotations

from config import DEFAULT_MODEL, DOMAINS, LLM_MAX_TOKENS_GENERATE, LLM_MAX_TOKENS_SYNTH
from prompts import (
    DOMAIN_AGENT_SYSTEM,
    DOMAIN_AGENT_USER,
    DOMAIN_AGENT_USER_NO_CONTEXT,
    NO_EVIDENCE_RESPONSE,
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
            answer_text = NO_EVIDENCE_RESPONSE + f"\n\n_(Error: {exc})_"

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
        answer_text = (
            f"I encountered an error generating the answer: {exc}\n\n"
            f"Retrieved context was available ({len(retrieved_chunks)} chunks)."
        )

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
        return _add_reference_list(da.answer, da)

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
        # Fallback: concatenate answers with headers
        parts = []
        for da in valid_answers:
            domain_name = DOMAINS.get(da.domain, {}).get("name", da.domain)
            parts.append(f"## {domain_name}\n\n{da.answer}")
        synthesized = "\n\n---\n\n".join(parts)

    return synthesized


def _add_reference_list(answer_text: str, da: DomainAnswer) -> str:
    """Append a formatted reference list to a single-domain answer."""
    if not da.citations:
        return answer_text

    refs = "\n\n---\n\n**References:**\n"
    for i, cit in enumerate(da.citations, 1):
        refs += f"- [Source {i}] {cit}\n"

    return answer_text + refs
