"""
Rolling conversation memory
===========================
A conversation that runs long cannot be sent to the model verbatim: prompt cost
grows with every turn, and the oldest turns get silently truncated by whatever
window is in force — so the model forgets the beginning of a conversation
precisely when the student starts referring back to it.

This keeps the last `KEEP_RECENT_TURNS` turns verbatim and folds everything
older into a running digest stored on the session. The digest is rebuilt
incrementally: each pass reads only the turns added since `summary_through_id`,
so summarization cost stays constant no matter how long the conversation gets.

    memory = ConversationMemory(call_llm)
    digest, recent = memory.context_for(session_id, model="...")

The digest is *conversation state*, not retrieved evidence. It never licenses a
factual claim — every claim in an answer must still cite a source excerpt. Its
job is to resolve references ("explain that in more detail", "how does it
compare to the one you mentioned?") that would otherwise be unanswerable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from edupilot import db

logger = logging.getLogger(__name__)

#: Turns kept verbatim. Below this, no summarization happens at all — short
#: conversations pay nothing.
KEEP_RECENT_TURNS = 6

#: Summarize only once this many un-folded messages have accumulated beyond the
#: verbatim window. Batching keeps it to roughly one extra LLM call per eight
#: turns rather than one per turn.
SUMMARIZE_BATCH = 6

#: Hard cap on digest length. A digest that grows every pass would recreate the
#: unbounded-prompt problem it exists to solve.
MAX_DIGEST_CHARS = 1200

_SUMMARY_SYSTEM = """\
You maintain a running summary of a tutoring conversation so the tutor can \
follow back-references in later questions.

Write a compact digest capturing:
- what topics the student has asked about, in order
- specific concepts, algorithms, or terms that were explained
- anything the student stated about themselves (their course, what they are \
studying for, what they said they already understand)
- questions that were refused or left unresolved

Rules:
- Under 150 words. Prose or terse bullets, no preamble.
- Record what was discussed, not the explanations themselves. This is an index \
to the conversation, not a replacement for it.
- Never invent. If the transcript does not say it, it does not go in.
- If an existing digest is supplied, merge the new turns into it and return the \
merged result — do not append a second summary.\
"""

_SUMMARY_USER = """\
{existing_block}<new_turns>
{turns}
</new_turns>

Return only the merged digest.\
"""


class ConversationMemory:
    """
    Builds and maintains the per-session digest.

    Args:
        llm: The shared LLM caller, matching `(messages, system, model,
            max_tokens) -> str`.
    """

    def __init__(self, llm: Callable) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def context_for(
        self, session_id: str, *, model: str, exclude_message_ids: set[int] | None = None
    ) -> tuple[str, list[dict]]:
        """
        Return `(digest, recent_turns)` for a session, summarizing if needed.

        `recent_turns` is a list of {role, content} in chronological order,
        ready for `format_history`. `exclude_message_ids` drops messages the
        caller has already written for the in-flight request — the current
        question should not appear in its own history block.
        """
        if not session_id:
            return "", []

        digest, through_id = db.get_session_memory(session_id)
        pending = db.get_messages_after(session_id, through_id)

        if exclude_message_ids:
            pending = [m for m in pending if m["id"] not in exclude_message_ids]

        # Everything still fits verbatim — nothing to compact.
        if len(pending) <= KEEP_RECENT_TURNS + SUMMARIZE_BATCH:
            return digest, [{"role": m["role"], "content": m["content"]} for m in pending]

        # Fold the older half in, keep the tail verbatim.
        to_fold = pending[:-KEEP_RECENT_TURNS]
        recent = pending[-KEEP_RECENT_TURNS:]

        merged = self._summarize(digest, to_fold, model=model)
        if merged:
            db.set_session_memory(session_id, merged, to_fold[-1]["id"])
            digest = merged
        else:
            # Summarization failed. Degrade to verbatim recent turns rather
            # than dropping context silently — the digest simply does not
            # advance, and the next request retries.
            logger.warning("digest update failed for session %s; using recent turns only", session_id)

        return digest, [{"role": m["role"], "content": m["content"]} for m in recent]

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def _summarize(self, existing: str, turns: list[dict], *, model: str) -> str:
        """One LLM call folding `turns` into `existing`. Returns "" on failure."""
        transcript = "\n".join(
            f"{'Student' if t['role'] == 'user' else 'Assistant'}: "
            f"{str(t['content']).strip()[:800]}"
            for t in turns
            if str(t.get("content", "")).strip()
        )
        if not transcript:
            return existing

        existing_block = (
            f"<existing_digest>\n{existing}\n</existing_digest>\n\n" if existing else ""
        )

        try:
            out = self._llm(
                messages=[{
                    "role": "user",
                    "content": _SUMMARY_USER.format(
                        existing_block=existing_block, turns=transcript
                    ),
                }],
                system=_SUMMARY_SYSTEM,
                model=model,
                max_tokens=400,
            )
        except Exception:
            logger.warning("conversation summarization failed", exc_info=True)
            return ""

        return (out or "").strip()[:MAX_DIGEST_CHARS]


def format_memory(digest: str) -> str:
    """
    Render the digest as a prompt block.

    Labelled as conversation state and explicitly denied evidentiary weight —
    otherwise a model that sees "we discussed bagging" may treat that as
    licence to assert facts about bagging without a source excerpt.
    """
    if not digest.strip():
        return ""
    return (
        "<conversation_memory>\n"
        "Summary of earlier turns in this conversation. Use it only to "
        "understand what the student is referring back to. It is NOT evidence "
        "and can never support a factual claim — only source excerpts can.\n"
        f"{digest.strip()}\n"
        "</conversation_memory>\n"
    )


__all__ = [
    "KEEP_RECENT_TURNS",
    "MAX_DIGEST_CHARS",
    "SUMMARIZE_BATCH",
    "ConversationMemory",
    "format_memory",
]
