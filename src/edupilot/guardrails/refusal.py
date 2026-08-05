"""
Refusal contract
================
The marker an agent emits when the evidence does not support an answer, and
the helpers for detecting it.

This lives in `guardrails` rather than `agents` for dependency direction:
`agents` produces answers and `guardrails` inspects them, so guardrails must
not import from agents. Both need the refusal contract, so it belongs in the
lower layer. `agents.prompts` imports the marker from here.

Detection is by an exact machine-readable marker rather than by matching
phrases like "I cannot find". That heuristic — which the previous code used —
fires on any answer that merely *discusses* not finding something, and misses
a refusal phrased even slightly differently.
"""

from __future__ import annotations

import re

#: Prefix an agent emits when it declines for lack of evidence.
REFUSAL_MARKER = "INSUFFICIENT_EVIDENCE"

_REFUSAL = re.compile(rf"^\s*{re.escape(REFUSAL_MARKER)}\s*:?\s*", re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    """True when the agent declined for lack of evidence."""
    return bool(_REFUSAL.match(answer or ""))


def strip_refusal_marker(answer: str) -> str:
    """Remove the machine-readable marker, leaving the student-facing sentence."""
    return _REFUSAL.sub("", answer or "").strip()


def make_refusal(topic: str) -> str:
    """Build the canonical refusal string for a topic the evidence does not cover."""
    return f"{REFUSAL_MARKER}: The course materials I have access to do not cover {topic}."
