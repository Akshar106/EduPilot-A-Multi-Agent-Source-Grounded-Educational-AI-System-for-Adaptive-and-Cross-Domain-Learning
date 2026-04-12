"""
EduPilot Verifier
=================
A second LLM pass that acts as a quality-control critic.

The verifier receives:
  - The original student query
  - The sub-questions that were answered
  - A summary of the retrieved evidence
  - The synthesized answer

It checks coverage, grounding, completeness, and coherence.
If the answer is unsatisfactory, it produces a revised version.

The verifier is optional — it can be disabled via the sidebar toggle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import DEFAULT_MODEL, LLM_MAX_TOKENS_VERIFY
from prompts import VERIFIER_SYSTEM, VERIFIER_USER
from utils import DomainAnswer, call_llm, format_evidence_summary, parse_json_response


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    is_satisfactory: bool = True
    quality_score: float = 1.0
    coverage_score: float = 1.0
    grounding_score: float = 1.0
    issues: list[str] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)
    has_unsupported_claims: bool = False
    revised_answer: str | None = None
    raw_response: str = ""
    skipped: bool = False      # True when verification is disabled


def _default_pass_result() -> VerificationResult:
    """Return a passing result used when verification is skipped."""
    return VerificationResult(
        is_satisfactory=True,
        quality_score=1.0,
        coverage_score=1.0,
        grounding_score=1.0,
        skipped=True,
    )


# ---------------------------------------------------------------------------
# Main verifier function
# ---------------------------------------------------------------------------

def verify_answer(
    original_query: str,
    sub_questions: list[dict],
    domain_answers: list[DomainAnswer],
    synthesized_answer: str,
    model: str = DEFAULT_MODEL,
    enabled: bool = True,
) -> VerificationResult:
    """
    Verify the quality of the synthesized answer against the original query
    and retrieved evidence.

    Args:
        original_query: The student's original question.
        sub_questions: List of decomposed sub-questions with domain assignments.
        domain_answers: Domain-specific answers with retrieved chunks.
        synthesized_answer: The merged answer from the synthesizer.
        model: Claude model to use.
        enabled: If False, skip verification and return a passing result.

    Returns:
        VerificationResult — contains quality scores, issues, and revised answer
        if needed.
    """
    if not enabled:
        return _default_pass_result()

    # Build compact prompts
    sub_q_str = "\n".join(
        f"- [{sq['domain']}] {sq['question']}" for sq in sub_questions
    )
    evidence_str = format_evidence_summary(domain_answers)

    user_prompt = VERIFIER_USER.format(
        original_query=original_query,
        sub_questions=sub_q_str,
        evidence_summary=evidence_str,
        answer=synthesized_answer,
    )

    try:
        raw = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            system=VERIFIER_SYSTEM,
            model=model,
            max_tokens=LLM_MAX_TOKENS_VERIFY,
        )
        data = parse_json_response(raw)
    except Exception as exc:
        # If verifier fails, pass the original answer through unchanged
        return VerificationResult(
            is_satisfactory=True,
            quality_score=0.7,
            issues=[f"Verifier unavailable: {exc}"],
            raw_response=str(exc),
        )

    if not data:
        return VerificationResult(
            is_satisfactory=True,
            quality_score=0.7,
            issues=["Verifier returned unparseable response; passing original answer."],
            raw_response=raw if "raw" in locals() else "",
        )

    # Coerce fields with safe defaults
    is_satisfactory = bool(data.get("is_satisfactory", True))
    quality_score = _clamp(float(data.get("quality_score", 0.7)))
    coverage_score = _clamp(float(data.get("coverage_score", 0.7)))
    grounding_score = _clamp(float(data.get("grounding_score", 0.7)))
    issues = [str(i) for i in data.get("issues", [])]
    missing = [str(m) for m in data.get("missing_topics", [])]
    unsupported = bool(data.get("has_unsupported_claims", False))
    revised = data.get("revised_answer")

    # Only keep the revised answer if the verifier flagged the answer as unsatisfactory
    if is_satisfactory:
        revised = None

    return VerificationResult(
        is_satisfactory=is_satisfactory,
        quality_score=quality_score,
        coverage_score=coverage_score,
        grounding_score=grounding_score,
        issues=issues,
        missing_topics=missing,
        has_unsupported_claims=unsupported,
        revised_answer=revised if isinstance(revised, str) and revised.strip() else None,
        raw_response=raw if "raw" in locals() else "",
    )


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Helper: pick the best answer after verification
# ---------------------------------------------------------------------------

def get_final_answer(
    synthesized_answer: str,
    verification: VerificationResult,
) -> str:
    """
    Return the final answer to show the user.
    Prefer the verifier's revised answer if one was produced.
    """
    if (
        not verification.is_satisfactory
        and verification.revised_answer
        and len(verification.revised_answer.strip()) > 50
    ):
        return verification.revised_answer
    return synthesized_answer
