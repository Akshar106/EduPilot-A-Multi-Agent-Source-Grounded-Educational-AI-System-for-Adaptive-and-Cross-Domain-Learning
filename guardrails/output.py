"""
Output policy
=============
The single gate every answer passes through before it reaches a student.

This is what the old verifier was supposed to be and structurally could not
be. That verifier asked the model to grade itself, hard-coded
`"revised_answer": null` in its own prompt, and then nulled the field again in
code whenever `is_satisfactory` was true — so `get_final_answer()` returned
the original string on every path. The score was displayed; nothing acted on
it. Self Study then applied `quality = max(quality, 0.75)`, so the number
shown to the student was a floor, not a measurement.

Here the checks are mechanical and the consequences are real:

  fabricated citations  markers pointing at sources that were never supplied
                        are removed, because they assert provenance that does
                        not exist.
  high fabrication risk an answer whose claims mostly cannot be matched to
                        the evidence is replaced by a refusal. A wrong answer
                        a student trusts is worse than no answer.
  medium risk           the answer stands, with the specific unsupported
                        sentences named inline.

Escalation is deliberately asymmetric: annotate when uncertain, refuse only
when the evidence clearly does not support the answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Sequence

from .citations import (
    CitationReport,
    build_reference_list,
    repair_out_of_range,
    validate_citations,
)
from .grounding import GroundingReport, annotate_unsupported, check_grounding
from .refusal import REFUSAL_MARKER, is_refusal, make_refusal, strip_refusal_marker

logger = logging.getLogger(__name__)

Action = Literal["pass", "refusal", "annotated", "downgraded", "blocked"]

#: Below this grounding score the answer is replaced by a refusal.
REFUSE_BELOW_GROUNDING = 0.50

#: Below this it is shown with the unsupported sentences flagged.
ANNOTATE_BELOW_GROUNDING = 0.90

#: Answers shorter than this are not worth scoring for grounding.
MIN_ANSWER_CHARS = 40


@dataclass
class OutputVerdict:
    """Final decision about an answer, plus the evidence behind it."""

    answer: str
    action: Action = "pass"
    citations: CitationReport | None = None
    grounding: GroundingReport | None = None
    warnings: list[str] = field(default_factory=list)
    repaired_citations: int = 0

    @property
    def grounding_score(self) -> float | None:
        """
        Measured grounding, or None when it could not be measured.

        Deliberately nullable. Reporting a default when nothing was checked is
        how the previous pipeline ended up displaying quality numbers it had
        not computed.
        """
        if self.grounding is None or not self.grounding.checked:
            return None
        return self.grounding.grounding_score

    @property
    def is_refusal(self) -> bool:
        return self.action in ("refusal", "downgraded")

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "grounding_score": self.grounding_score,
            "repaired_citations": self.repaired_citations,
            "warnings": self.warnings,
            "citations": self.citations.as_dict() if self.citations else None,
            "grounding": self.grounding.as_dict() if self.grounding else None,
        }


DOWNGRADE_MESSAGE = """\
I could not produce an answer that stays within the course materials on this \
topic.

The material I retrieved does not actually support a substantive answer to \
your question, and I will not fill that gap from general knowledge — an \
answer you cannot trace back to your course documents is not something you \
should rely on for coursework.

**What helps:** rephrase using the course's own terminology, ask about a \
narrower part of the topic, or upload the relevant lecture notes with the 📎 \
button.\
"""


def apply_output_guardrails(
    answer: str,
    evidence_texts: Sequence[str],
    source_labels: Sequence[str] | None = None,
    *,
    check_claims: bool = True,
    append_references: bool = True,
) -> OutputVerdict:
    """
    Validate and, where necessary, repair or replace a generated answer.

    Args:
        answer: Raw model output.
        evidence_texts: Excerpt texts, ordered so index i is [Source i+1].
        source_labels: Citation labels for the reference section.
        check_claims: Run cross-encoder grounding. Disable to save latency on
            paths where the answer is already known to be grounded.
        append_references: Append a reference list of *cited* sources only.

    Returns:
        An `OutputVerdict` carrying the final answer text and the reasoning.
    """
    text = (answer or "").strip()

    # A refusal is a correct outcome. Pass it through untouched.
    if is_refusal(text):
        return OutputVerdict(answer=strip_refusal_marker(text), action="refusal")

    if len(text) < MIN_ANSWER_CHARS:
        return OutputVerdict(
            answer=text,
            action="pass",
            warnings=["answer too short to validate"],
        )

    n_sources = len(evidence_texts)
    verdict = OutputVerdict(answer=text)

    # --- citations ------------------------------------------------------
    repaired, removed = repair_out_of_range(text, n_sources)
    if removed:
        verdict.repaired_citations = removed
        verdict.warnings.append(
            f"removed {removed} citation marker(s) referring to sources that were "
            f"never supplied (only {n_sources} provided)"
        )
        text = repaired

    verdict.citations = validate_citations(text, n_sources)

    # --- grounding ------------------------------------------------------
    if check_claims and n_sources:
        verdict.grounding = check_grounding(text, list(evidence_texts))
    else:
        verdict.grounding = GroundingReport(checked=False)

    score = verdict.grounding_score

    # --- policy ---------------------------------------------------------
    if score is not None and score < REFUSE_BELOW_GROUNDING:
        logger.warning(
            "downgrading answer to refusal: grounding %.2f below %.2f (%d/%d claims unsupported)",
            score, REFUSE_BELOW_GROUNDING,
            len(verdict.grounding.unsupported), len(verdict.grounding.verdicts),
        )
        verdict.answer = DOWNGRADE_MESSAGE
        verdict.action = "downgraded"
        verdict.warnings.append(
            f"answer replaced: only {score:.0%} of its claims were supported by the evidence"
        )
        return verdict

    if score is not None and score < ANNOTATE_BELOW_GROUNDING:
        text = annotate_unsupported(text, verdict.grounding)
        verdict.action = "annotated"
        verdict.warnings.append(
            f"{len(verdict.grounding.unsupported)} unsupported statement(s) flagged inline"
        )
    elif verdict.citations and verdict.citations.has_fabricated_citations:
        verdict.action = "annotated"

    if append_references and source_labels:
        text += build_reference_list(text, list(source_labels))

    verdict.answer = text
    return verdict

