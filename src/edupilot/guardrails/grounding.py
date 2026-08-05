"""
Groundedness checking
=====================
Measures whether an answer's claims are actually supported by the evidence,
without asking an LLM to grade itself.

The previous system's `grounding_score` came from the same model family that
wrote the answer, in the same JSON blob as its own quality self-assessment,
and nothing downstream acted on it. A model that fabricated a claim is not
well placed to notice — and the prompt hard-coded `revised_answer: null`, so
even a failing score changed nothing.

This module scores support mechanically with the cross-encoder that is
already loaded for reranking. For each claim sentence it asks: does any
supplied excerpt entail this? A claim no excerpt supports is flagged, and —
critically — a claim whose *cited* excerpt does not support it is flagged as
a miscitation even when some other excerpt happens to.

Reusing the reranker costs no extra model download and no extra memory. When
a dedicated NLI model is configured it is used instead, which is more precise
about entailment versus mere topical similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from edupilot.guardrails.citations import CITATION, extract_claim_sentences

logger = logging.getLogger(__name__)

#: Cross-encoder support score above which a claim counts as entailed.
#:
#: Higher than the retrieval floor (0.05) on purpose. Retrieval asks "is this
#: passage worth reading for this question", a deliberately permissive bar.
#: Grounding asks "does this passage actually state this claim", which must be
#: strict — that is the whole point of the check.
SUPPORT_THRESHOLD = 0.35

#: Claims scoring below this are treated as clearly unsupported rather than
#: borderline, and are reported separately.
CLEARLY_UNSUPPORTED = 0.05


@dataclass
class ClaimVerdict:
    """Support assessment for one claim sentence."""

    claim: str
    best_score: float
    best_source: int | None
    cited_sources: list[int] = field(default_factory=list)
    cited_score: float | None = None
    """Support from the excerpt the answer actually cited. None if uncited."""

    @property
    def is_supported(self) -> bool:
        return self.best_score >= SUPPORT_THRESHOLD

    @property
    def is_miscited(self) -> bool:
        """
        Cited a source that does not support the claim.

        A distinct and more serious failure than an uncited claim: the answer
        asserts provenance it does not have, which is exactly what makes a
        fabricated statement look verified.
        """
        return bool(self.cited_sources) and (
            self.cited_score is None or self.cited_score < SUPPORT_THRESHOLD
        )


@dataclass
class GroundingReport:
    """Groundedness assessment for a whole answer."""

    verdicts: list[ClaimVerdict] = field(default_factory=list)
    checked: bool = True
    """False when no scorer was available and the result is not meaningful."""

    @property
    def grounding_score(self) -> float:
        if not self.verdicts:
            return 1.0
        return sum(1 for v in self.verdicts if v.is_supported) / len(self.verdicts)

    @property
    def unsupported(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if not v.is_supported]

    @property
    def miscited(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.is_miscited]

    @property
    def fabrication_risk(self) -> str:
        """Coarse risk band, used to decide whether to warn or block."""
        if not self.checked:
            return "unknown"
        score = self.grounding_score
        if score >= 0.90:
            return "low"
        if score >= 0.70:
            return "medium"
        return "high"

    def as_dict(self) -> dict:
        return {
            "checked": self.checked,
            "grounding_score": round(self.grounding_score, 3),
            "fabrication_risk": self.fabrication_risk,
            "claims_checked": len(self.verdicts),
            "unsupported_count": len(self.unsupported),
            "miscited_count": len(self.miscited),
            "unsupported_claims": [
                {"claim": v.claim[:220], "best_score": round(v.best_score, 4)}
                for v in self.unsupported[:5]
            ],
            "miscited_claims": [
                {
                    "claim": v.claim[:220],
                    "cited": v.cited_sources,
                    "cited_score": round(v.cited_score, 4) if v.cited_score is not None else None,
                }
                for v in self.miscited[:5]
            ],
        }


def _score_pairs(pairs: list[tuple[str, str]]) -> list[float] | None:
    """
    Score (claim, evidence) pairs for entailment.

    Uses the reranking cross-encoder, which is already resident. Returns None
    when no scorer can be loaded, so the caller reports "unchecked" rather
    than silently claiming everything is grounded.
    """
    if not pairs:
        return []
    try:
        from edupilot.retrieval.rerank import _emits_probabilities, _sigmoid, get_cross_encoder

        model, _ = get_cross_encoder()
        if model is None:
            return None
        raw = model.predict(pairs, batch_size=32, show_progress_bar=False)
        normalize = (lambda s: s) if _emits_probabilities(model) else _sigmoid
        return [float(normalize(float(s))) for s in raw]
    except Exception as exc:
        logger.warning("grounding scorer unavailable: %s", exc)
        return None


def check_grounding(
    answer: str,
    evidence_texts: list[str],
    *,
    max_claims: int = 40,
) -> GroundingReport:
    """
    Check whether an answer's claims are supported by its evidence.

    Args:
        answer: The generated markdown answer.
        evidence_texts: Excerpt texts, ordered so index i is [Source i+1].
        max_claims: Cap on claims scored, to bound latency on long answers.

    Returns:
        A `GroundingReport`. `checked=False` means no scorer was available and
        `grounding_score` must not be reported as a real measurement.
    """
    report = GroundingReport()
    if not answer or not evidence_texts:
        return report

    claims = extract_claim_sentences(answer)[:max_claims]
    if not claims:
        return report

    # One pair per (claim, excerpt). Bounded by max_claims x len(evidence).
    pairs: list[tuple[str, str]] = []
    for claim in claims:
        for text in evidence_texts:
            pairs.append((claim, text))

    scores = _score_pairs(pairs)
    if scores is None:
        report.checked = False
        return report

    n_sources = len(evidence_texts)
    for i, claim in enumerate(claims):
        window = scores[i * n_sources : (i + 1) * n_sources]
        if not window:
            continue
        best_score = max(window)
        best_index = window.index(best_score)

        cited = sorted({int(m.group(1)) for m in CITATION.finditer(claim)})
        valid_cited = [c for c in cited if 1 <= c <= n_sources]
        cited_score = max((window[c - 1] for c in valid_cited), default=None)

        report.verdicts.append(
            ClaimVerdict(
                claim=claim,
                best_score=best_score,
                best_source=best_index + 1,
                cited_sources=cited,
                cited_score=cited_score,
            )
        )

    if report.fabrication_risk == "high":
        logger.warning(
            "high fabrication risk: %d/%d claims unsupported",
            len(report.unsupported), len(report.verdicts),
        )
    return report


def annotate_unsupported(answer: str, report: GroundingReport) -> str:
    """
    Append a warning naming the claims the evidence does not support.

    Flagging rather than deleting is deliberate. Silently removing sentences
    produces an incoherent answer and hides the failure from the student,
    while a visible warning lets them judge — and makes the system's own
    uncertainty legible, which is the point of a source-grounded tutor.
    """
    if report.checked and not report.unsupported and not report.miscited:
        return answer

    lines: list[str] = []
    if report.unsupported:
        lines.append(
            f"\n\n> ⚠️ **Grounding warning** — {len(report.unsupported)} statement(s) "
            f"above could not be matched to the retrieved course material. "
            f"Verify these against your notes before relying on them:"
        )
        for verdict in report.unsupported[:3]:
            snippet = verdict.claim[:160] + ("…" if len(verdict.claim) > 160 else "")
            lines.append(f">\n> - *{snippet}*")

    if report.miscited:
        lines.append(
            f"\n>\n> ⚠️ {len(report.miscited)} statement(s) cite a source that does "
            f"not appear to support them."
        )

    return answer + "\n".join(lines) if lines else answer
