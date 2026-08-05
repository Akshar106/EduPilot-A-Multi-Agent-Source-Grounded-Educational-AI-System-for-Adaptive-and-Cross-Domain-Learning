"""
Citation validation
===================
Deterministic checks on the [Source N] markers in a generated answer.

Nothing in the previous pipeline verified citations at all. The answer could
cite [Source 7] when five sources were supplied, cite a source that says
nothing about the claim, or state facts with no citation whatsoever — and the
UI would render a tidy reference list regardless, because
`_add_reference_list()` simply listed every retrieved chunk as a reference
whether or not the model had cited it.

Three failures are checked here, all without a model call:

  out-of-range     [Source 9] when 5 sources were provided. Proof of
                   fabrication: the model invented a citation index.
  uncited claims   factual sentences carrying no [Source N] marker.
  phantom refs     sources listed in the reference section that were never
                   cited in the body.

Whether a citation *supports* its claim is a semantic question and lives in
`guardrails/grounding.py`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

CITATION = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

#: Lines that make no factual claim and so need no citation.
_SKIP_LINE = re.compile(
    r"""^\s*(?:
        \#{1,6}\s          # markdown heading
        |[-*+]\s*$         # empty list bullet
        |\|                # table row
        |>\s               # blockquote
        |```               # code fence
        |---+\s*$          # horizontal rule
    )""",
    re.VERBOSE,
)

#: Sentences that are structural or meta rather than claims about the material.
_META_SENTENCE = re.compile(
    r"^\s*(?:"
    r"the (?:course )?(?:materials?|excerpts?|sources?|documents?) "
    r"(?:do not|don't|does not|doesn't|only|contain|cover|provide|include)"
    r"|based on the (?:excerpts?|sources?|materials?)"
    r"|(?:this|these) (?:section|answer|response)"
    r"|in summary"
    r"|to summari[sz]e"
    r"|here(?:'s| is) (?:what|how)"
    r"|note that the (?:excerpt|source)"
    r")",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

#: Below this word count a sentence is a fragment, label, or transition.
_MIN_CLAIM_WORDS = 8


@dataclass
class CitationReport:
    """Outcome of validating one answer's citations."""

    total_citations: int = 0
    distinct_sources_cited: set[int] = field(default_factory=set)
    out_of_range: list[int] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    unused_sources: list[int] = field(default_factory=list)
    claim_sentences: int = 0

    @property
    def has_fabricated_citations(self) -> bool:
        """True when the answer cites a source index that was never supplied."""
        return bool(self.out_of_range)

    @property
    def citation_coverage(self) -> float:
        """Fraction of factual sentences carrying at least one citation."""
        if self.claim_sentences == 0:
            return 1.0
        return 1.0 - len(self.uncited_claims) / self.claim_sentences

    @property
    def is_valid(self) -> bool:
        """No fabricated indices and most claims cited."""
        return not self.has_fabricated_citations and self.citation_coverage >= 0.70

    def as_dict(self) -> dict:
        return {
            "total_citations": self.total_citations,
            "distinct_sources_cited": sorted(self.distinct_sources_cited),
            "out_of_range": self.out_of_range,
            "uncited_claim_count": len(self.uncited_claims),
            "uncited_claims": self.uncited_claims[:5],
            "unused_sources": self.unused_sources,
            "claim_sentences": self.claim_sentences,
            "citation_coverage": round(self.citation_coverage, 3),
            "has_fabricated_citations": self.has_fabricated_citations,
            "is_valid": self.is_valid,
        }


def _body_without_references(answer: str) -> str:
    """
    Drop a trailing reference/citation section.

    Reference lists cite every source by design; counting them would mask
    uncited claims in the body.
    """
    pattern = re.compile(
        r"\n\s*(?:#{1,6}\s*)?(?:\*\*)?(?:references|sources|citations)(?:\*\*)?\s*:?\s*\n",
        re.IGNORECASE,
    )
    match = pattern.search(answer)
    return answer[: match.start()] if match else answer


def extract_claim_sentences(answer: str) -> list[str]:
    """
    Sentences that assert something about the course material.

    Headings, list scaffolding, table rows, code, and meta-commentary are
    excluded — requiring "## Overview" to carry a citation would make the
    coverage metric meaningless.
    """
    claims: list[str] = []
    in_code = False

    for line in _body_without_references(answer).split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line.strip() or _SKIP_LINE.match(line):
            continue

        # Strip list markers so the sentence itself is evaluated.
        content = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).strip()
        if not content:
            continue

        for sentence in _SENTENCE_SPLIT.split(content):
            sentence = sentence.strip()
            if len(sentence.split()) < _MIN_CLAIM_WORDS:
                continue
            if _META_SENTENCE.match(sentence):
                continue
            claims.append(sentence)

    return claims


def validate_citations(answer: str, num_sources: int) -> CitationReport:
    """
    Validate the [Source N] markers in `answer` against `num_sources` supplied.

    Args:
        answer: The generated markdown answer.
        num_sources: How many numbered excerpts were given to the model.

    Returns:
        A `CitationReport`. `has_fabricated_citations` is the hard failure —
        it proves the model invented a source index.
    """
    report = CitationReport()
    if not answer:
        return report

    body = _body_without_references(answer)

    for match in CITATION.finditer(answer):
        index = int(match.group(1))
        report.total_citations += 1
        report.distinct_sources_cited.add(index)
        if index < 1 or index > num_sources:
            report.out_of_range.append(index)

    claims = extract_claim_sentences(answer)
    report.claim_sentences = len(claims)
    report.uncited_claims = [s for s in claims if not CITATION.search(s)]

    cited_in_body = {int(m.group(1)) for m in CITATION.finditer(body)}
    report.unused_sources = [
        n for n in range(1, num_sources + 1) if n not in cited_in_body
    ]

    if report.out_of_range:
        logger.warning(
            "answer cites out-of-range sources %s (only %d supplied)",
            sorted(set(report.out_of_range)), num_sources,
        )
    return report


def repair_out_of_range(answer: str, num_sources: int) -> tuple[str, int]:
    """
    Remove citation markers pointing at sources that do not exist.

    Deleting the marker is the conservative repair: the sentence remains but
    stops making a false provenance claim, and it then shows up as an uncited
    claim in the report rather than looking verified. Rewriting the index to
    a real source would be worse — it would attach the claim to an excerpt
    chosen arbitrarily.

    Returns (repaired_answer, markers_removed).
    """
    removed = 0

    def replace(match: re.Match) -> str:
        nonlocal removed
        index = int(match.group(1))
        if 1 <= index <= num_sources:
            return match.group(0)
        removed += 1
        return ""

    repaired = CITATION.sub(replace, answer)
    if removed:
        # Tidy the punctuation left behind, e.g. "text  ." or "( , )".
        repaired = re.sub(r"\s+([.,;:])", r"\1", repaired)
        repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    return repaired, removed


def build_reference_list(answer: str, labels: list[str]) -> str:
    """
    Build a reference section listing only sources actually cited in the body.

    The previous implementation appended every retrieved chunk regardless of
    use, so an answer citing one source displayed five references and looked
    considerably better evidenced than it was.
    """
    cited = sorted(
        n for n in {int(m.group(1)) for m in CITATION.finditer(_body_without_references(answer))}
        if 1 <= n <= len(labels)
    )
    if not cited:
        return ""
    lines = "\n".join(f"- **[Source {n}]** {labels[n - 1]}" for n in cited)
    return f"\n\n---\n\n**Sources cited**\n{lines}\n"
