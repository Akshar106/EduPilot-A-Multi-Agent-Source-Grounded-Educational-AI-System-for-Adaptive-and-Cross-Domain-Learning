"""
EduPilot guardrails
===================
Checks applied to model input and output, none of which trust the model to
police itself.

    from edupilot.guardrails import validate_citations, check_grounding, scan

    citations = validate_citations(answer, num_sources=len(evidence.labels))
    grounding = check_grounding(answer, evidence_texts)
    if citations.has_fabricated_citations or grounding.fabrication_risk == "high":
        ...

Layers:

  citations   deterministic [Source N] validation — out-of-range markers,
              uncited claims, phantom references. No model call.
  grounding   cross-encoder support scoring per claim, including detection of
              claims that cite a source which does not support them.
  injection   pattern scanning of untrusted document and user text.
  output      response-level policy checks before anything reaches the student.
"""

from .citations import (
    CitationReport,
    build_reference_list,
    extract_claim_sentences,
    repair_out_of_range,
    validate_citations,
)
from .grounding import (
    ClaimVerdict,
    GroundingReport,
    annotate_unsupported,
    check_grounding,
)
from .injection import (
    InjectionReport,
    Severity,
    neutralize,
    scan,
    scan_chunks,
)
from .output import OutputVerdict, apply_output_guardrails
from .refusal import REFUSAL_MARKER, is_refusal, make_refusal, strip_refusal_marker

__all__ = [
    "CitationReport",
    "REFUSAL_MARKER",
    "ClaimVerdict",
    "GroundingReport",
    "InjectionReport",
    "OutputVerdict",
    "Severity",
    "annotate_unsupported",
    "apply_output_guardrails",
    "build_reference_list",
    "check_grounding",
    "is_refusal",
    "make_refusal",
    "extract_claim_sentences",
    "neutralize",
    "repair_out_of_range",
    "scan",
    "scan_chunks",
    "strip_refusal_marker",
    "validate_citations",
]
