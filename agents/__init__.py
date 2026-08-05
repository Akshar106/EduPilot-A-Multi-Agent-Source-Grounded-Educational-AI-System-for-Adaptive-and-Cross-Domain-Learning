"""
EduPilot agents
===============
Agent instructions and the pipeline that runs them.

    Router -> Planner -> [Retriever -> Answerer] x N -> Synthesizer -> Guardrails

    from agents import EduPilotPipeline, PipelineConfig

    pipeline = EduPilotPipeline(call_llm, DOMAINS, get_retriever)
    result = pipeline.run("what is a p-value", PipelineConfig(model="..."))

`result.grounding_score` is None when grounding was not measured — it is never
defaulted, so a displayed score always reflects a real check.
"""

from .contracts import (
    EvidenceBlock,
    SourceFence,
    build_evidence,
    build_evidence_summary,
    citation_label,
    format_history,
    is_refusal,
    make_fence,
    parse_json_response,
    strip_refusal_marker,
)
from .pipeline import (
    Answerer,
    EduPilotPipeline,
    PipelineConfig,
    PipelineResult,
    Planner,
    Router,
    RouterDecision,
    SubAnswer,
    Synthesizer,
    Verifier,
)
from .prompts import REFUSAL_MARKER

__all__ = [
    "Answerer",
    "EduPilotPipeline",
    "EvidenceBlock",
    "PipelineConfig",
    "PipelineResult",
    "Planner",
    "REFUSAL_MARKER",
    "Router",
    "RouterDecision",
    "SourceFence",
    "SubAnswer",
    "Synthesizer",
    "Verifier",
    "build_evidence",
    "build_evidence_summary",
    "citation_label",
    "format_history",
    "is_refusal",
    "make_fence",
    "parse_json_response",
    "strip_refusal_marker",
]
