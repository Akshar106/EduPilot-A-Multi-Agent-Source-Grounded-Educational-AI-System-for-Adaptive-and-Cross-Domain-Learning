"""
Live pipeline adapter
=====================
Bridges the evaluation runner to the real agent pipeline.

The runner takes `pipeline_fn` as an argument precisely so it does not depend
on the composition root. This module is the one place that closes that gap, so
the CLI and the API score answers through an identical path — a metric that
differs between "run from the terminal" and "run from the UI" is worthless.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from edupilot.core.config import VERIFY_MODEL


def build_pipeline_fn(*, verify_model: str = VERIFY_MODEL) -> Callable:
    """
    Return a `pipeline_fn` matching the runner's contract.

    Evaluation deliberately runs with no conversation history and no answer
    cache: a case must be scored on what the pipeline produces for that
    question alone, not on what a previous case left behind.
    """
    from edupilot.agents import PipelineConfig
    from edupilot.core.services import services
    from edupilot.retrieval import RetrievalConfig

    def pipeline_fn(
        query: str,
        model: str,
        top_k: int,
        rerank_top_k: int,
        enable_verification: bool,
    ) -> dict[str, Any]:
        result = services.pipeline.run(
            query,
            PipelineConfig(
                model=model,
                verify_model=verify_model,
                retrieval=RetrievalConfig(
                    top_k=rerank_top_k,
                    candidate_multiplier=max(3, top_k // 2),
                ),
                enable_verification=enable_verification,
            ),
        )
        return {
            "final_answer": result.final_answer,
            "intent_type": result.intent_type,
            "detected_domains": result.domains,
            "is_course_related": result.is_course_related,
            "needs_clarification": result.needs_clarification,
            "refused": result.refused,
            # The runner reads `quality_score`; grounding is the only score the
            # pipeline actually measures, so it is what gets reported.
            "quality_score": result.grounding_score or 0.0,
            "grounding_score": result.grounding_score,
            "sources": result.sources,
            "debug": dict(result.diagnostics),
        }

    return pipeline_fn


def result_to_dict(result) -> dict[str, Any]:
    """Flatten a `TestResult` for JSON transport."""
    tc = result.test_case
    return {
        "id": tc.id,
        "name": tc.name,
        "category": tc.category,
        "query": tc.query,
        "expected_intent": tc.expected_intent,
        "expected_domains": tc.expected_domains,
        "expected_behavior": tc.expected_behavior,
        "passed": result.passed,
        "intent_match": result.intent_match,
        "domain_match": result.domain_match,
        "actual_intent": result.actual_intent,
        "actual_domains": result.actual_domains,
        "behavior_notes": result.behavior_notes,
        "answer_preview": result.answer_preview,
        "quality_score": result.quality_score,
        "grounding_score": result.grounding_score,
        "coverage_score": result.coverage_score,
        "retrieval_hit_rate": result.retrieval_hit_rate,
        "faithfulness_score": result.faithfulness_score,
        "citation_accuracy": result.citation_accuracy,
        "answer_relevance": result.answer_relevance,
        "latency_ms": result.latency_ms,
        "error": result.error,
    }


def summarize_dicts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate flattened results using the same `summary_stats` the CLI uses.

    Rebuilds `TestResult` objects rather than reimplementing the aggregation in
    the browser: the scoping rules (faithfulness excludes edge cases, quality
    excludes correct refusals) are subtle, and two implementations would drift.
    """
    from .cases import TEST_CASES, TestResult
    from .runner import summary_stats

    by_id = {tc.id: tc for tc in TEST_CASES}
    results = []
    for row in rows:
        tc = by_id.get(row.get("id"))
        if tc is None:
            continue
        results.append(
            TestResult(
                test_case=tc,
                passed=bool(row.get("passed")),
                intent_match=bool(row.get("intent_match")),
                domain_match=bool(row.get("domain_match")),
                behavior_notes=str(row.get("behavior_notes", "")),
                actual_intent=str(row.get("actual_intent", "")),
                actual_domains=list(row.get("actual_domains") or []),
                quality_score=float(row.get("quality_score") or 0.0),
                coverage_score=float(row.get("coverage_score") or 0.0),
                grounding_score=float(row.get("grounding_score") or 0.0),
                retrieval_hit_rate=float(row.get("retrieval_hit_rate") or 0.0),
                faithfulness_score=float(row.get("faithfulness_score") or 0.0),
                citation_accuracy=float(row.get("citation_accuracy") or 0.0),
                answer_relevance=float(row.get("answer_relevance") or 0.0),
                latency_ms=float(row.get("latency_ms") or 0.0),
            )
        )
    return summary_stats(results)


__all__ = ["build_pipeline_fn", "result_to_dict", "summarize_dicts"]
