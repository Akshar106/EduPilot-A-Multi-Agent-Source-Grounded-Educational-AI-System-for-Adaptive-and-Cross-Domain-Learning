"""
Evaluation runner
=================
Drives test cases through a pipeline callable and aggregates the results.

The pipeline is injected rather than imported, so the same suite can score the
live API, a locally-assembled pipeline, or a stub::

    from edupilot.evaluation import run_all_evaluations, summary_stats

    results = run_all_evaluations(pipeline_fn=my_pipeline, model="llama-3.3-70b-versatile")
    print(summary_stats(results))
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from edupilot.core.config import DEFAULT_MODEL

from .cases import TEST_CASES, TestCase, TestResult
from .metrics import answer_relevance, citation_accuracy, faithfulness, retrieval_hit_rate


def run_evaluation(
    test_case: TestCase,
    pipeline_fn: Callable,
    model: str = DEFAULT_MODEL,
    top_k: int = 5,
    rerank_top_k: int = 3,
    enable_verification: bool = True,
) -> TestResult:
    """
    Run a single test case through the pipeline and compute all metrics.

    `pipeline_fn` must accept (query, model, top_k, rerank_top_k,
    enable_verification) and return the dict shape produced by the chat route.
    """
    # ── Run pipeline + measure latency ─────────────────────────────────────
    t0 = time.perf_counter()
    try:
        result = pipeline_fn(
            query=test_case.query,
            model=model,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            enable_verification=enable_verification,
        )
    except Exception as exc:
        return TestResult(
            test_case=test_case,
            passed=False,
            intent_match=False,
            domain_match=False,
            behavior_notes=f"Pipeline raised an exception: {exc}",
            error=str(exc),
        )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Unpack pipeline result ──────────────────────────────────────────────
    actual_intent       = result.get("intent_type", "")
    actual_domains      = result.get("detected_domains") or []
    needs_clarification = result.get("needs_clarification", False)
    is_course_related   = result.get("is_course_related", True)
    final_answer        = result.get("final_answer", "")
    quality_score       = float(result.get("quality_score", 0.0))

    debug = result.get("debug", {})
    verif_debug = debug.get("verification", {})
    coverage_score  = float(verif_debug.get("coverage_score",  quality_score))
    grounding_score = float(verif_debug.get("grounding_score", quality_score))

    sources = result.get("sources", [])
    retrieved_chunk_texts = [s.get("text", "") for s in sources if s.get("text")]

    # ── System behaviour checks ────────────────────────────────────────────
    intent_match = (
        test_case.expected_intent == "any"
        or actual_intent == test_case.expected_intent
    )

    if not test_case.expected_domains:
        domain_match = (
            not actual_domains
            or needs_clarification
            or not is_course_related
        )
    else:
        domain_match = all(d in actual_domains for d in test_case.expected_domains)

    extra_pass = True
    extra_note = ""
    if test_case.check_fn:
        try:
            extra_pass = bool(test_case.check_fn(final_answer))
            if not extra_pass:
                extra_note = " | Programmatic check FAILED (e.g. missing citation)."
        except Exception as exc:
            extra_note = f" | Check error: {exc}"

    passed = intent_match and domain_match and extra_pass

    notes_parts = [test_case.expected_behavior]
    if not intent_match:
        notes_parts.append(
            f"INTENT MISMATCH: expected '{test_case.expected_intent}', got '{actual_intent}'"
        )
    if not domain_match:
        notes_parts.append(
            f"DOMAIN MISMATCH: expected {test_case.expected_domains}, got {actual_domains}"
        )
    if extra_note:
        notes_parts.append(extra_note)

    # ── Objective metrics ───────────────────────────────────────────────────
    hit_rate = retrieval_hit_rate(test_case.relevant_keywords, retrieved_chunk_texts)

    # Edge cases are supposed to produce a refusal, so a correct refusal is
    # perfectly faithful and has no relevance score to compute.
    is_edge_case = test_case.category == "edge-case"
    faithfulness_score = 0.0
    if is_edge_case:
        faithfulness_score = 1.0
    elif final_answer and retrieved_chunk_texts:
        faithfulness_score = faithfulness(
            question=test_case.query,
            answer=final_answer,
            retrieved_chunk_texts=retrieved_chunk_texts,
            model=model,
        )

    relevance = 0.0
    if final_answer and not is_edge_case:
        relevance = answer_relevance(test_case.query, final_answer)

    return TestResult(
        test_case=test_case,
        passed=passed,
        intent_match=intent_match,
        domain_match=domain_match,
        behavior_notes=" | ".join(notes_parts),
        actual_intent=actual_intent,
        actual_domains=actual_domains,
        answer_preview=final_answer[:500],
        quality_score=quality_score,
        coverage_score=coverage_score,
        grounding_score=grounding_score,
        retrieval_hit_rate=hit_rate,
        faithfulness_score=faithfulness_score,
        citation_accuracy=citation_accuracy(final_answer, sources),
        answer_relevance=relevance,
        latency_ms=latency_ms,
        retrieved_chunk_texts=retrieved_chunk_texts,
    )


def run_all_evaluations(
    pipeline_fn: Callable,
    model: str = DEFAULT_MODEL,
    top_k: int = 5,
    rerank_top_k: int = 3,
    enable_verification: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> list[TestResult]:
    """Run every case in `TEST_CASES` and return the results in order."""
    results: list[TestResult] = []
    total = len(TEST_CASES)

    for i, tc in enumerate(TEST_CASES):
        if on_progress:
            on_progress(tc.name, i + 1, total)
        results.append(
            run_evaluation(
                test_case=tc,
                pipeline_fn=pipeline_fn,
                model=model,
                top_k=top_k,
                rerank_top_k=rerank_top_k,
                enable_verification=enable_verification,
            )
        )

    return results


def summary_stats(results: list[TestResult]) -> dict:
    """
    Aggregate statistics across all test results.

    Metric           Scope
    ──────────────── ─────────────────────────────────────────────────────
    pass_rate        All tests
    intent_accuracy  All tests
    domain_accuracy  All tests
    avg_quality      Tests with quality_score > 0 (excludes correct edge-cases)
    avg_faithfulness Substantive answer tests only (not edge-case refusals)
    avg_hit_rate     Tests with relevant_keywords defined
    avg_citation_acc Tests that should produce answers
    avg_relevance    Substantive answer tests
    avg_latency_ms   All tests
    """
    total = len(results)
    if not total:
        return {}

    passed    = sum(1 for r in results if r.passed)
    intent_ok = sum(1 for r in results if r.intent_match)
    domain_ok = sum(1 for r in results if r.domain_match)

    # Quality / coverage / grounding — exclude correct edge-cases (0 by design)
    answer_tests = [r for r in results if r.quality_score > 0]
    avg_quality   = _avg(r.quality_score   for r in answer_tests)
    avg_coverage  = _avg(r.coverage_score  for r in answer_tests)
    avg_grounding = _avg(r.grounding_score for r in answer_tests)

    # Faithfulness — substantive tests only (edge-cases get 1.0 trivially)
    faith_tests = [r for r in results if r.test_case.category != "edge-case"]
    avg_faith = _avg(r.faithfulness_score for r in faith_tests if r.faithfulness_score > 0)

    hr_tests = [r for r in results if r.test_case.relevant_keywords]
    avg_hit_rate  = _avg(r.retrieval_hit_rate for r in hr_tests)
    avg_citation  = _avg(r.citation_accuracy  for r in answer_tests)
    avg_relevance = _avg(r.answer_relevance for r in faith_tests if r.answer_relevance > 0)
    avg_latency   = _avg(r.latency_ms for r in results)

    categories: dict[str, dict] = {}
    for r in results:
        cat = categories.setdefault(r.test_case.category, {"total": 0, "passed": 0})
        cat["total"] += 1
        if r.passed:
            cat["passed"] += 1

    return {
        "total":                  total,
        "passed":                 passed,
        "failed":                 total - passed,
        "pass_rate":              _pct(passed, total),
        "intent_accuracy":        _pct(intent_ok, total),
        "domain_accuracy":        _pct(domain_ok, total),
        "avg_quality_score":      round(avg_quality, 3),
        "avg_answer_quality":     round(avg_quality, 3),   # kept for UI compatibility
        "answer_tests_count":     len(answer_tests),
        "avg_coverage_score":     round(avg_coverage, 3),
        "avg_grounding_score":    round(avg_grounding, 3),
        "avg_faithfulness":       round(avg_faith, 3),
        "avg_retrieval_hit_rate": round(avg_hit_rate, 3),
        "avg_citation_accuracy":  round(avg_citation, 3),
        "avg_answer_relevance":   round(avg_relevance, 3),
        "avg_latency_ms":         round(avg_latency, 1),
        "by_category":            categories,
    }


def _avg(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 1) if denominator else 0.0


__all__ = ["run_all_evaluations", "run_evaluation", "summary_stats"]
