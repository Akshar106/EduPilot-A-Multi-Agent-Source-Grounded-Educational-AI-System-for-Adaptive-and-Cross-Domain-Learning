"""Evaluation suite integrity and metric behaviour (no LLM calls)."""

from __future__ import annotations

import pytest

from edupilot.evaluation import TEST_CASES, summary_stats
from edupilot.evaluation.cases import TestCase, TestResult
from edupilot.evaluation.metrics import citation_accuracy, retrieval_hit_rate
from edupilot.evaluation.runner import run_evaluation

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_case_ids_are_unique():
    ids = [tc.id for tc in TEST_CASES]
    assert len(ids) == len(set(ids))


def test_expected_intent_is_a_known_value():
    for tc in TEST_CASES:
        assert tc.expected_intent in {"single", "multi", "any"}, tc.id


def test_expected_domains_are_configured_domains():
    from edupilot.core.config import DOMAINS

    for tc in TEST_CASES:
        for domain in tc.expected_domains:
            assert domain in DOMAINS, f"{tc.id} expects unknown domain {domain}"


def test_every_case_has_a_category_and_behaviour():
    for tc in TEST_CASES:
        assert tc.category, tc.id
        assert tc.expected_behavior.strip(), tc.id


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_hit_rate_uses_stem_prefix_matching():
    """"overfit" should match "overfitting" in retrieved text."""
    assert retrieval_hit_rate(["overfit"], ["the model is overfitting badly"]) == 1.0


def test_hit_rate_matches_multiword_keywords_by_substring():
    assert retrieval_hit_rate(["p-value"], ["a small p-value rejects H0"]) == 1.0


def test_hit_rate_is_zero_without_keywords_or_chunks():
    assert retrieval_hit_rate([], ["anything"]) == 0.0
    assert retrieval_hit_rate(["bias"], []) == 0.0


def test_hit_rate_is_a_fraction():
    score = retrieval_hit_rate(["bias", "variance", "quantum"], ["bias and variance tradeoff"])
    assert score == pytest.approx(2 / 3, abs=1e-3)


def test_citation_accuracy_is_perfect_when_there_is_nothing_to_check():
    assert citation_accuracy("no citations here", []) == 1.0


def test_citation_accuracy_credits_a_supported_citation():
    sources = [{"source_num": 1, "text": "Regularization penalises large coefficients."}]
    answer = "Regularization reduces overfitting [Source 1]."
    assert citation_accuracy(answer, sources) == 1.0


def test_citation_accuracy_penalises_a_citation_with_no_overlap():
    sources = [{"source_num": 1, "text": "Photosynthesis converts light into sugar."}]
    answer = "Gradient descent minimises the objective [Source 1]."
    assert citation_accuracy(answer, sources) == 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _case(**kw) -> TestCase:
    base = {
        "id": "TC-X", "name": "synthetic", "query": "q",
        "expected_intent": "single", "expected_domains": ["AML"],
        "expected_behavior": "behaves", "category": "edge-case",
    }
    base.update(kw)
    return TestCase(**base)


def test_runner_reports_a_pipeline_exception_as_a_failure():
    def boom(**_):
        raise RuntimeError("provider down")

    result = run_evaluation(_case(), pipeline_fn=boom)
    assert result.passed is False
    assert "provider down" in result.error


def test_runner_matches_intent_and_domain():
    def fake(**_):
        return {
            "intent_type": "single",
            "detected_domains": ["AML"],
            "final_answer": "grounded answer",
            "quality_score": 0.9,
            "sources": [],
        }

    result = run_evaluation(_case(), pipeline_fn=fake)
    assert result.intent_match and result.domain_match and result.passed
    assert result.latency_ms >= 0


def test_runner_flags_a_domain_mismatch():
    def fake(**_):
        return {"intent_type": "single", "detected_domains": ["STAT"], "final_answer": "a"}

    result = run_evaluation(_case(), pipeline_fn=fake)
    assert result.domain_match is False
    assert "DOMAIN MISMATCH" in result.behavior_notes


def test_edge_cases_are_treated_as_faithful_without_an_llm_call():
    """A correct refusal has nothing to hallucinate, so it must not be judged."""
    def refuse(**_):
        return {
            "intent_type": "single", "detected_domains": ["AML"],
            "final_answer": "I can't answer that.", "sources": [],
        }

    result = run_evaluation(_case(category="edge-case"), pipeline_fn=refuse)
    assert result.faithfulness_score == 1.0


def test_summary_stats_of_nothing_is_empty():
    assert summary_stats([]) == {}


def test_summary_stats_aggregates_by_category():
    results = [
        TestResult(
            test_case=_case(id="A", category="single-domain"),
            passed=True, intent_match=True, domain_match=True, behavior_notes="",
        ),
        TestResult(
            test_case=_case(id="B", category="single-domain"),
            passed=False, intent_match=True, domain_match=False, behavior_notes="",
        ),
    ]
    stats = summary_stats(results)
    assert stats["total"] == 2
    assert stats["passed"] == 1
    assert stats["pass_rate"] == 50.0
    assert stats["intent_accuracy"] == 100.0
    assert stats["by_category"]["single-domain"] == {"total": 2, "passed": 1}
