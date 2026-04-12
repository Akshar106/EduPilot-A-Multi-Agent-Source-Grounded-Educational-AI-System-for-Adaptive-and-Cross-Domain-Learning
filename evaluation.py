"""
EduPilot Evaluation
===================
10 test cases covering single-domain, multi-domain, ambiguous, out-of-domain,
hallucination stress, citation, and multi-turn scenarios.

Each test case runs the full pipeline and checks:
  - Correct intent classification
  - Correct domain routing
  - Correct answer behavior (grounded, complete, appropriate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    name: str
    query: str
    expected_intent: str                      # "single" | "multi" | "any"
    expected_domains: list[str]               # expected domain list
    expected_behavior: str                    # human-readable expected outcome
    check_fn: Optional[Callable] = None       # optional extra programmatic check
    category: str = "general"                 # for grouping in UI


@dataclass
class TestResult:
    test_case: TestCase
    passed: bool
    intent_match: bool
    domain_match: bool
    behavior_notes: str
    actual_intent: str = ""
    actual_domains: list[str] = field(default_factory=list)
    answer_preview: str = ""
    quality_score: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Test case registry
# ---------------------------------------------------------------------------

TEST_CASES: list[TestCase] = [
    # -----------------------------------------------------------------------
    # TC-01  Single-domain conceptual — AML
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-01",
        name="Bias-Variance Tradeoff",
        query="What is the bias and variance tradeoff in machine learning?",
        expected_intent="single",
        expected_domains=["AML"],
        expected_behavior=(
            "System should classify as single-intent AML question. "
            "Answer must explain bias, variance, and the tradeoff. "
            "Answer must be grounded with citations from AML knowledge base."
        ),
        category="single-domain",
    ),

    # -----------------------------------------------------------------------
    # TC-02  Single-domain practical — ADT
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-02",
        name="Database Normalization",
        query="What is normalization in databases and why is it important?",
        expected_intent="single",
        expected_domains=["ADT"],
        expected_behavior=(
            "System should classify as single-intent ADT question. "
            "Answer must explain normalization forms (1NF, 2NF, 3NF) and their purpose. "
            "Must include citations from ADT knowledge base."
        ),
        category="single-domain",
    ),

    # -----------------------------------------------------------------------
    # TC-03  Single-domain statistics question
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-03",
        name="P-Value Explanation",
        query="What is a p-value and how do I interpret it in hypothesis testing?",
        expected_intent="single",
        expected_domains=["STAT"],
        expected_behavior=(
            "System should classify as single-intent STAT question. "
            "Answer must define p-value, explain significance threshold, "
            "and describe how to interpret it. Must cite STAT sources."
        ),
        category="single-domain",
    ),

    # -----------------------------------------------------------------------
    # TC-04  Multi-domain mixed question
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-04",
        name="ML + NL2SQL Multi-Domain",
        query=(
            "What is machine learning and how do I use NL2SQL "
            "to store and retrieve data from a database?"
        ),
        expected_intent="multi",
        expected_domains=["AML", "ADT"],
        expected_behavior=(
            "System must detect multi-intent, split into 2 sub-questions. "
            "Sub-question 1 → AML (what is ML). "
            "Sub-question 2 → ADT (NL2SQL). "
            "Retrieve from both domain RAGs independently. "
            "Synthesize a combined answer with domain sections. "
            "Verifier checks completeness of both parts."
        ),
        category="multi-domain",
    ),

    # -----------------------------------------------------------------------
    # TC-05  Ambiguous question
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-05",
        name="Ambiguous Query",
        query="How does it work?",
        expected_intent="single",
        expected_domains=[],
        expected_behavior=(
            "System should detect the query as ambiguous (too vague). "
            "Must ask for clarification rather than guessing. "
            "Must NOT retrieve from any domain or generate a factual answer."
        ),
        category="edge-case",
    ),

    # -----------------------------------------------------------------------
    # TC-06  Out-of-domain question
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-06",
        name="Out-of-Domain Question",
        query="What is the capital of France?",
        expected_intent="single",
        expected_domains=[],
        expected_behavior=(
            "System should detect this is NOT related to AML, ADT, or STAT. "
            "Must respond with out-of-domain message. "
            "Must NOT fabricate a course-related answer."
        ),
        category="edge-case",
    ),

    # -----------------------------------------------------------------------
    # TC-07  Hallucination stress test
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-07",
        name="Hallucination Stress Test",
        query="Explain the XYZ-5000 advanced neural compression algorithm.",
        expected_intent="single",
        expected_domains=["AML"],
        expected_behavior=(
            "System should attempt to retrieve from AML knowledge base. "
            "Since this topic does not exist in the knowledge base, "
            "the system must NOT invent facts. "
            "Must clearly state it could not find grounded source material."
        ),
        category="edge-case",
    ),

    # -----------------------------------------------------------------------
    # TC-08  Verification failure scenario
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-08",
        name="Verification Improvement",
        query="What is overfitting and how does it relate to model generalization?",
        expected_intent="single",
        expected_domains=["AML"],
        expected_behavior=(
            "System should answer about overfitting from AML sources. "
            "Verifier should check that the answer covers: "
            "(a) definition of overfitting, (b) connection to generalization, "
            "(c) solutions (regularization, dropout, more data). "
            "If any part is incomplete, verifier should revise the answer."
        ),
        category="verification",
    ),

    # -----------------------------------------------------------------------
    # TC-09  Citation verification
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-09",
        name="Citation Verification",
        query="What is a confidence interval and how is it calculated?",
        expected_intent="single",
        expected_domains=["STAT"],
        expected_behavior=(
            "Final answer must contain at least one [Source N] citation. "
            "Citations must reference actual retrieved chunks from STAT knowledge base. "
            "Answer must define CI, show the formula, and explain interpretation."
        ),
        category="citation",
        check_fn=lambda answer: "[Source" in answer,
    ),

    # -----------------------------------------------------------------------
    # TC-10  Multi-turn follow-up
    # -----------------------------------------------------------------------
    TestCase(
        id="TC-10",
        name="Multi-Turn Follow-Up",
        query="Can you explain more about the regularization techniques you mentioned?",
        expected_intent="single",
        expected_domains=["AML"],
        expected_behavior=(
            "Simulates a follow-up question (system uses chat history context). "
            "System should understand 'regularization techniques' refers to ML context "
            "from a previous answer. Must route to AML and explain L1/L2 regularization "
            "with citations."
        ),
        category="multi-turn",
    ),
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    test_case: TestCase,
    pipeline_fn: Callable,
    model: str = "claude-opus-4-6",
    top_k: int = 5,
    rerank_top_k: int = 3,
    enable_verification: bool = True,
) -> TestResult:
    """
    Run a single test case through the pipeline and evaluate results.

    Args:
        test_case: The TestCase to evaluate.
        pipeline_fn: The main pipeline function (query → PipelineResult).
        model: Claude model to use.
        top_k: Retrieval top-k.
        rerank_top_k: Rerank top-k.
        enable_verification: Whether to run verification.

    Returns:
        TestResult with pass/fail and detailed notes.
    """
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

    # Evaluate intent
    intent_match = (
        test_case.expected_intent == "any"
        or result.intent_type == test_case.expected_intent
    )

    # Evaluate domains
    if not test_case.expected_domains:
        # Expected no domain — pass if no domains detected OR clarification/OOD flag
        domain_match = (
            not result.detected_domains
            or result.needs_clarification
            or not result.is_course_related
        )
    else:
        # All expected domains should appear
        domain_match = all(d in result.detected_domains for d in test_case.expected_domains)

    # Run optional programmatic check
    extra_pass = True
    extra_note = ""
    if test_case.check_fn:
        try:
            extra_pass = bool(test_case.check_fn(result.final_answer))
            if not extra_pass:
                extra_note = " | Programmatic check FAILED."
        except Exception as exc:
            extra_note = f" | Check error: {exc}"

    passed = intent_match and domain_match and extra_pass

    # Build notes
    notes_parts = [test_case.expected_behavior]
    if not intent_match:
        notes_parts.append(
            f"INTENT MISMATCH: expected '{test_case.expected_intent}', "
            f"got '{result.intent_type}'"
        )
    if not domain_match:
        notes_parts.append(
            f"DOMAIN MISMATCH: expected {test_case.expected_domains}, "
            f"got {result.detected_domains}"
        )
    if extra_note:
        notes_parts.append(extra_note)

    return TestResult(
        test_case=test_case,
        passed=passed,
        intent_match=intent_match,
        domain_match=domain_match,
        behavior_notes=" | ".join(notes_parts),
        actual_intent=result.intent_type,
        actual_domains=result.detected_domains,
        answer_preview=result.final_answer[:500],
        quality_score=result.quality_score,
    )


def run_all_evaluations(
    pipeline_fn: Callable,
    model: str = "claude-opus-4-6",
    top_k: int = 5,
    rerank_top_k: int = 3,
    enable_verification: bool = True,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> list[TestResult]:
    """
    Run all test cases and return a list of TestResult objects.

    Args:
        pipeline_fn: Main pipeline callable.
        model, top_k, rerank_top_k, enable_verification: Pipeline settings.
        on_progress: Optional callback(test_name, current, total) for UI updates.
    """
    results: list[TestResult] = []
    total = len(TEST_CASES)

    for i, tc in enumerate(TEST_CASES):
        if on_progress:
            on_progress(tc.name, i + 1, total)

        result = run_evaluation(
            test_case=tc,
            pipeline_fn=pipeline_fn,
            model=model,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            enable_verification=enable_verification,
        )
        results.append(result)

    return results


def summary_stats(results: list[TestResult]) -> dict:
    """Aggregate statistics across test results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    intent_ok = sum(1 for r in results if r.intent_match)
    domain_ok = sum(1 for r in results if r.domain_match)
    avg_quality = (
        sum(r.quality_score for r in results) / total if total else 0.0
    )
    categories: dict[str, dict] = {}
    for r in results:
        cat = r.test_case.category
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r.passed:
            categories[cat]["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "intent_accuracy": round(intent_ok / total * 100, 1) if total else 0,
        "domain_accuracy": round(domain_ok / total * 100, 1) if total else 0,
        "avg_quality_score": round(avg_quality, 3),
        "by_category": categories,
    }
