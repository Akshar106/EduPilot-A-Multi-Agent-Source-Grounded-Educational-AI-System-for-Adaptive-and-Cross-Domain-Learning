"""
Evaluation suite
================
50 test cases and the metrics that score them.

  `cases`   — the registry (pure data, no EduPilot imports)
  `metrics` — objective per-run scores, independent of the LLM's self-judgement
  `runner`  — drives cases through an injected pipeline and aggregates results

Listing the suite is cheap; running it is not — a full pass is hundreds of LLM
calls, which is why the API exposes it behind an admin check.
"""

from .cases import TEST_CASES, TestCase, TestResult
from .metrics import (
    answer_relevance,
    citation_accuracy,
    faithfulness,
    retrieval_hit_rate,
)
from .runner import run_all_evaluations, run_evaluation, summary_stats

__all__ = [
    "TEST_CASES",
    "TestCase",
    "TestResult",
    "answer_relevance",
    "citation_accuracy",
    "faithfulness",
    "retrieval_hit_rate",
    "run_all_evaluations",
    "run_evaluation",
    "summary_stats",
]
