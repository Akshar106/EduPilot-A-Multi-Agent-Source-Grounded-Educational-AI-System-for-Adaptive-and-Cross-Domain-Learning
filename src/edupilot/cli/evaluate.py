"""
Evaluation CLI
==============
Run the 50-case suite against the local pipeline and print a scorecard.

    edupilot-evaluate                      run everything
    edupilot-evaluate --category edge-case run one category
    edupilot-evaluate --case TC-01 --case TC-04
    edupilot-evaluate --model llama-3.1-8b-instant --json results.json

Equivalently, `python -m edupilot.cli.evaluate`.

This is a CLI rather than an API route on purpose: a full pass is hundreds of
LLM calls, which is an operator decision, not something a logged-in student
should be able to trigger over HTTP.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from edupilot.core.config import DEFAULT_MODEL, DEFAULT_RERANK_TOP_K, DEFAULT_TOP_K, VERIFY_MODEL
from edupilot.core.observability import configure_logging

configure_logging()
logger = logging.getLogger("edupilot.evaluate")


def _build_pipeline_fn():
    """
    Adapt the agent pipeline to the (query, model, top_k, ...) -> dict contract
    the evaluation runner expects.
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
    ) -> dict:
        result = services.pipeline.run(
            query,
            PipelineConfig(
                model=model,
                verify_model=VERIFY_MODEL,
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


def _print_report(results, stats: dict) -> None:
    print()
    print(f"{'CASE':<8} {'PASS':<6} {'INTENT':<8} {'DOMAIN':<8} {'FAITH':<7} {'HIT':<6} {'CITE':<6} {'ms':>7}")
    print("─" * 64)
    for r in results:
        print(
            f"{r.test_case.id:<8} "
            f"{'ok' if r.passed else 'FAIL':<6} "
            f"{'ok' if r.intent_match else 'FAIL':<8} "
            f"{'ok' if r.domain_match else 'FAIL':<8} "
            f"{r.faithfulness_score:<7.2f} "
            f"{r.retrieval_hit_rate:<6.2f} "
            f"{r.citation_accuracy:<6.2f} "
            f"{r.latency_ms:>7.0f}"
        )

    print("─" * 64)
    print(f"pass rate        {stats['pass_rate']}%  ({stats['passed']}/{stats['total']})")
    print(f"intent accuracy  {stats['intent_accuracy']}%")
    print(f"domain accuracy  {stats['domain_accuracy']}%")
    print(f"faithfulness     {stats['avg_faithfulness']}")
    print(f"retrieval hits   {stats['avg_retrieval_hit_rate']}")
    print(f"citation acc     {stats['avg_citation_accuracy']}")
    print(f"answer relevance {stats['avg_answer_relevance']}")
    print(f"mean latency     {stats['avg_latency_ms']} ms")
    print()
    for category, counts in sorted(stats["by_category"].items()):
        print(f"  {category:<16} {counts['passed']}/{counts['total']}")


def main() -> int:
    from edupilot.evaluation import TEST_CASES, run_evaluation, summary_stats

    parser = argparse.ArgumentParser(description="Run the EduPilot evaluation suite")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model under test")
    parser.add_argument("--case", action="append", metavar="ID", help="run only these case ids")
    parser.add_argument("--category", help="run only this category")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--no-verification", action="store_true")
    parser.add_argument("--json", metavar="PATH", help="also write raw results as JSON")
    args = parser.parse_args()

    cases = TEST_CASES
    if args.case:
        wanted = {c.upper() for c in args.case}
        cases = [tc for tc in cases if tc.id.upper() in wanted]
    if args.category:
        cases = [tc for tc in cases if tc.category == args.category]

    if not cases:
        logger.error("no test cases matched the given filters")
        return 1

    pipeline_fn = _build_pipeline_fn()
    results = []
    for i, tc in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {tc.id} {tc.name}", file=sys.stderr, flush=True)
        results.append(
            run_evaluation(
                test_case=tc,
                pipeline_fn=pipeline_fn,
                model=args.model,
                top_k=args.top_k,
                rerank_top_k=args.rerank_top_k,
                enable_verification=not args.no_verification,
            )
        )

    stats = summary_stats(results)
    _print_report(results, stats)

    if args.json:
        payload = {
            "model": args.model,
            "summary": stats,
            "results": [
                {
                    "id": r.test_case.id,
                    "name": r.test_case.name,
                    "category": r.test_case.category,
                    "passed": r.passed,
                    "intent_match": r.intent_match,
                    "domain_match": r.domain_match,
                    "actual_intent": r.actual_intent,
                    "actual_domains": r.actual_domains,
                    "faithfulness_score": r.faithfulness_score,
                    "retrieval_hit_rate": r.retrieval_hit_rate,
                    "citation_accuracy": r.citation_accuracy,
                    "answer_relevance": r.answer_relevance,
                    "latency_ms": r.latency_ms,
                    "answer_preview": r.answer_preview,
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")

    # Non-zero exit when anything failed, so this is usable as a CI gate.
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
