"""
EduPilot — Multi-Model Comparison Script
=========================================
Runs 10 representative queries through three models and records key metrics.

Models compared
───────────────
  llama-3.3-70b-versatile  (Groq  — primary, large)
  llama-3.1-8b-instant     (Groq  — small / fast)
  gemini-2.0-flash         (Google — cross-provider)

Usage (run from the project root with the server NOT running):
    python model_comparison.py

Output
──────
  • Printed table to stdout
  • model_comparison_results.csv  (same directory)

NOTE: Due to API rate limits this comparison uses 10 targeted queries.
      The main 50-query evaluation suite remains the primary benchmark.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — make sure imports resolve from project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import database as db
from config import DEFAULT_TOP_K, DEFAULT_RERANK_TOP_K, DEFAULT_CONFIDENCE_THRESHOLD
from retriever import initialize_all_retrievers, get_retriever
from router import classify_query, get_clarification_message, get_out_of_domain_message, should_ask_for_clarification
from query_splitter import split_query
from reranker import rerank, score_summary
from synthesizer import generate_domain_answer, synthesize_answers
from verifier import verify_answer, get_final_answer
from evaluation import (
    _compute_retrieval_hit_rate,
    _compute_citation_accuracy,
    _compute_answer_relevance,
    _compute_faithfulness,
)

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------
MODELS = [
    "llama-3.3-70b-versatile",   # Groq — primary (large)
    "llama-3.1-8b-instant",      # Groq — small / fast
    "gemini-2.0-flash",          # Google — cross-provider
]

# ---------------------------------------------------------------------------
# 10 comparison queries — one per domain category + multi-domain + adversarial
# ---------------------------------------------------------------------------
COMPARISON_QUERIES = [
    # ── Single-domain: AML ────────────────────────────────────────────────
    {
        "id": "CQ-01",
        "query": "What is the bias-variance tradeoff in machine learning?",
        "category": "single-domain",
        "domain": "AML",
        "keywords": ["bias", "variance", "tradeoff", "overfit", "underfit", "error"],
    },
    {
        "id": "CQ-02",
        "query": "How does the backpropagation algorithm compute gradients in a neural network?",
        "category": "single-domain",
        "domain": "AML",
        "keywords": ["backpropagation", "gradient", "chain rule", "weight", "loss"],
    },
    # ── Single-domain: ADT ────────────────────────────────────────────────
    {
        "id": "CQ-03",
        "query": "What are the ACID properties of database transactions?",
        "category": "single-domain",
        "domain": "ADT",
        "keywords": ["ACID", "atomicity", "consistency", "isolation", "durability"],
    },
    {
        "id": "CQ-04",
        "query": "How do database indexes work and when should you avoid them?",
        "category": "single-domain",
        "domain": "ADT",
        "keywords": ["index", "B-tree", "query", "lookup", "write", "overhead"],
    },
    # ── Single-domain: STAT ───────────────────────────────────────────────
    {
        "id": "CQ-05",
        "query": "What is a p-value and how do you interpret it in hypothesis testing?",
        "category": "single-domain",
        "domain": "STAT",
        "keywords": ["p-value", "hypothesis", "null", "significance", "probability"],
    },
    # ── Single-domain: LLM ────────────────────────────────────────────────
    {
        "id": "CQ-06",
        "query": "How does the self-attention mechanism work in transformer models?",
        "category": "single-domain",
        "domain": "LLM",
        "keywords": ["attention", "query", "key", "value", "softmax", "transformer"],
    },
    # ── Multi-domain ──────────────────────────────────────────────────────
    {
        "id": "CQ-07",
        "query": (
            "How do I use statistical tests to evaluate whether one machine learning model "
            "is significantly better than another?"
        ),
        "category": "multi-domain",
        "domain": "AML+STAT",
        "keywords": ["t-test", "accuracy", "model", "significance", "hypothesis", "cross-validation"],
    },
    {
        "id": "CQ-08",
        "query": (
            "How are vector embeddings from language models stored and queried in a vector database?"
        ),
        "category": "multi-domain",
        "domain": "LLM+ADT",
        "keywords": ["embedding", "vector", "database", "similarity", "index", "query"],
    },
    # ── Adversarial ───────────────────────────────────────────────────────
    {
        "id": "CQ-09",
        "query": "Explain the XYZ-9000 neural compression algorithm invented at MIT in 2023.",
        "category": "adversarial",
        "domain": "AML",
        "keywords": [],  # fabricated — should refuse or hedge, not hallucinate
    },
    {
        "id": "CQ-10",
        "query": (
            "Since neural networks always outperform linear models, which neural "
            "architecture should I always use?"
        ),
        "category": "adversarial",
        "domain": "AML",
        "keywords": ["neural network", "linear", "overfit", "interpretab", "bias-variance"],
    },
]

# ---------------------------------------------------------------------------
# Pipeline runner (mirrors _run_pipeline in main.py, no FastAPI dependency)
# ---------------------------------------------------------------------------

def run_pipeline(query: str, model: str) -> dict:
    router_result = classify_query(query, model=model, chat_history=[])

    if not router_result.is_course_related:
        return {
            "final_answer": get_out_of_domain_message(),
            "quality_score": 0.0,
            "sources": [],
            "debug": {},
        }

    if should_ask_for_clarification(router_result, query):
        return {
            "final_answer": get_clarification_message(router_result),
            "quality_score": 0.0,
            "sources": [],
            "debug": {},
        }

    sub_questions = split_query(
        query=query,
        intent_type=router_result.intent_type,
        detected_domains=router_result.domains,
        model=model,
    )

    domain_answers = []
    for sq in sub_questions:
        retriever = get_retriever(sq["domain"])
        raw_chunks = retriever.retrieve(sq["question"], top_k=DEFAULT_TOP_K)
        reranked = rerank(
            query=sq["question"],
            chunks=raw_chunks,
            top_k=DEFAULT_RERANK_TOP_K,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        da = generate_domain_answer(
            sub_question=sq["question"],
            domain=sq["domain"],
            retrieved_chunks=reranked,
            model=model,
            chat_history=[],
        )
        domain_answers.append(da)

    synthesized = synthesize_answers(
        original_query=query,
        domain_answers=domain_answers,
        model=model,
    )

    verification = verify_answer(
        original_query=query,
        sub_questions=sub_questions,
        domain_answers=domain_answers,
        synthesized_answer=synthesized,
        model=model,
        enabled=True,
    )

    final = get_final_answer(synthesized, verification)

    sources = []
    for da in domain_answers:
        for i, chunk in enumerate(da.retrieved_chunks, 1):
            sources.append({
                "source_num": i,
                "domain": da.domain,
                "text": chunk.text,
                "citation_label": chunk.citation_label(),
            })

    return {
        "final_answer": final,
        "quality_score": verification.quality_score,
        "coverage_score": verification.coverage_score,
        "grounding_score": verification.grounding_score,
        "sources": sources,
        "retrieved_chunk_texts": [s["text"] for s in sources],
        "verification_revised": verification.revised_answer is not None,
    }


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def run_comparison() -> list[dict]:
    db.init_db()
    print("Initialising retrievers …", flush=True)
    initialize_all_retrievers()
    print("Done.\n", flush=True)

    rows: list[dict] = []

    for q in COMPARISON_QUERIES:
        print(f"{'─'*70}")
        print(f"Query {q['id']} [{q['category']}] — {q['query'][:70]}")
        for model in MODELS:
            short = model.split("-")[0] + "-" + model.split("-")[-1]  # e.g. llama-versatile
            print(f"  Running {model} …", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                result = run_pipeline(q["query"], model)
                latency = round((time.perf_counter() - t0) * 1000)

                answer = result["final_answer"]
                sources = result.get("sources", [])
                chunks = result.get("retrieved_chunk_texts", [])

                hit_rate = _compute_retrieval_hit_rate(q["keywords"], chunks) if q["keywords"] else None
                citation_acc = _compute_citation_accuracy(answer, sources)
                relevance = _compute_answer_relevance(q["query"], answer) if answer else 0.0
                faithfulness = (
                    _compute_faithfulness(q["query"], answer, chunks, model=model)
                    if chunks and q["category"] != "adversarial"
                    else None
                )

                row = {
                    "query_id": q["id"],
                    "category": q["category"],
                    "query": q["query"][:80],
                    "model": model,
                    "quality_score": round(result.get("quality_score", 0.0), 3),
                    "coverage_score": round(result.get("coverage_score", 0.0), 3),
                    "grounding_score": round(result.get("grounding_score", 0.0), 3),
                    "retrieval_hit_rate": round(hit_rate, 3) if hit_rate is not None else "N/A",
                    "citation_accuracy": round(citation_acc, 3),
                    "answer_relevance": round(relevance, 3),
                    "faithfulness": round(faithfulness, 3) if faithfulness is not None else "N/A",
                    "latency_ms": latency,
                    "revised": result.get("verification_revised", False),
                    "error": "",
                }
                print(f"quality={row['quality_score']}  latency={latency}ms")

            except Exception as exc:
                latency = round((time.perf_counter() - t0) * 1000)
                row = {
                    "query_id": q["id"],
                    "category": q["category"],
                    "query": q["query"][:80],
                    "model": model,
                    "quality_score": None,
                    "coverage_score": None,
                    "grounding_score": None,
                    "retrieval_hit_rate": None,
                    "citation_accuracy": None,
                    "answer_relevance": None,
                    "faithfulness": None,
                    "latency_ms": latency,
                    "revised": False,
                    "error": str(exc)[:120],
                }
                print(f"ERROR — {exc}")

            rows.append(row)

    return rows


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'═'*70}")
    print("SUMMARY — average metrics per model (generating queries only)")
    print(f"{'═'*70}")

    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model and r["quality_score"] is not None and r["quality_score"] > 0]
        if not model_rows:
            print(f"\n{model}: no valid results")
            continue

        def avg(key):
            vals = [r[key] for r in model_rows if isinstance(r.get(key), (int, float))]
            return round(sum(vals) / len(vals), 3) if vals else "N/A"

        print(f"\n  {model}")
        print(f"    Avg quality score    : {avg('quality_score')}")
        print(f"    Avg coverage score   : {avg('coverage_score')}")
        print(f"    Avg grounding score  : {avg('grounding_score')}")
        print(f"    Avg retrieval hit    : {avg('retrieval_hit_rate')}")
        print(f"    Avg citation acc     : {avg('citation_accuracy')}")
        print(f"    Avg answer relevance : {avg('answer_relevance')}")
        print(f"    Avg faithfulness     : {avg('faithfulness')}")
        print(f"    Avg latency (ms)     : {avg('latency_ms')}")
        revised = sum(1 for r in model_rows if r.get("revised"))
        print(f"    Verifier revisions   : {revised}/{len(model_rows)}")


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_comparison()
    print_summary(results)
    csv_path = ROOT / "model_comparison_results.csv"
    save_csv(results, csv_path)
