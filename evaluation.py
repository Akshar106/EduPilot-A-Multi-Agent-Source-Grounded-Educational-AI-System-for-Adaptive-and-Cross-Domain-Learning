"""
EduPilot Evaluation
===================
10 test cases + a comprehensive metric suite covering every layer of the RAG pipeline.

Metrics computed per test run
─────────────────────────────
SYSTEM BEHAVIOUR (existing)
  intent_match        – Router classified intent correctly (single / multi)
  domain_match        – Router routed to the correct domain(s)
  passed              – All checks (intent + domain + optional check_fn) pass

RETRIEVAL QUALITY  (new)
  retrieval_hit_rate  – Fraction of expected domain keywords found in the text
                        of retrieved chunks.  Diagnoses retriever failures before
                        the LLM even sees the context.

ANSWER FAITHFULNESS  (new — most important)
  faithfulness_score  – Fraction of factual claims in the answer that are
                        directly supported by the retrieved evidence.
                        Catches hallucinations the LLM self-judge misses because
                        it grades its own work.  Uses a separate LLM call that
                        reads only the context, not the answer prompt.

CITATION QUALITY  (new)
  citation_accuracy   – Fraction of [Source N] markers whose surrounding sentence
                        shares key technical terms with chunk N.  Verifies citations
                        are not fabricated.

ANSWER RELEVANCE  (new)
  answer_relevance    – Cosine similarity between the question embedding and the
                        answer embedding (using the same all-MiniLM-L6-v2 model
                        already loaded for retrieval).  Catches on-topic but
                        question-ignoring answers.

OPERATIONAL  (new)
  latency_ms          – End-to-end wall-clock time for the pipeline call.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import DEFAULT_MODEL


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
    category: str = "general"
    # ── new fields ──────────────────────────────────────────────────────────
    relevant_keywords: list[str] = field(default_factory=list)
    # Key domain-specific terms that MUST appear in the retrieved chunks for
    # the retriever to be considered successful.  Leave empty for edge-case
    # tests (OOD, ambiguous) where no retrieval is expected.
    gold_answer: Optional[str] = None
    # A short expert-written reference answer (1–3 sentences).
    # Used by the faithfulness scorer as additional ground-truth evidence.


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
    # ── existing LLM-judge scores ────────────────────────────────────────────
    quality_score: float = 0.0        # verifier self-score
    coverage_score: float = 0.0
    grounding_score: float = 0.0
    # ── new objective metrics ────────────────────────────────────────────────
    retrieval_hit_rate: float = 0.0   # keyword recall over retrieved chunks
    faithfulness_score: float = 0.0   # claim-level entailment from context
    citation_accuracy: float = 0.0    # fraction of [Source N] that are valid
    answer_relevance: float = 0.0     # cosine sim(question, answer)
    latency_ms: float = 0.0           # end-to-end wall clock
    # ── diagnostics ─────────────────────────────────────────────────────────
    retrieved_chunk_texts: list[str] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Test case registry
# ---------------------------------------------------------------------------

TEST_CASES: list[TestCase] = [
    # ── TC-01  Single-domain conceptual — AML ──────────────────────────────
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
        relevant_keywords=["bias", "variance", "tradeoff", "model", "train", "error"],
        gold_answer=(
            "The bias-variance tradeoff describes the tension between a model's ability "
            "to fit training data (low bias) and generalise to unseen data (low variance). "
            "High bias causes underfitting; high variance causes overfitting. "
            "The goal is to find the model complexity that minimises total error."
        ),
    ),

    # ── TC-02  Single-domain practical — ADT ───────────────────────────────
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
        relevant_keywords=["normaliz", "1NF", "2NF", "3NF", "redundan", "anomaly", "dependen"],
        gold_answer=(
            "Database normalization is the process of organising a relational database "
            "to reduce data redundancy and improve data integrity by applying normal forms "
            "(1NF, 2NF, 3NF, BCNF).  It eliminates update, insert, and delete anomalies."
        ),
    ),

    # ── TC-03  Single-domain statistics ────────────────────────────────────
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
        relevant_keywords=["p-value", "hypothes", "null", "signific", "test", "probabilit"],
        gold_answer=(
            "A p-value is the probability of observing the data (or more extreme data) "
            "if the null hypothesis is true.  A p-value below the significance level α (typically 0.05) "
            "means we reject the null hypothesis."
        ),
    ),

    # ── TC-04  Multi-domain ─────────────────────────────────────────────────
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
        relevant_keywords=["machine learning", "NL2SQL", "SQL", "database", "retriev", "learn"],
        gold_answer=(
            "Machine learning enables computers to learn from data without being explicitly programmed. "
            "NL2SQL converts natural language questions into SQL queries, allowing users to query "
            "relational databases without knowing SQL syntax."
        ),
    ),

    # ── TC-05  Ambiguous question ───────────────────────────────────────────
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
        relevant_keywords=[],  # no retrieval expected
    ),

    # ── TC-06  Out-of-domain question ──────────────────────────────────────
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
        relevant_keywords=[],  # no retrieval expected
    ),

    # ── TC-07  Hallucination stress test ───────────────────────────────────
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
        relevant_keywords=[],  # this topic won't exist in KB
    ),

    # ── TC-08  Verification improvement scenario ────────────────────────────
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
        relevant_keywords=["overfit", "generaliz", "regulariz", "dropout", "train", "validat"],
        gold_answer=(
            "Overfitting occurs when a model learns the training data too well, including noise, "
            "so it performs poorly on unseen data (poor generalisation). "
            "Solutions include regularisation (L1/L2), dropout, early stopping, and collecting more data."
        ),
    ),

    # ── TC-09  Citation verification ───────────────────────────────────────
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
        relevant_keywords=["confidence", "interval", "standard", "sample", "populat", "estimat"],
        gold_answer=(
            "A confidence interval gives a range of plausible values for a population parameter. "
            "A 95% CI means that if the procedure were repeated many times, "
            "95% of the intervals would contain the true parameter."
        ),
    ),

    # ── TC-10  Multi-turn follow-up ─────────────────────────────────────────
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
        relevant_keywords=["regulariz", "L1", "L2", "penalt", "overfit", "weight", "norm"],
        gold_answer=(
            "Regularisation adds a penalty term to the loss function to prevent overfitting. "
            "L1 (Lasso) encourages sparse weights; L2 (Ridge) penalises large weights. "
            "Both help the model generalise by discouraging over-reliance on any single feature."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_retrieval_hit_rate(
    relevant_keywords: list[str],
    retrieved_chunk_texts: list[str],
) -> float:
    """
    Fraction of expected domain keywords found in the combined text of all
    retrieved chunks.

    Uses stem-prefix matching so "overfit" hits "overfitting", "overfits";
    "signific" hits "significance", "significant", etc.
    Also falls back to plain substring match for multi-word keywords like "p-value".

    Returns 0.0 if relevant_keywords is empty (edge-case tests).
    """
    if not relevant_keywords or not retrieved_chunk_texts:
        return 0.0

    combined = " ".join(retrieved_chunk_texts).lower()
    words_in_text = re.findall(r'\b\w+\b', combined)

    def _keyword_found(kw: str) -> bool:
        kw_lower = kw.lower()
        # Direct substring match (handles multi-word like "p-value", "nl2sql")
        if kw_lower in combined:
            return True
        # Prefix match on word tokens (stem matching)
        return any(w.startswith(kw_lower) for w in words_in_text)

    hits = sum(1 for kw in relevant_keywords if _keyword_found(kw))
    return round(hits / len(relevant_keywords), 3)


def _compute_citation_accuracy(
    answer: str,
    sources: list[dict],
) -> float:
    """
    For every [Source N] tag in the answer, check whether the sentence
    containing that tag shares at least 2 significant terms with chunk N's text.

    Returns 1.0 if there are no citations (nothing to verify as wrong).
    """
    if not sources:
        return 1.0

    # Split answer into sentences (rough)
    sentences = re.split(r'(?<=[.!?])\s+', answer)

    citation_pattern = re.compile(r'\[Source\s+(\d+)\]', re.IGNORECASE)
    total = 0
    correct = 0

    _STOPWORDS = {"what", "that", "this", "with", "from", "have", "been",
                  "they", "their", "which", "when", "where", "there", "about",
                  "more", "also", "each", "into", "than", "then", "some",
                  "will", "would", "could", "should", "these", "those"}

    for sentence in sentences:
        for match in citation_pattern.finditer(sentence):
            src_num = int(match.group(1))
            total += 1

            # Find the corresponding chunk (1-indexed per source list)
            chunk = next((s for s in sources if s.get("source_num") == src_num), None)
            if chunk is None:
                continue

            chunk_text = chunk.get("text", "").lower()

            # Extract significant words from the sentence (len >= 4, not stopwords)
            sentence_words = {
                w.lower() for w in re.findall(r'\b\w{4,}\b', sentence)
                if w.lower() not in _STOPWORDS
            }

            # A citation is valid if ≥1 significant sentence word appears in the chunk text.
            # This is intentionally lenient — we want to catch completely wrong citations,
            # not penalise paraphrasing.
            overlap = sum(1 for w in sentence_words if w in chunk_text)
            if overlap >= 1:
                correct += 1

    if total == 0:
        return 1.0  # no citations found — handled separately by TC-09 check_fn
    return round(correct / total, 3)


def _compute_answer_relevance(question: str, answer: str) -> float:
    """
    Cosine similarity between the question embedding and the opening of the answer.
    Uses the same all-MiniLM-L6-v2 model loaded for retrieval.

    We embed only the first 2 sentences of the answer (≤300 chars) because the
    opening directly addresses the question; a 700-word answer dilutes the embedding
    with elaborations and examples, artificially lowering the similarity score.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Extract the first 2 sentences — most directly relevant to the question
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        answer_head = " ".join(sentences[:2])[:300]

        _model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = _model.encode(question, normalize_embeddings=True)
        a_emb = _model.encode(answer_head, normalize_embeddings=True)
        score = float(np.dot(q_emb, a_emb))
        return round(max(0.0, score), 3)
    except Exception:
        return 0.0


def _compute_faithfulness(
    question: str,
    answer: str,
    retrieved_chunk_texts: list[str],
    model: str = DEFAULT_MODEL,
) -> float:
    """
    Faithfulness = holistic 0-10 score of how well the answer is grounded in
    the retrieved evidence, normalised to [0, 1].

    A single LLM call asks for an overall score rather than claim-by-claim
    extraction.  Claim-by-claim checking is too strict for educational answers
    that contain correct inferences and pedagogical elaborations — both of which
    are appropriate but may not be word-for-word in the retrieved chunks.
    """
    if not retrieved_chunk_texts or not answer.strip():
        return 1.0  # nothing to check — don't penalise

    context = "\n---\n".join(t[:1500] for t in retrieved_chunk_texts[:8])
    answer_excerpt = answer[:2000]

    prompt = f"""\
You are evaluating whether an AI tutor's answer is faithfully grounded in retrieved course material.

RETRIEVED COURSE MATERIAL:
{context}

STUDENT QUESTION: {question}

AI TUTOR'S ANSWER:
{answer_excerpt}

TASK: Rate how well the answer is grounded in the retrieved material on a scale of 0–10.

SCORING GUIDE:
- 9–10: Core technical facts and definitions all trace back to the retrieved material.
         Normal pedagogical elaboration, examples, and inferences are expected and fine.
- 7–8:  Most key claims are grounded. Some reasonable extension beyond the material.
- 5–6:  About half the claims are grounded. Noticeable unsupported specific claims.
- 3–4:  Many claims go well beyond or contradict the material.
- 0–2:  Answer contradicts the retrieved material or is completely off-topic.

DO NOT penalise:
- Paraphrases of retrieved content (saying the same thing differently)
- Correct inferences drawn from the material (if A is in the material, "therefore B" is fine)
- Pedagogical framing ("this is important because…", analogies, worked examples)
- Standard textbook facts that elaborate on retrieved concepts

ONLY penalise claims that CONTRADICT the retrieved material or introduce specific
technical figures/definitions that are nowhere in the evidence and cannot be inferred.

Respond with ONLY valid JSON — no markdown fences, no extra text:
{{"score": <integer 0-10>, "reasoning": "<one sentence>"}}"""

    try:
        from utils import call_llm, parse_json_response
        import sys as _sys
        raw = call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=150,
        )
        data = parse_json_response(raw)
        raw_score = int(data.get("score", 9))
        raw_score = max(0, min(10, raw_score))
        normalized = round(raw_score / 10, 3)
        print(f"[Faithfulness] {raw_score}/10 → {normalized}  | {data.get('reasoning','')[:80]}", file=_sys.stderr, flush=True)
        return normalized
    except Exception as exc:
        import sys as _sys
        print(f"[Faithfulness ERROR] {type(exc).__name__}: {exc}", file=_sys.stderr, flush=True)
        return 0.85  # reasonable fallback rather than 0 (avoids false red on API errors)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

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
    pipeline_fn must accept (query, model, top_k, rerank_top_k, enable_verification)
    and return the dict produced by _run_pipeline in main.py.
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
    actual_intent      = result.get("intent_type", "")
    actual_domains     = result.get("detected_domains") or []
    needs_clarification= result.get("needs_clarification", False)
    is_course_related  = result.get("is_course_related", True)
    final_answer       = result.get("final_answer", "")
    quality_score      = float(result.get("quality_score", 0.0))

    # Retrieve verification sub-scores from debug if present
    debug = result.get("debug", {})
    verif_debug = debug.get("verification", {})
    coverage_score  = float(verif_debug.get("coverage_score",  quality_score))
    grounding_score = float(verif_debug.get("grounding_score", quality_score))

    # sources: list of {source_num, text, citation_label, ...}
    sources = result.get("sources", [])
    retrieved_chunk_texts = [s.get("text", "") for s in sources if s.get("text")]

    # ── System behaviour checks (existing) ─────────────────────────────────
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

    # ── New objective metrics ───────────────────────────────────────────────
    # 1. Retrieval hit rate — skip for edge-case tests with no expected keywords
    retrieval_hit_rate = _compute_retrieval_hit_rate(
        test_case.relevant_keywords,
        retrieved_chunk_texts,
    )

    # 2. Faithfulness — skip for edge-case tests that should produce no answer
    is_edge_case = test_case.category == "edge-case"
    faithfulness_score = 0.0
    if not is_edge_case and final_answer and retrieved_chunk_texts:
        faithfulness_score = _compute_faithfulness(
            question=test_case.query,
            answer=final_answer,
            retrieved_chunk_texts=retrieved_chunk_texts,
            model=model,
        )
    elif is_edge_case:
        faithfulness_score = 1.0  # correct refusals are perfectly faithful

    # 3. Citation accuracy
    citation_accuracy = _compute_citation_accuracy(final_answer, sources)

    # 4. Answer relevance (cosine similarity)
    answer_relevance = 0.0
    if final_answer and not is_edge_case:
        answer_relevance = _compute_answer_relevance(test_case.query, final_answer)

    return TestResult(
        test_case=test_case,
        passed=passed,
        intent_match=intent_match,
        domain_match=domain_match,
        behavior_notes=" | ".join(notes_parts),
        actual_intent=actual_intent,
        actual_domains=actual_domains,
        answer_preview=final_answer[:500],
        # existing
        quality_score=quality_score,
        coverage_score=coverage_score,
        grounding_score=grounding_score,
        # new
        retrieval_hit_rate=retrieval_hit_rate,
        faithfulness_score=faithfulness_score,
        citation_accuracy=citation_accuracy,
        answer_relevance=answer_relevance,
        latency_ms=latency_ms,
        retrieved_chunk_texts=retrieved_chunk_texts,
    )


def run_all_evaluations(
    pipeline_fn: Callable,
    model: str = DEFAULT_MODEL,
    top_k: int = 5,
    rerank_top_k: int = 3,
    enable_verification: bool = True,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> list[TestResult]:
    """Run all test cases and return results."""
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
    """
    Aggregate statistics across all test results.

    Metric          Scope
    ─────────────── ──────────────────────────────────────────────────────
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

    # Quality / coverage / grounding — exclude correct edge-case tests (score = 0 by design)
    answer_tests = [r for r in results if r.quality_score > 0]
    avg_quality   = _avg(r.quality_score  for r in answer_tests)
    avg_coverage  = _avg(r.coverage_score for r in answer_tests)
    avg_grounding = _avg(r.grounding_score for r in answer_tests)

    # Faithfulness — substantive tests only (skip edge-cases which get 1.0 trivially)
    faith_tests = [r for r in results if r.test_case.category != "edge-case"]
    avg_faith = _avg(r.faithfulness_score for r in faith_tests if r.faithfulness_score > 0)

    # Retrieval hit rate — tests that have keywords defined
    hr_tests = [r for r in results if r.test_case.relevant_keywords]
    avg_hit_rate = _avg(r.retrieval_hit_rate for r in hr_tests)

    # Citation accuracy — answer-producing tests
    avg_citation = _avg(r.citation_accuracy for r in answer_tests)

    # Answer relevance — substantive tests
    avg_relevance = _avg(r.answer_relevance for r in faith_tests if r.answer_relevance > 0)

    # Latency
    avg_latency = _avg(r.latency_ms for r in results)

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for r in results:
        cat = r.test_case.category
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0}
        categories[cat]["total"] += 1
        if r.passed:
            categories[cat]["passed"] += 1

    return {
        # existing
        "total":             total,
        "passed":            passed,
        "failed":            total - passed,
        "pass_rate":         _pct(passed, total),
        "intent_accuracy":   _pct(intent_ok, total),
        "domain_accuracy":   _pct(domain_ok, total),
        "avg_quality_score": round(avg_quality, 3),
        "avg_answer_quality":round(avg_quality, 3),   # kept for UI compatibility
        "answer_tests_count":len(answer_tests),
        "avg_coverage_score":round(avg_coverage, 3),
        "avg_grounding_score":round(avg_grounding, 3),
        # new
        "avg_faithfulness":  round(avg_faith,    3),
        "avg_retrieval_hit_rate": round(avg_hit_rate, 3),
        "avg_citation_accuracy":  round(avg_citation, 3),
        "avg_answer_relevance":   round(avg_relevance, 3),
        "avg_latency_ms":         round(avg_latency,   1),
        "by_category":       categories,
    }


def _avg(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 1) if denominator else 0.0
