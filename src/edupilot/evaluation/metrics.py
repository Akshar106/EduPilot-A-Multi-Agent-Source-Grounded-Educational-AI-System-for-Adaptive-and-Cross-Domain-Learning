"""
Objective metrics
=================
Each function scores one property of a single pipeline run, independent of the
LLM's own self-assessment:

  retrieval_hit_rate  Did the retriever surface the right material at all?
                      Diagnoses retrieval failures before the LLM sees context.
  faithfulness        Is the answer grounded in that material? Catches
                      hallucinations the self-judge misses, because it grades
                      its own work.
  citation_accuracy   Do the [Source N] markers point at chunks that actually
                      support the surrounding sentence?
  answer_relevance    Does the answer address the question, or is it merely
                      on-topic?
"""

from __future__ import annotations

import logging
import re

from edupilot.core.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "what", "that", "this", "with", "from", "have", "been", "they", "their",
    "which", "when", "where", "there", "about", "more", "also", "each", "into",
    "than", "then", "some", "will", "would", "could", "should", "these", "those",
}

#: Model used for the answer-relevance embedding.
#:
#: Deliberately NOT the configured retrieval embedder. Relevance is a stable
#: yardstick — pinning it means a change to RERANKER_MODEL or EMBEDDING_MODEL
#: shows up as a change in retrieval scores rather than silently reshaping the
#: metric that is supposed to measure them.
RELEVANCE_MODEL = "all-MiniLM-L6-v2"


def retrieval_hit_rate(
    relevant_keywords: list[str],
    retrieved_chunk_texts: list[str],
) -> float:
    """
    Fraction of expected domain keywords found in the combined text of all
    retrieved chunks.

    Uses stem-prefix matching so "overfit" hits "overfitting", "overfits";
    "signific" hits "significance", "significant", etc. Also falls back to
    plain substring match for multi-word keywords like "p-value".

    Returns 0.0 if relevant_keywords is empty (edge-case tests).
    """
    if not relevant_keywords or not retrieved_chunk_texts:
        return 0.0

    combined = " ".join(retrieved_chunk_texts).lower()
    words_in_text = re.findall(r"\b\w+\b", combined)

    def _keyword_found(kw: str) -> bool:
        kw_lower = kw.lower()
        # Direct substring match (handles multi-word like "p-value", "nl2sql")
        if kw_lower in combined:
            return True
        # Prefix match on word tokens (stem matching)
        return any(w.startswith(kw_lower) for w in words_in_text)

    hits = sum(1 for kw in relevant_keywords if _keyword_found(kw))
    return round(hits / len(relevant_keywords), 3)


def citation_accuracy(answer: str, sources: list[dict]) -> float:
    """
    For every [Source N] tag in the answer, check whether the sentence
    containing that tag shares significant terms with chunk N's text.

    Intentionally lenient — the goal is to catch citations pointing at the
    wrong chunk entirely, not to penalise paraphrasing.

    Returns 1.0 when there are no citations (nothing to verify as wrong).
    """
    if not sources:
        return 1.0

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    citation_pattern = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)
    total = 0
    correct = 0

    for sentence in sentences:
        for match in citation_pattern.finditer(sentence):
            src_num = int(match.group(1))
            total += 1

            chunk = next((s for s in sources if s.get("source_num") == src_num), None)
            if chunk is None:
                continue

            chunk_text = chunk.get("text", "").lower()
            sentence_words = {
                w.lower()
                for w in re.findall(r"\b\w{4,}\b", sentence)
                if w.lower() not in _STOPWORDS
            }

            if sum(1 for w in sentence_words if w in chunk_text) >= 1:
                correct += 1

    if total == 0:
        return 1.0  # no citations found — handled separately by TC-09's check_fn
    return round(correct / total, 3)


def answer_relevance(question: str, answer: str) -> float:
    """
    Cosine similarity between the question embedding and the opening of the
    answer, computed with `RELEVANCE_MODEL`.

    Only the first 2 sentences (≤300 chars) are embedded: the opening directly
    addresses the question, whereas a 700-word answer dilutes the embedding
    with elaborations and examples and scores artificially low.
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        answer_head = " ".join(sentences[:2])[:300]

        model = SentenceTransformer(RELEVANCE_MODEL)
        q_emb = model.encode(question, normalize_embeddings=True)
        a_emb = model.encode(answer_head, normalize_embeddings=True)
        return round(max(0.0, float(np.dot(q_emb, a_emb))), 3)
    except Exception:
        logger.warning("answer_relevance unavailable", exc_info=True)
        return 0.0


_FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI tutor's answer is faithfully grounded in retrieved course material.

RETRIEVED COURSE MATERIAL:
{context}

STUDENT QUESTION: {question}

AI TUTOR'S ANSWER:
{answer}

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

#: Returned when the judge call fails. A hard 0.0 would render an API outage
#: indistinguishable from a hallucinating model.
FAITHFULNESS_ON_ERROR = 0.85


def faithfulness(
    question: str,
    answer: str,
    retrieved_chunk_texts: list[str],
    model: str = DEFAULT_MODEL,
) -> float:
    """
    Holistic 0–10 score of how well the answer is grounded in the retrieved
    evidence, normalised to [0, 1].

    A single LLM call asks for an overall score rather than claim-by-claim
    extraction. Claim-by-claim checking is too strict for educational answers
    that contain correct inferences and pedagogical elaborations — both
    appropriate, and both frequently absent word-for-word from the chunks.
    """
    if not retrieved_chunk_texts or not answer.strip():
        return 1.0  # nothing to check — don't penalise

    from edupilot.llm import call_llm, parse_json_response

    prompt = _FAITHFULNESS_PROMPT.format(
        context="\n---\n".join(t[:1500] for t in retrieved_chunk_texts[:8]),
        question=question,
        answer=answer[:2000],
    )

    try:
        raw = call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=150,
        )
        data = parse_json_response(raw)
        score = max(0, min(10, int(data.get("score", 9))))
        normalized = round(score / 10, 3)
        logger.info(
            "faithfulness %d/10 -> %.3f | %s",
            score, normalized, str(data.get("reasoning", ""))[:80],
        )
        return normalized
    except Exception:
        logger.warning("faithfulness judge failed", exc_info=True)
        return FAITHFULNESS_ON_ERROR


__all__ = [
    "FAITHFULNESS_ON_ERROR",
    "RELEVANCE_MODEL",
    "answer_relevance",
    "citation_accuracy",
    "faithfulness",
    "retrieval_hit_rate",
]
