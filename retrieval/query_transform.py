"""
Query transformation
====================
Rewrites a student's question into a set of retrieval probes before search.

The previous pipeline embedded the raw question verbatim and searched once.
That loses recall in three specific ways for this corpus:

  * **Acronym mismatch.** A student writes "what is SGD"; the lecture slide
    says "stochastic gradient descent". A 384-dimensional embedding does not
    reliably bridge that, and BM25 certainly does not. Expansion is
    deterministic and costs nothing.
  * **Vocabulary mismatch.** "How do I stop my model memorising the training
    set" and "regularization" are the same question in different words.
    Paraphrase variants cover more of the embedding neighbourhood.
  * **Question/passage asymmetry.** Questions and lecture prose live in
    different regions of embedding space. HyDE sidesteps this by embedding a
    hypothetical *answer* instead of the question.

The deterministic step always runs. The two LLM-backed steps are opt-in per
call because each adds a model round-trip.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

LLMCallable = Callable[[str, str], str]
"""(system_prompt, user_prompt) -> completion text."""


# ---------------------------------------------------------------------------
# Deterministic expansion
# ---------------------------------------------------------------------------

#: Course acronyms mapped to their expansions. Both forms are kept in the
#: probe so lexical and semantic matching each get what they need.
ACRONYMS: dict[str, str] = {
    # Machine learning
    "sgd": "stochastic gradient descent",
    "gd": "gradient descent",
    "mse": "mean squared error",
    "mae": "mean absolute error",
    "rmse": "root mean squared error",
    "svm": "support vector machine",
    "knn": "k nearest neighbors",
    "pca": "principal component analysis",
    "lda": "linear discriminant analysis",
    "em": "expectation maximization",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short term memory",
    "gru": "gated recurrent unit",
    "gan": "generative adversarial network",
    "vae": "variational autoencoder",
    "mlp": "multilayer perceptron",
    "relu": "rectified linear unit",
    "bn": "batch normalization",
    "auc": "area under the curve",
    "roc": "receiver operating characteristic",
    "cv": "cross validation",
    "eda": "exploratory data analysis",
    # Statistics
    "clt": "central limit theorem",
    "ci": "confidence interval",
    "anova": "analysis of variance",
    "mle": "maximum likelihood estimation",
    "map": "maximum a posteriori",
    "pdf": "probability density function",
    "cdf": "cumulative distribution function",
    "pmf": "probability mass function",
    "iid": "independent and identically distributed",
    "ols": "ordinary least squares",
    "hypothesis test": "hypothesis testing significance test",
    # Databases
    "acid": "atomicity consistency isolation durability",
    "bcnf": "boyce codd normal form",
    "1nf": "first normal form",
    "2nf": "second normal form",
    "3nf": "third normal form",
    "er": "entity relationship",
    "dbms": "database management system",
    "rdbms": "relational database management system",
    "oltp": "online transaction processing",
    "olap": "online analytical processing",
    "cte": "common table expression",
    "nl2sql": "natural language to SQL",
    # LLMs
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "rlhf": "reinforcement learning from human feedback",
    "dpo": "direct preference optimization",
    "ppo": "proximal policy optimization",
    "sft": "supervised fine tuning",
    "lora": "low rank adaptation",
    "qlora": "quantized low rank adaptation",
    "peft": "parameter efficient fine tuning",
    "cot": "chain of thought",
    "moe": "mixture of experts",
    "kv cache": "key value cache",
    "mha": "multi head attention",
    "ffn": "feed forward network",
    "bpe": "byte pair encoding",
    "nlp": "natural language processing",
    "icl": "in context learning",
}

#: Phrasings students use that do not appear in lecture text.
PARAPHRASE_HINTS: dict[str, str] = {
    "memoriz": "overfitting",
    "memoris": "overfitting",
    "too simple": "underfitting bias",
    "too complex": "overfitting variance",
    "speed up training": "optimization convergence learning rate",
    "pick the best model": "model selection validation",
    "how sure": "confidence interval uncertainty",
    "make it faster": "efficiency optimization complexity",
    "stop it from": "regularization constraint",
    "makes stuff up": "hallucination grounding",
    "made up": "hallucination",
}

_WORD = re.compile(r"\b[\w-]+\b")


def expand_acronyms(query: str) -> str:
    """
    Append expansions for any acronyms or known paraphrases in the query.

    Additive rather than substitutive: the original wording is preserved so a
    student who wrote "SGD" still matches slides that also write "SGD".
    Returns the query unchanged when nothing matches.
    """
    lowered = query.lower()
    additions: list[str] = []

    for token in {m.group(0).lower() for m in _WORD.finditer(query)}:
        expansion = ACRONYMS.get(token)
        if expansion and expansion not in lowered:
            additions.append(expansion)

    # Multi-word keys need a substring scan.
    for key, expansion in ACRONYMS.items():
        if " " in key and key in lowered and expansion not in lowered:
            additions.append(expansion)

    for hint, expansion in PARAPHRASE_HINTS.items():
        if hint in lowered and expansion not in lowered:
            additions.append(expansion)

    if not additions:
        return query
    logger.debug("expanded query with: %s", additions)
    return f"{query} {' '.join(dict.fromkeys(additions))}"


# ---------------------------------------------------------------------------
# LLM-backed transformations
# ---------------------------------------------------------------------------

MULTI_QUERY_SYSTEM = """\
You rewrite a student's question into alternative search queries for a \
course-material retrieval system.

Rules:
- Produce queries that would match lecture slides and textbook prose, not \
conversational phrasing.
- Vary the vocabulary: use the technical term where the student used a casual \
one, and vice versa.
- Preserve the original meaning exactly. Never broaden to a related topic.
- Output one query per line. No numbering, no commentary, no blank lines.\
"""

HYDE_SYSTEM = """\
You write a short, factual passage that would plausibly appear in university \
lecture notes answering the given question.

Rules:
- 2 to 4 sentences. Dense with the technical terms an actual lecture would use.
- Write it as expository course prose, not as an answer to a person.
- Accuracy is secondary to vocabulary: this text is used only to search for \
real course material, and is never shown to anyone.\
"""


@dataclass
class TransformedQuery:
    """The set of probes derived from one student question."""

    original: str
    expanded: str
    variants: list[str] = field(default_factory=list)
    hyde: str | None = None

    @property
    def probes(self) -> list[str]:
        """
        Every distinct string to run a search for.

        The expanded original always comes first so it can be weighted highest
        during fusion.
        """
        seen: list[str] = []
        for candidate in [self.expanded, *self.variants, self.hyde]:
            if candidate and candidate.strip() and candidate not in seen:
                seen.append(candidate)
        return seen


def generate_variants(query: str, llm: LLMCallable, *, n: int = 3) -> list[str]:
    """
    Ask an LLM for `n` paraphrases of the query.

    Returns [] on any failure — retrieval must still work when the model is
    rate-limited, so this never raises.
    """
    prompt = f"Question: {query}\n\nWrite {n} alternative search queries, one per line."
    try:
        raw = llm(MULTI_QUERY_SYSTEM, prompt)
    except Exception as exc:
        logger.warning("multi-query generation failed: %s", exc)
        return []

    variants: list[str] = []
    for line in (raw or "").split("\n"):
        cleaned = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip().strip('"')
        # Guard against the model returning prose instead of queries.
        if 3 <= len(cleaned.split()) <= 40 and cleaned.lower() != query.lower():
            variants.append(cleaned)
    return variants[:n]


def generate_hyde(query: str, llm: LLMCallable) -> str | None:
    """
    Generate a hypothetical answer passage to embed in place of the question.

    The passage is a retrieval probe only. It is never shown to the student
    and never enters the answering context, so its factual accuracy does not
    affect grounding — only which real chunks come back.
    """
    try:
        raw = llm(HYDE_SYSTEM, f"Question: {query}\n\nWrite the passage.")
    except Exception as exc:
        logger.warning("HyDE generation failed: %s", exc)
        return None

    text = (raw or "").strip()
    if len(text.split()) < 8:
        return None
    return text


def transform_query(
    query: str,
    *,
    llm: LLMCallable | None = None,
    use_multi_query: bool = False,
    use_hyde: bool = False,
    n_variants: int = 3,
) -> TransformedQuery:
    """
    Build the probe set for a query.

    Acronym expansion always runs — it is deterministic and free. The two
    LLM-backed steps are off by default because each costs a round-trip;
    enable them for hard or low-recall queries.
    """
    expanded = expand_acronyms(query)
    result = TransformedQuery(original=query, expanded=expanded)

    if llm is None:
        return result
    if use_multi_query:
        result.variants = generate_variants(query, llm, n=n_variants)
    if use_hyde:
        result.hyde = generate_hyde(query, llm)
    return result
