"""
EduPilot Configuration
======================
Central configuration for all modules. Loads secrets from .env.
Edit DOMAINS here to add new course domains — no other code changes needed.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Must be set before the HuggingFace `tokenizers` Rust extension initializes.
#
# This pipeline touches the tokenizer from two places in one process: the
# chunker counts tokens to size chunks, and the embedder tokenizes to encode.
# `tokenizers` parallelizes with a rayon thread pool, and re-entering that pool
# from a second call site deadlocks — the process parks in `__psynch_cvwait`
# with zero CPU and never recovers. Observed here as a rebuild that hung
# indefinitely after loading the embedding model.
#
# The lost parallelism is irrelevant: encoding is already batched, and the
# batch loop is where the real throughput comes from.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Load .env (python-dotenv — safe no-op if file is absent)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    # Walk up from this file to the repo root looking for .env. The extra
    # levels versus the flat layout are src/ and edupilot/.
    _here = Path(__file__).resolve().parent
    for _candidate in [_here, *_here.parents[:4]]:
        if (_candidate / ".env").exists():
            load_dotenv(_candidate / ".env")
            break
except ImportError:
    pass  # dotenv not installed; rely on shell environment

# ---------------------------------------------------------------------------
# Project paths
#
# Code and data are separated: PACKAGE_DIR holds importable Python and the
# bundled web assets, DATA_DIR holds everything the application reads or
# writes at runtime. Only DATA_DIR needs to be writable, and it is the only
# path that needs mounting when this runs in a container.
# ---------------------------------------------------------------------------

#: The installed package — src/edupilot. Never written to.
PACKAGE_DIR: Path = Path(__file__).resolve().parent.parent

#: Repo root. Correct for an editable/source checkout; irrelevant once
#: EDUPILOT_DATA_DIR is set, which is how a real deployment should run.
PROJECT_ROOT: Path = Path(os.getenv("EDUPILOT_PROJECT_ROOT", str(PACKAGE_DIR.parent.parent)))

#: Everything mutable: the corpus, student uploads, the database, local caches.
DATA_DIR: Path = Path(os.getenv("EDUPILOT_DATA_DIR", str(PROJECT_ROOT / "data")))

KNOWLEDGE_BASE_DIR: Path = DATA_DIR / "knowledge_base"
SELF_STUDY_DIR: Path = DATA_DIR / "self_study_files"

#: Bundled single-page frontend, served at /static.
STATIC_DIR: Path = PACKAGE_DIR / "web" / "static"

# ---------------------------------------------------------------------------
# API keys (read from environment / .env)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str      = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY: str    = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY: str  = os.getenv("PINECONE_API_KEY", "")

# ---------------------------------------------------------------------------
# Pinecone settings
# GCP us-central1 = Pinecone free-tier region; compatible with Streamlit Cloud
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "edupilot")
PINECONE_CLOUD: str      = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str     = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIMENSION: int = 384        # bge-small-en-v1.5 output dimension

# Versioned index base name. Concrete indexes are "<base>-v1", "<base>-v2", ...
# and INDEX_POINTER_PATH records which one serves traffic. A rebuild writes to
# a new version and promotes it only on success (blue/green), so a failed
# re-index can never leave the live index half-populated.
PINECONE_INDEX_BASE: str = os.getenv("PINECONE_INDEX_BASE", "edupilot")

# Native sparse-dense hybrid search requires the dotproduct metric. Because all
# embeddings are L2-normalized, the dot product equals cosine similarity, so
# dense ranking is identical to the previous cosine index.
PINECONE_METRIC: str = os.getenv("PINECONE_METRIC", "dotproduct")

# ---------------------------------------------------------------------------
# SQLite database path
# ---------------------------------------------------------------------------
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "edupilot.db"))

# ---------------------------------------------------------------------------
# Local state (caches and index metadata — safe to delete, rebuilt on demand)
# ---------------------------------------------------------------------------
STATE_DIR: Path = Path(os.getenv("EDUPILOT_STATE_DIR", str(DATA_DIR / "state")))

#: Embedding cache, keyed by (model, content hash). Makes re-ingesting an
#: unchanged document nearly free.
EMBEDDING_CACHE_PATH: str = str(STATE_DIR / "embeddings.db")

#: Fitted BM25 IDF table. Only the table persists, never the corpus.
SPARSE_ENCODER_PATH: str = str(STATE_DIR / "bm25.json")

#: Points at the index version currently serving traffic.
INDEX_POINTER_PATH: str = str(STATE_DIR / "index_pointer.json")

#: Frequently-asked answer cache. Only questions asked repeatedly are stored,
#: and every entry is bound to the index version that produced it.
ANSWER_CACHE_PATH: str = str(STATE_DIR / "answer_cache.db")

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
ENV: str = os.getenv("EDUPILOT_ENV", "development").lower()
IS_PRODUCTION: bool = ENV == "production"

# ---------------------------------------------------------------------------
# Open mode
#
# With auth disabled there is no sign-in. Access control is not removed: each
# browser is issued its own anonymous identity (security/deps.anonymous_user),
# sessions are still written with an owner, and every ownership check still
# runs. What goes away is only the login step.
#
# This was previously refused outright in production, because "auth off" then
# meant every caller shared one *admin* identity — which let any visitor upload
# into the shared knowledge base and read everyone else's conversations. Both
# are now addressed directly: identities are per-browser, and anonymous callers
# are granted the admin role only outside production. So the mode is allowed in
# production, and the role boundary carries the safety instead of a blanket ban.
#
# Requiring sign-in is still the stronger posture and remains the production
# default; turn it off deliberately when the deployment has no login UI.
# ---------------------------------------------------------------------------
AUTH_REQUIRED: bool = (
    os.getenv("EDUPILOT_AUTH_REQUIRED", "true" if IS_PRODUCTION else "false").lower()
    == "true"
)

# Signing key for JWTs. Required in production; security/auth.py raises at
# startup rather than falling back to a default, because a predictable signing
# key lets anyone mint an admin token.
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")

# Allowed browser origins. The previous config used allow_origins=["*"], which
# lets any website call the API with the user's cookies attached.
_origins_raw = os.getenv("CORS_ALLOWED_ORIGINS", "")
CORS_ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _origins_raw.split(",") if o.strip()]
    if _origins_raw
    else ["http://localhost:8000", "http://127.0.0.1:8000"]
)

# Only honour X-Forwarded-For when actually behind a trusted reverse proxy.
# The header is client-controlled, so trusting it otherwise lets a caller
# forge a new identity per request and bypass rate limiting entirely.
TRUST_PROXY_HEADERS: bool = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"

# Longest accepted student question. Unbounded input is unbounded prompt cost.
MAX_QUERY_CHARS: int = int(os.getenv("MAX_QUERY_CHARS", "4000"))

# Bootstrap admin, created on first startup when no users exist. Without this
# there is no way to reach the admin-only knowledge-base routes.
BOOTSTRAP_ADMIN_EMAIL: str = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "")
BOOTSTRAP_ADMIN_PASSWORD: str = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")

# ---------------------------------------------------------------------------
# Domain registry
# Each domain has its own isolated RAG pipeline (separate Pinecone namespace).
# To add a new domain, add an entry here — no other core code changes needed.
# ---------------------------------------------------------------------------
DOMAINS: dict[str, dict] = {
    "AML": {
        "name": "Applied Machine Learning",
        "abbr": "AML",
        "color": "#4CAF50",
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR / "aml"),
        "pinecone_namespace": "aml",
        "description": (
            "Machine learning algorithms, supervised/unsupervised learning, "
            "bias-variance tradeoff, overfitting, regularization, neural networks, "
            "model evaluation, feature engineering, and deep learning."
        ),
        "keywords": [
            "machine learning", "ML", "neural network", "deep learning",
            "bias", "variance", "overfitting", "underfitting", "regularization",
            "gradient descent", "classification", "regression", "clustering",
            "random forest", "SVM", "cross-validation", "feature engineering",
        ],
    },
    "ADT": {
        "name": "Applied Database Technologies",
        "abbr": "ADT",
        "color": "#2196F3",
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR / "adt"),
        "pinecone_namespace": "adt",
        "description": (
            "SQL, relational databases, normalization (1NF–BCNF), transactions, "
            "ACID properties, indexing, NoSQL databases, NL2SQL, query optimization, "
            "and entity-relationship modeling."
        ),
        "keywords": [
            "database", "SQL", "NoSQL", "normalization", "1NF", "2NF", "3NF",
            "BCNF", "transaction", "ACID", "index", "query", "join", "NL2SQL",
            "relational", "schema", "ER diagram", "stored procedure", "trigger",
        ],
    },
    "STAT": {
        "name": "Statistics",
        "abbr": "STAT",
        "color": "#FF9800",
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR / "stats"),
        "pinecone_namespace": "stat",
        "description": (
            "Descriptive statistics, probability distributions, hypothesis testing, "
            "p-values, confidence intervals, t-tests, ANOVA, regression analysis, "
            "Bayesian statistics, and the central limit theorem."
        ),
        "keywords": [
            "statistics", "probability", "distribution", "hypothesis", "p-value",
            "confidence interval", "t-test", "ANOVA", "regression", "correlation",
            "normal distribution", "Bayesian", "central limit theorem", "variance",
            "standard deviation", "mean", "median", "mode", "chi-square",
        ],
    },
    "LLM": {
        "name": "Large Language Models",
        "abbr": "LLM",
        "color": "#9C27B0",
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR / "llm"),
        "pinecone_namespace": "llm",
        "description": (
            "Transformer architecture, attention mechanisms, pretraining, instruction "
            "tuning, RLHF, DPO, LoRA, prompting techniques, RAG pipelines, LLM agents, "
            "hallucination, quantization, and LLM evaluation."
        ),
        "keywords": [
            "LLM", "large language model", "transformer", "attention", "GPT", "BERT",
            "LLaMA", "Mistral", "fine-tuning", "LoRA", "QLoRA", "RLHF", "DPO",
            "prompt", "chain of thought", "RAG", "retrieval augmented generation",
            "agent", "hallucination", "tokenization", "embedding", "pretraining",
            "instruction tuning", "in-context learning", "few-shot", "zero-shot",
            "quantization", "KV cache", "context window", "perplexity",
        ],
    },
}

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
# Evidence budget. DEFAULT_RERANK_TOP_K is the single strongest lever on
# answer depth: it is how many chunks the generator actually sees. At 3-5 the
# model has too little to be detailed from, and the grounding rules correctly
# stop it inventing the difference — producing short, paraphrase-like answers.
DEFAULT_TOP_K: int = 12          # candidates retrieved before reranking
DEFAULT_RERANK_TOP_K: int = 8    # chunks handed to the generator

# Chunking is now measured in embedding-model TOKENS, not words. The previous
# 800-*word* setting was applied per PDF page, so it almost never bound: 46% of
# chunks still overflowed the 256-token window of the old embedder and were
# silently truncated, discarding 47.9% of the indexed corpus.
CHUNK_MAX_TOKENS: int = 448      # hard ceiling, under the 512-token window
CHUNK_TARGET_TOKENS: int = 320   # preferred size
CHUNK_MIN_TOKENS: int = 64       # below this, merge into a neighbour
CHUNK_OVERLAP_SENTENCES: int = 1
PARENT_MAX_TOKENS: int = 1400    # wider window for small-to-big retrieval

# Hybrid search blend lives with the fusion code that applies it — see
# PROBE_WEIGHTS and RRF_K in retrieval/hybrid.py. Reciprocal Rank Fusion
# weights each *probe* (dense, sparse, HyDE, multi-query), not two fixed
# modalities, so a pair of scalars here could not express it.

# Minimum cross-encoder relevance for a chunk to count as evidence. Below this
# the retriever returns nothing and the agent refuses — answering off
# low-relevance context is how ungrounded answers happen.
#
# The scale is model-specific, so this is calibrated per reranker in
# retrieval/rerank.py rather than fixed here. Leave as None to use the
# calibrated default for the configured RERANKER_MODEL.
DEFAULT_MIN_RELEVANCE: float | None = None

# Query transformation. Acronym expansion is deterministic and always on; the
# two LLM-backed steps each cost a round-trip.
ENABLE_MULTI_QUERY: bool = os.getenv("ENABLE_MULTI_QUERY", "false").lower() == "true"
ENABLE_HYDE: bool = os.getenv("ENABLE_HYDE", "false").lower() == "true"
ENABLE_PARENT_EXPANSION: bool = True

# Accepted upload types are owned by the ingestion package, which is what
# actually dispatches on them — see ingestion.SUPPORTED_EXTENSIONS.

# ---------------------------------------------------------------------------
# LLM / Embedding models
# ---------------------------------------------------------------------------
# Gemini model IDs  (https://ai.google.dev/gemini-api/docs/models)
DEFAULT_MODEL: str = "llama-3.3-70b-versatile"    # Groq — free, fast, no daily cap issues
VERIFY_MODEL: str  = "llama-3.3-70b-versatile"   # Groq — same large model; small models anchor scores

# bge-small-en-v1.5: 384-dim (matches the previous index dimension) with a
# 512-token window — double MiniLM's 256 — and materially stronger retrieval on
# technical text. It is asymmetric: queries get an instruction prefix,
# passages do not. See retrieval/embeddings.py.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Cross-encoder used to rerank retrieval candidates. Now active on every path;
# previously the Chat pipeline silently used keyword overlap instead.
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# Groq counts reserved output tokens against TPM even if the response is shorter.
# Keep these tight — they are upper bounds, not targets.
LLM_MAX_TOKENS_CLASSIFY: int = 512
LLM_MAX_TOKENS_GENERATE: int = 3500   # domain chat answers — high for quality
LLM_MAX_TOKENS_SS: int       = 4096   # self-study: user wants long, detailed answers
LLM_MAX_TOKENS_SYNTH: int    = 3500   # cross-domain synthesis — high for quality
LLM_MAX_TOKENS_VERIFY: int   = 768    # JSON blob with scores

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
ENABLE_VERIFICATION_DEFAULT: bool = True

# ---------------------------------------------------------------------------
# Available models (for sidebar dropdown)
# Groq models are prefixed with "groq:" to distinguish them from Gemini.
# ---------------------------------------------------------------------------
# gemma2-9b-it was removed: Groq decommissioned it, so every call returned
# 400 and it silently consumed a slot in the fallback chain.
AVAILABLE_MODELS: list[str] = [
    # ── Groq (free tier, fast) ──────────────────────────────────────────────
    "llama-3.3-70b-versatile",    # best quality
    "llama-3.1-8b-instant",       # fastest, higher free quota
    # ── Gemini (Google AI Studio) ───────────────────────────────────────────
    "gemini-2.0-flash",           # 1500 req/day free
    "gemini-2.0-flash-lite",      # highest Gemini free quota
    "gemini-2.5-flash",           # latest — only 20 req/day free
]

# Groq models — used to detect provider in call_llm
GROQ_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
