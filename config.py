"""
EduPilot Configuration
======================
Central configuration for all modules. Loads secrets from .env.
Edit DOMAINS here to add new course domains — no other code changes needed.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env (python-dotenv — safe no-op if file is absent)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed; rely on shell environment

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

# ---------------------------------------------------------------------------
# API keys (read from environment / .env)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str      = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY: str  = os.getenv("PINECONE_API_KEY", "")

# ---------------------------------------------------------------------------
# Pinecone settings
# GCP us-central1 = Pinecone free-tier region; compatible with Streamlit Cloud
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "edupilot")
PINECONE_CLOUD: str      = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str     = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIMENSION: int = 384        # all-MiniLM-L6-v2 output dimension

# ---------------------------------------------------------------------------
# SQLite database path
# ---------------------------------------------------------------------------
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "edupilot.db"))

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
DEFAULT_TOP_K: int = 5
DEFAULT_RERANK_TOP_K: int = 3
CHUNK_SIZE: int = 600
CHUNK_OVERLAP: int = 80

# Hybrid search blend (must sum to 1.0)
SEMANTIC_WEIGHT: float = 0.60
BM25_WEIGHT: float = 0.40

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.20

# ---------------------------------------------------------------------------
# Supported document extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt", ".md", ".docx"]

# ---------------------------------------------------------------------------
# LLM / Embedding models
# ---------------------------------------------------------------------------
# Groq model IDs  (https://console.groq.com/docs/models)
DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

LLM_MAX_TOKENS_CLASSIFY: int = 1024
LLM_MAX_TOKENS_GENERATE: int = 4096
LLM_MAX_TOKENS_SYNTH: int = 6144
LLM_MAX_TOKENS_VERIFY: int = 6144

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
ENABLE_VERIFICATION_DEFAULT: bool = True

# ---------------------------------------------------------------------------
# Available Groq models (for sidebar dropdown)
# ---------------------------------------------------------------------------
AVAILABLE_MODELS: list[str] = [
    "llama-3.3-70b-versatile",    # best quality — 128K context
    "llama-3.1-70b-versatile",    # slightly older, similar quality
    "mixtral-8x7b-32768",         # good for long-context tasks
    "llama-3.1-8b-instant",       # fast / low-latency
    "gemma2-9b-it",               # lightweight Google model
]
