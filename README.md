# EduPilot — Educational Multi-Agent RAG System

A production-quality multi-agent Retrieval-Augmented Generation (RAG) system for answering questions across three university course domains: Applied Machine Learning (AML), Applied Database Technologies (ADT), and Statistics (STAT).

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────┐
│   Router    │  ← Intent classification + domain detection (LLM + keyword fallback)
└─────────────┘
    │
    ▼
┌──────────────────┐
│  Query Splitter  │  ← Decomposes multi-intent queries into sub-questions per domain
└──────────────────┘
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
┌─────────────────────┐        ┌─────────────────────┐
│  Domain Retriever   │        │  Domain Retriever   │  (one per domain, fully isolated)
│  (AML/ADT/STAT)     │        │  (AML/ADT/STAT)     │
│  ┌───────────────┐  │        │  ┌───────────────┐  │
│  │ ChromaDB      │  │        │  │ ChromaDB      │  │
│  │ (semantic)    │  │        │  │ (semantic)    │  │
│  └───────────────┘  │        └──┴───────────────┘  │
│  ┌───────────────┐  │                               │
│  │ BM25 Index    │  │  RRF Fusion (Hybrid Search)   │
│  │ (keyword)     │  │                               │
│  └───────────────┘  │                               │
└─────────────────────┘                               │
    │                                                 │
    ▼                                                 ▼
┌──────────────┐                           ┌──────────────┐
│  Reranker    │                           │  Reranker    │
└──────────────┘                           └──────────────┘
    │                                                 │
    ▼                                                 ▼
┌──────────────────────┐              ┌──────────────────────┐
│  Domain Agent        │              │  Domain Agent        │
│  (grounded answer)   │              │  (grounded answer)   │
└──────────────────────┘              └──────────────────────┘
    │                                                 │
    └────────────────────┬────────────────────────────┘
                         ▼
               ┌──────────────────┐
               │   Synthesizer    │  ← Cross-domain fusion (multi-domain queries)
               └──────────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │    Verifier      │  ← Second LLM pass: coverage + grounding check
               └──────────────────┘
                         │
                         ▼
               Final Answer + Citations
```

## Project Structure

```
edupilot/
├── app.py                    # Streamlit UI (main entry point)
├── config.py                 # Central configuration (domains, models, hyperparameters)
├── prompts.py                # All LLM prompt templates
├── utils.py                  # Data models, LLM caller, document loading, chunking
├── retriever.py              # DomainRetriever (ChromaDB + BM25 + RRF fusion)
├── reranker.py               # Keyword overlap + optional cross-encoder reranking
├── router.py                 # Intent classification and domain detection
├── query_splitter.py         # Multi-intent query decomposition
├── synthesizer.py            # Domain answer generation + cross-domain synthesis
├── verifier.py               # Answer verification and refinement
├── evaluation.py             # 10 built-in test cases + evaluation runner
├── requirements.txt          # Python dependencies
├── knowledge_base/
│   ├── aml/
│   │   ├── ml_fundamentals.md
│   │   └── ml_algorithms.md
│   ├── adt/
│   │   ├── database_fundamentals.md
│   │   └── advanced_databases.md
│   └── stats/
│       ├── statistics_fundamentals.md
│       └── hypothesis_testing.md
└── vector_stores/            # Auto-created on first run
    ├── aml/
    ├── adt/
    └── stats/
```

## Setup and Installation

### 1. Prerequisites

- Python 3.10 or later
- An Anthropic API key ([get one here](https://console.anthropic.com))

### 2. Create and activate a virtual environment

```bash
cd /Users/khushishah/Documents/Projects/ML/edupilot
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create a `.env` file (requires `python-dotenv` if you add auto-loading):

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. On first launch, the knowledge base is automatically indexed into ChromaDB — this takes ~15 seconds.

## Key Features

| Feature | Description |
|---------|-------------|
| **Separate RAG per domain** | AML, ADT, and STAT each have isolated ChromaDB collections + BM25 indexes |
| **Hybrid retrieval** | Cosine semantic search + BM25 keyword search fused via Reciprocal Rank Fusion |
| **Adaptive thinking** | Claude uses `thinking: {type: "adaptive"}` for complex reasoning tasks |
| **Cross-domain synthesis** | Multi-domain queries get answers from each domain then merged coherently |
| **Answer verification** | A second LLM pass checks coverage, grounding, and completeness |
| **Citations** | All answers include `[Source N: file.md, chunk C]` references |
| **Debug panel** | Full pipeline trace: routing, retrieval scores, reranking, verification |
| **Document upload** | Upload PDF/DOCX/MD/TXT files to extend any domain's knowledge base |
| **Built-in evaluation** | 10 test cases covering single-domain, multi-domain, OOD, hallucination stress |

## Using the App

### Chat

Type a question in the chat box. Sample prompts:

- "Explain the bias-variance tradeoff" → AML domain
- "What are the normal forms in database normalization?" → ADT domain
- "What is a p-value and how do I interpret it?" → STAT domain
- "How does NL2SQL work and what ML techniques does it use?" → ADT + AML
- "What are the assumptions of linear regression from both a statistical and ML perspective?" → STAT + AML

### Sidebar Controls

- **Model**: Choose Claude model (default: `claude-opus-4-6`)
- **Top-K Retrieval**: Number of chunks retrieved per domain (default: 8)
- **Rerank Top-K**: Chunks kept after reranking (default: 4)
- **Confidence Threshold**: Minimum relevance score to include a chunk
- **Reranking Mode**: Keyword overlap (fast) or cross-encoder (requires `sentence-transformers`)
- **Answer Verification**: Enable/disable second-pass verification
- **Debug Mode**: Show full pipeline trace in the UI

### Evaluation Tab

Run the 10 built-in test cases to assess system quality. Results include intent match, domain match, behavior notes, and quality scores.

### Knowledge Base Tab

View indexed chunk counts per domain and add new documents via the sidebar upload widget.

## Adding a New Domain

EduPilot is designed so adding a new course domain requires changes in only **two places**:

### Step 1: Add knowledge base documents

```bash
mkdir -p knowledge_base/newdomain/
# Add .md, .txt, .pdf, or .docx files here
```

### Step 2: Register the domain in `config.py`

```python
DOMAINS: dict[str, dict] = {
    # ... existing domains ...
    "NEW": {
        "name": "New Course Name",
        "abbr": "NEW",
        "color": "#9C27B0",           # Any hex color for UI badges
        "knowledge_base_path": str(KNOWLEDGE_BASE_DIR / "newdomain"),
        "vector_store_path": str(VECTOR_STORE_DIR / "newdomain"),
        "collection_name": "newdomain_docs",
        "keywords": ["keyword1", "keyword2", "..."],  # For keyword-based domain detection
        "description": "One-sentence description of the course",
    },
}
```

### Step 3: Restart the app

The new domain's vector store will be automatically created and indexed from the knowledge base files on startup.

No changes to `retriever.py`, `router.py`, `synthesizer.py`, or any other module are needed.

## Pipeline Step-by-Step

| Step | Module | Description |
|------|--------|-------------|
| 1. Routing | `router.py` | LLM classifies intent (single/multi/clarify/OOD) and detects domains |
| 2. Query splitting | `query_splitter.py` | Multi-intent queries split into per-domain sub-questions |
| 3. Hybrid retrieval | `retriever.py` | ChromaDB semantic + BM25 keyword search, RRF fusion |
| 4. Reranking | `reranker.py` | Keyword overlap scoring (or cross-encoder) |
| 5. Domain generation | `synthesizer.py` | Per-domain grounded answer with citations |
| 6. Cross-domain synthesis | `synthesizer.py` | Merges domain answers (skipped for single-domain) |
| 7. Verification | `verifier.py` | Checks coverage, grounding, revises if needed |

## Configuration Reference

Key parameters in `config.py`:

```python
SEMANTIC_WEIGHT = 0.60       # Weight for ChromaDB semantic scores in RRF fusion
BM25_WEIGHT = 0.40           # Weight for BM25 keyword scores in RRF fusion
DEFAULT_TOP_K = 8            # Chunks retrieved per domain
DEFAULT_RERANK_K = 4         # Chunks kept after reranking
CONFIDENCE_THRESHOLD = 0.3   # Minimum hybrid score to include a chunk
CHUNK_SIZE = 400             # Target words per chunk
CHUNK_OVERLAP = 80           # Overlapping words between chunks
DEFAULT_MODEL = "claude-opus-4-6"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

## Troubleshooting

**App won't start / import errors:**
```bash
pip install -r requirements.txt --upgrade
```

**`ANTHROPIC_API_KEY` not found:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

**Vector store reset (if index is corrupted):**
```bash
rm -rf vector_stores/
# Restart the app — will re-index automatically
```

**Cross-encoder reranking unavailable:**
The app falls back to keyword reranking automatically. To enable cross-encoder:
```bash
pip install sentence-transformers
```
Then select "Cross-Encoder" in the sidebar.

**Slow first startup:**
The `all-MiniLM-L6-v2` embedding model is downloaded on first run (~90MB). Subsequent starts use the cached model.
