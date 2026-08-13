# EduPilot — container image
# =============================================================================
# Targets Hugging Face Spaces (Docker SDK) but is plain Docker, so it runs
# unchanged on Fly, Render, or any VPS.
#
# Two decisions dominate this file:
#
#   1. The model weights are baked in. bge-small-en-v1.5 and bge-reranker-base
#      are ~1.2 GB together, and downloading them on first request means the
#      first visitor waits minutes and a restart re-downloads. Fetching at
#      build time trades image size for a predictable cold start.
#
#   2. CPU-only torch. The default wheel pulls the entire CUDA stack — several
#      GB of libraries that cannot be used on a CPU host.
#
# Peak resident memory serving a query is ~1.1 GB with both models loaded, so
# give the container at least 2 GB.
# =============================================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Set before the tokenizers Rust extension initializes. The chunker and the
    # embedder both touch a tokenizer in one process, and re-entering rayon's
    # pool from a second call site deadlocks — see core/config.py.
    TOKENIZERS_PARALLELISM=false \
    # Where the baked weights live. HF_HOME must match at build and run time or
    # the runtime re-downloads everything.
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

# tesseract: OCR fallback for scanned PDF pages (ingestion only, but the
#            import path touches it).
# libgl1/libglib: PyMuPDF and Pillow runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependencies
#
# CPU torch first and pinned to the CPU index, so the resolver never considers
# the CUDA build. Copying only the packaging metadata keeps this layer cached
# across source edits.
# ---------------------------------------------------------------------------
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src/edupilot/__init__.py src/edupilot/__init__.py
# No `pip cache purge` here: PIP_NO_CACHE_DIR=1 above already prevents a
# cache from existing, and purging a disabled cache exits 1 —
# "pip cache commands can not function since cache is disabled" — which
# fails the layer even though the install succeeded.
RUN pip install -e .

# ---------------------------------------------------------------------------
# Bake the models
#
# Runs before the source copy so editing code does not re-download 1.2 GB.
# ---------------------------------------------------------------------------
RUN python - <<'PY'
from sentence_transformers import CrossEncoder, SentenceTransformer

# Same ids as core/config.py: EMBEDDING_MODEL and RERANKER_MODEL.
SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("BAAI/bge-reranker-base")
print("models cached under /opt/models")
PY

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
COPY src/ src/
COPY deploy/ deploy/

# HF Spaces runs the container as uid 1000 and mounts persistent storage at
# /data owned by that user.
RUN useradd -m -u 1000 edupilot \
    && mkdir -p /data \
    && chown -R edupilot:edupilot /app /data /opt/models
USER edupilot

# Open mode: the bundled frontend has no login screen, so sign-in is off and
# each browser gets its own anonymous identity instead. Anonymous callers are
# students in production, so knowledge-base writes are refused — manage the
# corpus with `edupilot-reindex`, or set EDUPILOT_AUTH_REQUIRED=true once a
# login UI exists.
ENV EDUPILOT_DATA_DIR=/data \
    EDUPILOT_ENV=production \
    EDUPILOT_AUTH_REQUIRED=false \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
