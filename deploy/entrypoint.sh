#!/usr/bin/env sh
# EduPilot container entrypoint.
#
# Seeds the data volume on first boot, then execs the server.
#
# Two files under state/ are required to serve properly but are derived
# artifacts, so they are shipped in the image rather than the volume:
#
#   index_pointer.json  names the Pinecone index version serving traffic.
#     Without it the app reports "no active index" and every answer refuses.
#   bm25.json           the fitted BM25 IDF table. Without it hybrid search
#     silently degrades to dense-only — it still works, just worse.
#
# They are copied only when absent, so a volume that already has state (or a
# newer one written by `edupilot-reindex`) is never overwritten.

set -eu

DATA_DIR="${EDUPILOT_DATA_DIR:-/data}"
SEED_DIR="${SEED_DIR:-/app/deploy/seed}"
PORT="${PORT:-7860}"

mkdir -p "$DATA_DIR/state" "$DATA_DIR/knowledge_base" "$DATA_DIR/self_study_files"

# Only these belong on the volume. indexed_documents.json also lives in
# SEED_DIR but is read from there directly at startup — it seeds a database
# table, not a file, so copying it here would just leave a stray artifact.
for name in bm25.json index_pointer.json; do
    src="$SEED_DIR/$name"
    dest="$DATA_DIR/state/$name"
    if [ -f "$src" ] && [ ! -f "$dest" ]; then
        cp "$src" "$dest"
        echo "seeded $name"
    fi
done

# Fail loudly rather than serving a broken deployment: without a pointer there
# is no index to query, and every answer would be an unexplained refusal.
if [ ! -f "$DATA_DIR/state/index_pointer.json" ]; then
    echo "WARNING: no index_pointer.json in $DATA_DIR/state." >&2
    echo "         The app will start but report no active index and refuse" >&2
    echo "         every question. Run 'edupilot-reindex --rebuild', or ship" >&2
    echo "         a pointer in deploy/seed/." >&2
fi

echo "EduPilot starting: data=$DATA_DIR port=$PORT env=${EDUPILOT_ENV:-development}"

# exec so uvicorn is PID 1 and receives SIGTERM directly — otherwise the
# platform's stop signal goes to the shell and the server is killed instead of
# shut down, dropping in-flight requests.
#
# PYTHON is overridable so this same script can be run outside the container
# (against a virtualenv) to rehearse the deployment boot path.
exec "${PYTHON:-python}" -m uvicorn edupilot.api.app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --timeout-keep-alive 65
