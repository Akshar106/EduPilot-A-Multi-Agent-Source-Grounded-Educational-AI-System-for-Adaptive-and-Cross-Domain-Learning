# Deploying EduPilot

The container targets Hugging Face Spaces (Docker SDK) but is plain Docker, so
it runs unchanged on Fly, Render, or any VPS.

```
Dockerfile              image: CPU torch, models baked in, non-root, port 7860
.dockerignore           keeps data/, .env and the venv out of the build context
deploy/entrypoint.sh    seeds the volume, then execs uvicorn as PID 1
deploy/seed/            state a fresh deployment cannot generate for itself
deploy/huggingface/     the Space's own README (frontmatter config)
```

## What the deployment needs, and why

**~1 GB of RAM, minimum 2 GB.** Measured resident set with both models loaded
and serving:

```
baseline python              15 MB
+ edupilot imported          91 MB
+ embedder (bge-small)      551 MB
+ reranker (bge-reranker)  1065 MB
```

This rules out 512 MB free tiers. Hugging Face's free CPU tier (16 GB) is
comfortable.

**The models are baked into the image.** Together they are ~1.2 GB. Downloading
them on first request means the first visitor waits minutes and every restart
re-downloads. `HF_HOME` is set identically at build and run time; changing one
without the other silently re-downloads everything.

**Seed state.** Two files are required to serve but are derived artifacts that
git ignores, so they ship in `deploy/seed/` and the entrypoint copies them to
the volume when absent:

| File | Without it |
|---|---|
| `index_pointer.json` | no active index — every answer refuses |
| `bm25.json` | hybrid search degrades to dense-only; still works, worse |
| `indexed_documents.json` | Knowledge Base tab lists no documents |

`indexed_documents.json` seeds a database table, not a file, and is read from
`SEED_DIR` at startup. It is inserted only into an empty registry, so a real
ingest is never overwritten. It contains course-corpus rows only — per-student
namespaces are excluded at export.

**The corpus is not shipped.** `data/knowledge_base` is ~191 MB of copyrighted
lecture material. The vectors live in Pinecone, so answers and citations work
normally; only downloading an original PDF from a citation does not, and that
route says so rather than returning a bare 404.

## Access model

The bundled frontend has no login screen, so the image sets
`EDUPILOT_AUTH_REQUIRED=false`. That does **not** mean access control is off:

- Each browser is issued its own anonymous identity in a long-lived cookie.
  One visitor cannot list or open another's conversations.
- Anonymous callers are **students** in production, so uploading to or deleting
  from the shared knowledge base is refused (403). Manage the corpus with
  `edupilot-reindex`.
- The cookie authorizes nothing beyond "these are my chats", so it is not
  signed. Forging one yields a different, empty account.

Set `EDUPILOT_AUTH_REQUIRED=true` once a login UI exists; the backend already
has registration, login, refresh, logout and per-user ownership.

In development the anonymous identity is granted admin, so the operator can use
the Knowledge Base tab locally. That is why production must not grant it.

## CI/CD — GitHub Actions to Fly.io

```
push / PR ──► CI ──► lint · 172 tests · docker build + container smoke test
                 │
     main only   └──► Deploy ──► push image to GHCR ──► flyctl deploy ──► verify /api/health
```

`CI` runs on every push and pull request. `Deploy` triggers on `workflow_run`,
so it waits for CI to conclude and then checks it actually *succeeded* — a
plain `push` trigger would race the test job and could ship a red build.

Fly is deployed by digest (`:${{ github.sha }}`), not `:latest`, so a rollback
is `flyctl deploy --image ghcr.io/<owner>/<repo>:<old-sha>` and a re-run cannot
silently pick up a newer image.

### One-time setup

1. **Create the app and volume.** From the repo root, with
   [flyctl](https://fly.io/docs/flyctl/install/) installed:

   ```bash
   flyctl auth login
   flyctl apps create edupilot              # must match `app` in fly.toml
   flyctl volumes create edupilot_data --size 1 --region ord
   ```

   Without the volume, accounts and chat history reset on every restart. The
   Pinecone index is unaffected either way.

2. **Set the runtime secrets on Fly** (these are for the *running app*):

   ```bash
   flyctl secrets set \
     GROQ_API_KEY=... \
     PINECONE_API_KEY=... \
     JWT_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
     GEMINI_API_KEY=... \
     CORS_ALLOWED_ORIGINS=https://edupilot.fly.dev
   ```

3. **Set one secret on GitHub** (this is for the *pipeline*):

   Settings → Secrets and variables → Actions → New repository secret

   | Secret | Value |
   |---|---|
   | `FLY_API_TOKEN` | `flyctl tokens create deploy -x 999999h` |

   GHCR needs no secret — `GITHUB_TOKEN` is provided automatically.

4. **Push to main.** CI runs, then Deploy publishes and rolls out.

### What CI proves

The Docker job does not merely build the image — it runs the container and
curls it, because a build that succeeds and a container that serves are
different claims. It boots with no provider keys, so health reports `degraded`;
that is expected. What is asserted is that the container starts, seeds its
volume, and answers on the HTTP port.

The test job needs no secrets at all: `conftest.py` pins `EDUPILOT_DATA_DIR` at
a temp directory and nothing in the suite calls a provider. If a test ever
needs an API key, that test is reaching the network and should be rewritten.

### Cost and cold starts

`fly.toml` sets `min_machines_running = 0`, so the machine sleeps when idle and
an idle demo costs almost nothing. The tradeoff is that the first request after
a sleep pays roughly 20 seconds loading 1.1 GB of models. Before a live demo,
either warm it with one request or set `min_machines_running = 1`.

## Hugging Face Spaces (alternative)

1. Create a Space — **SDK: Docker**, hardware **CPU basic** (free).

2. Copy the Space README so Hugging Face can configure the container:

   ```bash
   cp deploy/huggingface/README.md README.md   # in the Space clone only
   ```

   It is kept separate so the GitHub README is not prefixed with a YAML block.

3. Add these under **Settings → Secrets**:

   | Secret | Required | Notes |
   |---|---|---|
   | `GROQ_API_KEY` | yes | default model is Groq |
   | `PINECONE_API_KEY` | yes | holds the vector index |
   | `JWT_SECRET_KEY` | yes | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
   | `GEMINI_API_KEY` | no | fallback when Groq is rate limited |
   | `BOOTSTRAP_ADMIN_EMAIL` | no | creates an admin account on first boot |
   | `BOOTSTRAP_ADMIN_PASSWORD` | no | needed with the above |
   | `CORS_ALLOWED_ORIGINS` | no | set to the Space URL |

4. Push:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space>
   git push space main
   ```

5. **Persistent storage** (Settings → Persistent storage) mounts at `/data`,
   which is already `EDUPILOT_DATA_DIR`. Without it, accounts and chat history
   reset on every restart; the Pinecone index is unaffected either way.

## Running the image directly

```bash
docker build -t edupilot .
docker run --rm -p 7860:7860 \
  -e GROQ_API_KEY=... -e PINECONE_API_KEY=... -e JWT_SECRET_KEY=... \
  -v edupilot-data:/data \
  edupilot
```

## Rehearsing the boot path without Docker

The entrypoint honours `PYTHON`, so the exact production startup can be
exercised against a virtualenv — useful for checking seeding and migrations
before a build:

```bash
EDUPILOT_DATA_DIR=/tmp/vol EDUPILOT_ENV=production EDUPILOT_AUTH_REQUIRED=false \
EDUPILOT_STATE_DIR=/tmp/vol/state SQLITE_DB_PATH=/tmp/vol/edupilot.db \
SEED_DIR="$PWD/deploy/seed" PORT=8000 PYTHONPATH="$PWD/src" \
PYTHON="$PWD/.venv/bin/python" ./deploy/entrypoint.sh
```

Verified this way on an empty volume: schema migrates to v3, the registry seeds
54 documents, health reports `ok` with `index=edupilot-v1`, two browsers get
distinct identities, knowledge-base writes return 403, one visitor cannot read
another's session (404), and a chat answer returns 545 words with 8 sources, 22
citations and grounding 0.95 at ~1.1 GB RSS.

## Updating the index

The deployment reads Pinecone; it does not index. To change the corpus, run a
rebuild where the source documents live and let the deployment pick it up:

```bash
edupilot-reindex --rebuild        # builds and promotes a new index version
cp data/state/{bm25.json,index_pointer.json} deploy/seed/
```

Then redeploy. Blue/green means the running deployment keeps serving the old
version until the new one is promoted.
