#!/usr/bin/env bash
# Publish the current branch to a Hugging Face Space.
#
#   ./deploy/push-to-space.sh https://huggingface.co/spaces/<user>/<space>
#
# Hugging Face configures a Space from YAML frontmatter at the top of its
# README.md. Putting that in the repository README would prefix the GitHub page
# with a config block, so the Space's README is kept at
# deploy/huggingface/README.md and swapped in on a throwaway `space` branch.
#
# The branch is recreated from your current HEAD every run and force-pushed, so
# it never accumulates history and never needs merging back. Your working
# branch is restored on exit, including if the push fails.

set -euo pipefail

REMOTE_URL="${1:-}"
BRANCH="space"
SPACE_README="deploy/huggingface/README.md"

if [ -z "$REMOTE_URL" ]; then
    echo "usage: $0 https://huggingface.co/spaces/<user>/<space>" >&2
    exit 2
fi

cd "$(git rev-parse --show-toplevel)"

if [ -n "$(git status --porcelain)" ]; then
    echo "error: working tree is dirty. Commit or stash first — this script" >&2
    echo "       switches branches and will not risk your uncommitted work." >&2
    exit 1
fi

if [ ! -f "$SPACE_README" ]; then
    echo "error: $SPACE_README is missing; it carries the Space's frontmatter." >&2
    exit 1
fi

ORIGINAL="$(git rev-parse --abbrev-ref HEAD)"
restore() { git checkout --quiet "$ORIGINAL" 2>/dev/null || true; }
trap restore EXIT

git remote get-url space >/dev/null 2>&1 || git remote add space "$REMOTE_URL"
git remote set-url space "$REMOTE_URL"

echo "building $BRANCH from $ORIGINAL"
git checkout --quiet -B "$BRANCH"
cp "$SPACE_README" README.md
git add README.md
git commit --quiet -m "Space configuration (generated — do not edit on this branch)"

echo "pushing to $REMOTE_URL"
# Force: this branch is regenerated every run and the Space is a deploy target,
# not somewhere work is authored.
git push --force space "$BRANCH:main"

echo
echo "pushed. The Space will now build the Docker image (first build is slow —"
echo "it downloads ~1.2 GB of model weights into the image)."
echo
echo "If you have not already, set these under Settings -> Secrets:"
echo "  GROQ_API_KEY, PINECONE_API_KEY, JWT_SECRET_KEY"
