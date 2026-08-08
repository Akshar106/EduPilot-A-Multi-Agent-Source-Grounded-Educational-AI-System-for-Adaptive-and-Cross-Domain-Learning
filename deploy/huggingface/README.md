---
title: EduPilot
emoji: 🎓
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-agent, source-grounded AI tutor for four IU graduate courses
---

# EduPilot

A multi-agent, source-grounded educational AI system. It answers questions
across four Indiana University graduate courses — Applied Machine Learning,
Applied Database Technologies, Statistics, and Large Language Models — by
retrieving from the course materials themselves, citing every claim, and
refusing rather than guessing when the corpus does not cover the question.

Source: https://github.com/Akshar106/EduPilot-A-Multi-Agent-Source-Grounded-Educational-AI-System-for-Adaptive-and-Cross-Domain-Learning

## This Space

This file is the Space's README — Hugging Face reads the frontmatter above to
configure the container. It is deliberately separate from the repository's own
README so the GitHub page is not prefixed with a YAML block.

`deploy/README.md` in the repository explains how to publish and what secrets
the Space needs.

## Notes on this deployment

- **Source PDFs are not shipped.** The corpus is ~191 MB of copyrighted
  lecture material. The vector index lives in Pinecone, so answers and
  citations work normally; only downloading an original PDF from a citation
  does not.
- **Storage.** Without a persistent disk attached, accounts and chat history
  reset whenever the Space restarts. The Pinecone index is unaffected.
- **Cold start.** The models are baked into the image, so the first request
  after a restart costs a model load (a few seconds), not a 1.2 GB download.
