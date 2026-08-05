"""
Index management CLI
====================
Blue/green rebuild of the vector index, plus a few migration helpers.

    python reindex.py --status              show the active index and counts
    python reindex.py --rebuild             full rebuild into a new version
    python reindex.py --rebuild --dry-run   extract and chunk only, no writes
    python reindex.py --promote edupilot-v2 point traffic at a built version
    python reindex.py --rollback            revert to the previous version
    python reindex.py --claim-sessions me@iu.edu
                                            assign pre-auth sessions to a user

A rebuild writes into a **new** index (`edupilot-v1`, `-v2`, ...) and promotes
it only after every domain has been indexed successfully. The live index keeps
serving throughout, and a failed rebuild changes nothing. This is also the
mechanism for the cosine → dotproduct migration: native sparse-dense hybrid
requires dotproduct, and since embeddings are L2-normalized the dot product
equals cosine, so dense ranking is unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from config import DOMAINS, SPARSE_ENCODER_PATH  # noqa: E402
from observability import configure_logging  # noqa: E402

configure_logging()
logger = logging.getLogger("edupilot.reindex")


def cmd_status() -> int:
    from services import services

    pointer = services.index_pointer
    print(f"active index : {pointer.active or '(none — run --rebuild)'}")
    print(f"version      : v{pointer.version}")
    print(f"embed model  : {services.embedder.model_name} ({services.embedder.dimension}d, "
          f"{services.embedder.max_tokens} tokens)")

    encoder = services.sparse_encoder
    print(f"bm25 encoder : {'fitted, ' + str(len(encoder.idf)) + ' terms' if encoder else 'missing'}")

    if not pointer.active:
        return 1

    try:
        stats = services.store.stats()
        print(f"\nvectors      : {stats['total_vectors']}")
        for namespace, count in sorted(stats.get("namespaces", {}).items()):
            print(f"  {namespace:24s} {count:6d}")
    except Exception as exc:
        print(f"\ncould not read index stats: {exc}")
        return 1

    docs = services.registry.list_documents()
    print(f"\nindexed documents: {len(docs)}")
    return 0


def cmd_rebuild(dry_run: bool, force_promote: bool) -> int:
    from ingestion import chunk_document, extract_document
    from retrieval import BM25Encoder, Indexer
    from services import services

    started = time.perf_counter()
    pointer = services.index_pointer
    version, index_name = pointer.next_version_name()
    chunking = services.chunking

    corpus = [
        (p, domain)
        for domain, cfg in DOMAINS.items()
        for p in sorted(Path(cfg["knowledge_base_path"]).glob("*"))
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}
    ]
    if not corpus:
        logger.error("no documents found under any knowledge_base_path")
        return 1

    print(f"target index : {index_name} (v{version})")
    print(f"documents    : {len(corpus)}")
    print(f"embed model  : {services.embedder.model_name}")
    print()

    # --- fit BM25 over the whole corpus before indexing anything ------------
    # IDF must be stable across every document or term weights are not
    # comparable between them, so this cannot be done incrementally.
    print("extracting and chunking the corpus...")
    texts: list[str] = []
    failures: list[str] = []
    # Chunks are kept so the indexing pass does not extract everything a
    # second time. Holding the whole corpus is a few MB of text and saves
    # roughly four minutes on this knowledge base.
    prepared: dict[str, list] = {}
    for i, (path, domain) in enumerate(corpus, 1):
        try:
            doc = extract_document(path)
            namespace = DOMAINS[domain]["pinecone_namespace"]
            chunks = chunk_document(doc, domain, config=chunking, scope=namespace)
            prepared[str(path)] = chunks
            texts.extend(c.text for c in chunks if not c.is_parent)
            print(f"  [{i}/{len(corpus)}] {path.name[:52]}", flush=True)
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            logger.warning("skipping %s during extraction: %s", path.name, exc)

    if not texts:
        logger.error("extraction produced no text — aborting")
        return 1

    encoder = BM25Encoder().fit(texts)
    print(f"\n  {len(texts)} chunks, {len(encoder.idf)} distinct terms\n")

    if dry_run:
        tokens = sorted(len(t.split()) for t in texts)
        print("DRY RUN — nothing written")
        print(f"  chunks       : {len(texts)}")
        print(f"  median words : {tokens[len(tokens) // 2]}")
        print(f"  failures     : {len(failures)}")
        for f in failures:
            print(f"    {f}")
        return 0

    encoder.save(SPARSE_ENCODER_PATH)

    # --- index into the new version -----------------------------------------
    print("embedding and upserting...")
    store = services.store_for_index(index_name)
    indexer = Indexer(
        store, services.embedder, services.registry, sparse_encoder=encoder, chunking=chunking
    )

    total = 0
    per_domain: dict[str, dict] = {}
    for domain, cfg in DOMAINS.items():
        namespace = cfg["pinecone_namespace"]
        results = indexer.index_directory(
            cfg["knowledge_base_path"],
            namespace=namespace,
            domain=domain,
            force=True,
            prepared=prepared,
        )
        indexed = sum(r.chunks_indexed for r in results)
        errored = [f"{r.filename}: {r.error}" for r in results if r.error]
        total += indexed
        per_domain[domain] = {"documents": len(results), "chunks": indexed, "errors": errored}
        print(f"  {domain:6s} {len(results):3d} docs -> {indexed:5d} chunks"
              + (f"  ({len(errored)} failed)" if errored else ""))
        for e in errored:
            print(f"         {e}")

    elapsed = time.perf_counter() - started
    print(f"\nindexed {total} chunks in {elapsed:.0f}s")

    if total == 0:
        logger.error("rebuild produced no chunks — not promoting")
        return 1

    all_errors = [e for d in per_domain.values() for e in d["errors"]]
    if all_errors and not force_promote:
        print(f"\n{len(all_errors)} document(s) failed. Not promoting.")
        print(f"Built index {index_name} is complete but inactive. Review the errors, then:")
        print(f"  python reindex.py --promote {index_name}   # accept as-is")
        print(f"  python reindex.py --rebuild --force        # promote despite failures")
        return 1

    pointer.promote(index_name, version, note=f"{total} chunks, {len(all_errors)} failures")
    services.reload_sparse_encoder()
    print(f"\npromoted {index_name} — now serving traffic")
    return 0


def cmd_promote(index_name: str) -> int:
    from services import services

    pointer = services.index_pointer
    try:
        version = int(index_name.rsplit("-v", 1)[1])
    except (IndexError, ValueError):
        logger.error("index name must end in -v<N>, got %r", index_name)
        return 1

    store = services.store_for_index(index_name)
    try:
        stats = store.stats()
    except Exception as exc:
        logger.error("cannot read %s — refusing to promote an unreachable index: %s", index_name, exc)
        return 1

    if not stats.get("total_vectors"):
        logger.error("%s is empty — refusing to promote", index_name)
        return 1

    pointer.promote(index_name, version, note="manual promotion")
    print(f"promoted {index_name} ({stats['total_vectors']} vectors)")
    return 0


def cmd_rollback() -> int:
    from services import services

    pointer = services.index_pointer
    history = json.loads(Path(pointer.path).read_text()).get("history", []) if Path(pointer.path).exists() else []
    if not history:
        logger.error("no previous version recorded")
        return 1

    previous = history[-1]
    print(f"rolling back from {pointer.active} to {previous['name']}")
    return cmd_promote(previous["name"])


def cmd_claim_sessions(email: str) -> int:
    import database as db
    from services import services

    db.init_db()
    before = db.orphaned_session_count()
    if not any(before.values()):
        print("no unowned sessions")
        return 0

    row = services.users._conn().execute(  # noqa: SLF001 - admin utility
        "SELECT user_id FROM users WHERE email=?", (email.strip().lower(),)
    ).fetchone()
    if not row:
        logger.error("no user with email %s — register first", email)
        return 1

    claimed = db.claim_orphaned_sessions(row["user_id"])
    print(f"assigned {claimed} session(s) to {email}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="EduPilot index management")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="show index state")
    group.add_argument("--rebuild", action="store_true", help="full blue/green rebuild")
    group.add_argument("--promote", metavar="INDEX", help="promote a built index")
    group.add_argument("--rollback", action="store_true", help="revert to the previous version")
    group.add_argument("--claim-sessions", metavar="EMAIL", help="assign pre-auth sessions")
    parser.add_argument("--dry-run", action="store_true", help="with --rebuild: no writes")
    parser.add_argument("--force", action="store_true", help="with --rebuild: promote despite failures")
    args = parser.parse_args()

    if args.status:
        return cmd_status()
    if args.rebuild:
        return cmd_rebuild(args.dry_run, args.force)
    if args.promote:
        return cmd_promote(args.promote)
    if args.rollback:
        return cmd_rollback()
    if args.claim_sessions:
        return cmd_claim_sessions(args.claim_sessions)
    return 1


if __name__ == "__main__":
    sys.exit(main())
