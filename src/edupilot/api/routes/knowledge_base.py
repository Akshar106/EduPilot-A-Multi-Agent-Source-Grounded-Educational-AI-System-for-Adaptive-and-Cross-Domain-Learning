"""
Course knowledge base
=====================
The shared corpus every student's answers are grounded in.

Reads are open to any authenticated user; mutation is admin-only. Both were
previously unauthenticated, so any visitor could inject documents that every
student's answers would then cite.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from edupilot import db
from edupilot.api.deps import AdminUser, CurrentUser, run_blocking
from edupilot.core.config import DOMAINS
from edupilot.core.services import services
from edupilot.security import AppError, ErrorCode, rate_limited, resolve_within, validate_batch
from edupilot.security.uploads import UploadRejected

router = APIRouter(prefix="/api", tags=["knowledge-base"])


def _require_domain(domain: str) -> dict:
    if domain not in DOMAINS:
        raise AppError(
            code=ErrorCode.VALIDATION_FAILED, message=f"Unknown domain '{domain}'."
        )
    return DOMAINS[domain]


@router.get("/kb/status", dependencies=[Depends(rate_limited("read"))])
async def kb_status(user: CurrentUser):
    stats = await run_blocking(services.store.stats)
    out = {}
    for domain, cfg in DOMAINS.items():
        namespace = cfg["pinecone_namespace"]
        out[domain] = {
            "name": cfg["name"],
            "color": cfg["color"],
            "chunk_count": stats.get("namespaces", {}).get(namespace, 0),
            "documents": services.registry.list_documents(namespace),
        }
    return out


@router.get("/kb/documents", dependencies=[Depends(rate_limited("read"))])
async def list_kb_documents(user: CurrentUser):
    return {
        domain: {
            "name": cfg["name"],
            "color": cfg["color"],
            "documents": services.registry.list_documents(cfg["pinecone_namespace"]),
        }
        for domain, cfg in DOMAINS.items()
    }


@router.post("/kb/upload", dependencies=[Depends(rate_limited("upload"))])
async def upload_to_kb(
    admin: AdminUser,
    domain: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Add documents to the shared course knowledge base. Admin only."""
    cfg = _require_domain(domain)

    raw = [(f.filename or "", await f.read()) for f in files]
    try:
        validated = validate_batch(raw)
    except UploadRejected as exc:
        raise AppError(code=ErrorCode.UPLOAD_REJECTED, message=str(exc)) from exc

    kb_path = Path(cfg["knowledge_base_path"])
    kb_path.mkdir(parents=True, exist_ok=True)
    namespace = cfg["pinecone_namespace"]

    results = []
    for item in validated:
        # resolve_within re-checks containment after sanitization. The old
        # code did `kb_path / uf.filename` with no check at all.
        dest = resolve_within(kb_path, item.safe_name)
        dest.write_bytes(item.content)

        outcome = await run_blocking(
            services.indexer.index_document, dest, namespace=namespace, domain=domain
        )
        db.save_uploaded_doc(
            filename=item.safe_name,
            domain=domain,
            file_type=item.extension,
            chunk_count=outcome.chunks_indexed,
            file_size_bytes=item.size_bytes,
        )
        results.append({
            "filename": item.safe_name,
            "original_name": item.original_name,
            "chunks_indexed": outcome.chunks_indexed,
            "skipped": outcome.skipped,
            "replaced_version": outcome.replaced_version,
            "error": outcome.error,
            "warnings": item.warnings,
        })

    return {"uploaded": results}


@router.get("/documents/{domain}/{filename}", dependencies=[Depends(rate_limited("read"))])
async def serve_document(domain: str, filename: str, user: CurrentUser):
    """Serve a source PDF so a citation can be opened at its page."""
    if domain not in DOMAINS:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"unknown domain {domain}")
    kb_path = Path(DOMAINS[domain]["knowledge_base_path"])
    try:
        path = resolve_within(kb_path, filename)
    except UploadRejected as exc:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=str(exc)) from exc
    if not path.is_file():
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"missing {path}")
    return FileResponse(str(path), filename=path.name)


@router.delete("/kb/{domain}/{filename}")
async def delete_kb_document(domain: str, filename: str, admin: AdminUser):
    if domain not in DOMAINS:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"unknown domain {domain}")
    namespace = DOMAINS[domain]["pinecone_namespace"]
    safe = resolve_within(Path(DOMAINS[domain]["knowledge_base_path"]), filename)

    removed = await run_blocking(services.indexer.remove_document, namespace, safe.name)
    if safe.is_file():
        safe.unlink()
    return {"deleted": safe.name, "was_indexed": removed}
