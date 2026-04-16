"""
EduPilot — FastAPI Backend
==========================
Run with:
    uvicorn main:app --reload --port 8000

Environment variables (set in .env or shell):
    GROQ_API_KEY=gsk_...
    PINECONE_API_KEY=...
"""

from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database as db
from config import (
    AVAILABLE_MODELS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_TOP_K,
    DOMAINS,
)

# ---------------------------------------------------------------------------
# Thread pool for sync pipeline calls (avoids blocking the event loop)
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# Lazy-loaded pipeline modules
# ---------------------------------------------------------------------------
_pipeline: dict | None = None


def _get_pipeline() -> dict:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from retriever import get_retriever, initialize_all_retrievers
    from router import (
        classify_query,
        get_clarification_message,
        get_out_of_domain_message,
        should_ask_for_clarification,
    )
    from query_splitter import split_query
    from reranker import rerank, score_summary
    from synthesizer import generate_domain_answer, synthesize_answers
    from verifier import get_final_answer, verify_answer
    from evaluation import TEST_CASES, run_all_evaluations, run_evaluation, summary_stats

    _pipeline = {
        "get_retriever": get_retriever,
        "initialize_all_retrievers": initialize_all_retrievers,
        "classify_query": classify_query,
        "should_ask_for_clarification": should_ask_for_clarification,
        "get_clarification_message": get_clarification_message,
        "get_out_of_domain_message": get_out_of_domain_message,
        "split_query": split_query,
        "rerank": rerank,
        "score_summary": score_summary,
        "generate_domain_answer": generate_domain_answer,
        "synthesize_answers": synthesize_answers,
        "verify_answer": verify_answer,
        "get_final_answer": get_final_answer,
        "TEST_CASES": TEST_CASES,
        "run_evaluation": run_evaluation,
        "run_all_evaluations": run_all_evaluations,
        "summary_stats": summary_stats,
    }
    return _pipeline


# ---------------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — initialise DB + all domain retrievers
    db.init_db()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, lambda: _get_pipeline()["initialize_all_retrievers"]())
    yield
    # Shutdown
    _executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EduPilot API",
    description="Multi-Agent Educational RAG System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: str
    model: str = DEFAULT_MODEL
    top_k: int = DEFAULT_TOP_K
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    enable_verification: bool = True
    manual_domains: list[str] | None = None
    attached_filenames: list[str] | None = None
    chat_history: list[dict] | None = None


class NewSessionRequest(BaseModel):
    title: str | None = None


# ---------------------------------------------------------------------------
# Sync pipeline runner (runs in executor thread)
# ---------------------------------------------------------------------------
def _run_pipeline(req: ChatRequest) -> dict:
    p = _get_pipeline()
    debug: dict = {}

    router_result = p["classify_query"](
        req.query, model=req.model, chat_history=req.chat_history or []
    )
    debug["router"] = {
        "intent_type": router_result.intent_type,
        "domains": router_result.domains,
        "is_course_related": router_result.is_course_related,
        "needs_clarification": router_result.needs_clarification,
        "reasoning": router_result.reasoning,
    }

    effective_domains = req.manual_domains if req.manual_domains else router_result.domains

    # Early exits — skip if the user manually supplied domains (e.g. via attachment)
    if not router_result.is_course_related and not req.manual_domains:
        return {
            "final_answer": p["get_out_of_domain_message"](),
            "intent_type": router_result.intent_type,
            "detected_domains": [],
            "sub_questions": [],
            "is_course_related": False,
            "needs_clarification": False,
            "quality_score": 0.0,
            "verification_issues": [],
            "verification_revised": False,
            "debug": debug,
        }

    if p["should_ask_for_clarification"](router_result, req.query) and not req.manual_domains:
        return {
            "final_answer": p["get_clarification_message"](router_result),
            "intent_type": router_result.intent_type,
            "detected_domains": [],
            "sub_questions": [],
            "is_course_related": True,
            "needs_clarification": True,
            "quality_score": 0.0,
            "verification_issues": [],
            "verification_revised": False,
            "debug": debug,
        }

    # Step 2: Decompose
    sub_questions = p["split_query"](
        query=req.query,
        intent_type=router_result.intent_type,
        detected_domains=effective_domains,
        model=req.model,
    )
    debug["sub_questions"] = sub_questions

    # Steps 3–5: Retrieve → Rerank → Generate
    domain_answers = []
    debug["retrieval"] = []

    for sq in sub_questions:
        domain = sq["domain"]
        question = sq["question"]

        retriever = p["get_retriever"](domain)
        raw_chunks = retriever.retrieve(
            question, top_k=req.top_k,
            source_filter=req.attached_filenames or None,
        )
        reranked = p["rerank"](
            query=question,
            chunks=raw_chunks,
            top_k=req.rerank_top_k,
            confidence_threshold=req.confidence_threshold,
        )

        debug["retrieval"].append({
            "domain": domain,
            "question": question,
            "raw_count": len(raw_chunks),
            "reranked_count": len(reranked),
            "score_summary": p["score_summary"](reranked),
            "chunks": [
                {
                    "text": c.text[:300],
                    "source": c.citation_label(),
                    "rerank_score": round(c.rerank_score, 4),
                    "semantic_score": round(c.semantic_score, 4),
                    "bm25_score": round(c.bm25_score, 4),
                }
                for c in reranked
            ],
        })

        da = p["generate_domain_answer"](
            sub_question=question,
            domain=domain,
            retrieved_chunks=reranked,
            model=req.model,
            chat_history=req.chat_history or [],
        )
        domain_answers.append(da)

    debug["domain_answers"] = [
        {"domain": da.domain, "question": da.sub_question, "preview": da.answer[:300]}
        for da in domain_answers
    ]

    # Step 6: Synthesis
    synthesized = p["synthesize_answers"](
        original_query=req.query,
        domain_answers=domain_answers,
        model=req.model,
    )
    debug["synthesized_preview"] = synthesized[:500]

    # Step 7: Verification
    verification = p["verify_answer"](
        original_query=req.query,
        sub_questions=sub_questions,
        domain_answers=domain_answers,
        synthesized_answer=synthesized,
        model=req.model,
        enabled=req.enable_verification,
    )
    debug["verification"] = {
        "is_satisfactory": verification.is_satisfactory,
        "quality_score": verification.quality_score,
        "coverage_score": verification.coverage_score,
        "grounding_score": verification.grounding_score,
        "issues": verification.issues,
        "missing_topics": verification.missing_topics,
        "was_revised": verification.revised_answer is not None,
        "skipped": verification.skipped,
    }

    final_answer = p["get_final_answer"](synthesized, verification)

    return {
        "final_answer": final_answer,
        "synthesized_answer": synthesized,
        "intent_type": router_result.intent_type,
        "detected_domains": effective_domains,
        "sub_questions": sub_questions,
        "is_course_related": True,
        "needs_clarification": False,
        "quality_score": verification.quality_score,
        "verification_issues": verification.issues,
        "verification_revised": verification.revised_answer is not None,
        "debug": debug,
    }


# ---------------------------------------------------------------------------
# Routes — frontend
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Routes — health / config
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    missing = []
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.getenv("PINECONE_API_KEY"):
        missing.append("PINECONE_API_KEY")
    return {
        "status": "ok" if not missing else "degraded",
        "missing_keys": missing,
        "domains": list(DOMAINS.keys()),
    }


@app.get("/api/config")
async def get_config():
    return {
        "available_models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL,
        "domains": {
            k: {
                "name": v["name"],
                "abbr": v["abbr"],
                "color": v["color"],
                "description": v["description"],
            }
            for k, v in DOMAINS.items()
        },
        "defaults": {
            "top_k": DEFAULT_TOP_K,
            "rerank_top_k": DEFAULT_RERANK_TOP_K,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        },
    }


# ---------------------------------------------------------------------------
# Routes — Knowledge Base
# ---------------------------------------------------------------------------
@app.get("/api/kb/status")
async def kb_status():
    p = _get_pipeline()
    result = {}
    for domain, cfg in DOMAINS.items():
        r = p["get_retriever"](domain)
        kb_path = Path(cfg["knowledge_base_path"])
        files = []
        for ext in [".pdf", ".txt", ".md", ".docx"]:
            files.extend(f.name for f in kb_path.glob(f"*{ext}"))
        result[domain] = {
            "name": cfg["name"],
            "color": cfg["color"],
            "chunk_count": r.document_count(),
            "kb_files": sorted(files),
            "uploaded_docs": db.list_uploaded_docs(domain=domain),
        }
    return result


@app.post("/api/kb/upload")
async def upload_document(
    domain: str = Form(...),
    files: list[UploadFile] = File(...),
):
    if domain not in DOMAINS:
        raise HTTPException(400, f"Unknown domain: {domain}")

    p = _get_pipeline()
    cfg = DOMAINS[domain]
    kb_path = Path(cfg["knowledge_base_path"])
    kb_path.mkdir(parents=True, exist_ok=True)

    results = []
    for uf in files:
        raw = await uf.read()
        dest = kb_path / uf.filename
        dest.write_bytes(raw)

        # Index in thread pool
        loop = asyncio.get_event_loop()
        retriever = p["get_retriever"](domain)
        n_chunks = await loop.run_in_executor(
            _executor, retriever.add_documents, [str(dest)]
        )

        db.save_uploaded_doc(
            filename=uf.filename,
            domain=domain,
            file_type=Path(uf.filename).suffix.lower(),
            chunk_count=n_chunks,
            file_size_bytes=len(raw),
        )
        results.append({"filename": uf.filename, "chunks_indexed": n_chunks})

    return {"uploaded": results}


# ---------------------------------------------------------------------------
# Routes — Chat
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Ensure session exists
    db.ensure_session(req.session_id)

    # Persist user message
    user_msg_id = db.save_message(req.session_id, "user", req.query)
    if len(db.get_session_messages(req.session_id)) == 1:
        db.update_session_title(req.session_id, req.query[:60])

    # Run pipeline in thread pool
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, _run_pipeline, req)
    except Exception as exc:
        raise HTTPException(500, f"Pipeline error: {exc}") from exc

    # Persist assistant message
    assistant_msg_id = db.save_message(
        session_id=req.session_id,
        role="assistant",
        content=result["final_answer"],
        intent_type=result.get("intent_type"),
        detected_domains=result.get("detected_domains"),
        quality_score=result.get("quality_score"),
        pipeline_meta=result.get("debug"),
    )

    result["user_message_id"] = user_msg_id
    result["assistant_message_id"] = assistant_msg_id
    return result


# ---------------------------------------------------------------------------
# Routes — Sessions
# ---------------------------------------------------------------------------
@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": db.list_sessions(limit=20)}


@app.post("/api/sessions")
async def create_session(req: NewSessionRequest):
    session_id = str(uuid.uuid4())
    db.ensure_session(session_id, title=req.title)
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    messages = db.get_session_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    db.delete_session(session_id)
    return {"deleted": session_id}


@app.delete("/api/sessions/{session_id}/messages/{message_id}")
async def truncate_from_message(session_id: str, message_id: int):
    """Delete a message and all subsequent messages (used when editing a sent message)."""
    db.delete_messages_from(session_id, message_id)
    return {"truncated": True, "from_message_id": message_id}


# ---------------------------------------------------------------------------
# Routes — Evaluation
# ---------------------------------------------------------------------------
@app.get("/api/evaluate/cases")
async def list_test_cases():
    p = _get_pipeline()
    return {
        "test_cases": [
            {
                "id": tc.id,
                "name": tc.name,
                "query": tc.query,
                "expected_intent": tc.expected_intent,
                "expected_domains": tc.expected_domains,
                "expected_behavior": tc.expected_behavior,
                "category": tc.category,
            }
            for tc in p["TEST_CASES"]
        ]
    }


def _run_single_eval(tc_id: str) -> dict:
    p = _get_pipeline()
    tc = next((t for t in p["TEST_CASES"] if t.id == tc_id), None)
    if not tc:
        return {"error": f"Test case {tc_id} not found"}

    def pipeline_fn(query, **kw):
        from pydantic import BaseModel as BM
        req = ChatRequest(
            query=query,
            session_id="eval-" + tc_id,
            model=kw.get("model", DEFAULT_MODEL),
            top_k=kw.get("top_k", DEFAULT_TOP_K),
            rerank_top_k=kw.get("rerank_top_k", DEFAULT_RERANK_TOP_K),
            enable_verification=kw.get("enable_verification", True),
        )
        return _run_pipeline(req)

    r = p["run_evaluation"](test_case=tc, pipeline_fn=pipeline_fn)
    return {
        "test_case_id": tc.id,
        "name": tc.name,
        "passed": r.passed,
        "intent_match": r.intent_match,
        "domain_match": r.domain_match,
        "actual_intent": r.actual_intent,
        "actual_domains": r.actual_domains,
        "quality_score": r.quality_score,
        "answer_preview": r.answer_preview,
        "error": r.error,
        "expected_behavior": tc.expected_behavior,
    }


@app.post("/api/evaluate/{tc_id}")
async def run_test_case(tc_id: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run_single_eval, tc_id)
    if "error" in result and "not found" in result.get("error", ""):
        raise HTTPException(404, result["error"])
    return result


@app.post("/api/evaluate")
async def run_all_evals():
    p = _get_pipeline()

    def _run_all():
        def pipeline_fn(query, **kw):
            req = ChatRequest(
                query=query,
                session_id="eval-all",
                model=kw.get("model", DEFAULT_MODEL),
                top_k=kw.get("top_k", DEFAULT_TOP_K),
                rerank_top_k=kw.get("rerank_top_k", DEFAULT_RERANK_TOP_K),
                enable_verification=kw.get("enable_verification", True),
            )
            return _run_pipeline(req)

        results = p["run_all_evaluations"](pipeline_fn=pipeline_fn)
        stats = p["summary_stats"](results)
        return {
            "stats": stats,
            "results": [
                {
                    "test_case_id": r.test_case.id,
                    "name": r.test_case.name,
                    "passed": r.passed,
                    "intent_match": r.intent_match,
                    "domain_match": r.domain_match,
                    "quality_score": r.quality_score,
                    "answer_preview": r.answer_preview,
                    "error": r.error,
                }
                for r in results
            ],
        }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _run_all)
