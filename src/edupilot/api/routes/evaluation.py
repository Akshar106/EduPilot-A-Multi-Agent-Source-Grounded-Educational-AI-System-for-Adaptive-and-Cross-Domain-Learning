"""
Evaluation
==========
Lists the suite, and runs it **one case per request**.

A full pass is 50 cases at roughly half a minute each. Running them inside a
single request would hold a worker for ~25 minutes and time out in the browser
long before finishing, so the client drives the loop and each case is its own
short request. That also makes progress reportable and lets a run be abandoned
part-way without stranding anything.

Running is admin-only and rate limited: each case is several LLM calls, and on
a metered provider a careless loop is real money.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from edupilot.api.deps import AdminUser, CurrentUser, run_blocking
from edupilot.core.config import DEFAULT_MODEL, DEFAULT_RERANK_TOP_K, DEFAULT_TOP_K
from edupilot.security import AppError, ErrorCode, rate_limited

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])


class RunCaseRequest(BaseModel):
    model: str = DEFAULT_MODEL
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    rerank_top_k: int = Field(default=DEFAULT_RERANK_TOP_K, ge=1, le=12)
    enable_verification: bool = True


class SummaryRequest(BaseModel):
    # Bounded so a client cannot post an unbounded blob for aggregation.
    results: list[dict] = Field(default_factory=list, max_length=200)


@router.get("/cases", dependencies=[Depends(rate_limited("read"))])
async def list_test_cases(user: CurrentUser):
    from edupilot.evaluation import TEST_CASES

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
            for tc in TEST_CASES
        ]
    }


def _run_one(case_id: str, req: RunCaseRequest) -> dict:
    """Score a single case. Runs in the worker pool — several LLM calls."""
    from edupilot.evaluation import TEST_CASES, run_evaluation
    from edupilot.evaluation.live import build_pipeline_fn, result_to_dict

    case = next((tc for tc in TEST_CASES if tc.id.upper() == case_id.upper()), None)
    if case is None:
        raise AppError(code=ErrorCode.NOT_FOUND, internal=f"unknown test case {case_id}")

    result = run_evaluation(
        test_case=case,
        pipeline_fn=build_pipeline_fn(),
        model=req.model,
        top_k=req.top_k,
        rerank_top_k=req.rerank_top_k,
        enable_verification=req.enable_verification,
    )
    return result_to_dict(result)


@router.post("/cases/{case_id}", dependencies=[Depends(rate_limited("chat"))])
async def run_test_case(case_id: str, req: RunCaseRequest, admin: AdminUser):
    """Run one case and return its metrics."""
    return await run_blocking(_run_one, case_id, req)


@router.post("/summary")
async def summarize(req: SummaryRequest, admin: AdminUser):
    """
    Aggregate per-case results the client has collected.

    Server-side so the scoping rules — faithfulness excludes edge cases,
    quality excludes correct refusals — stay in one implementation rather than
    being mirrored in the browser and drifting.
    """
    from edupilot.evaluation.live import summarize_dicts

    return {"stats": summarize_dicts(req.results)}
