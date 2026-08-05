"""
Evaluation
==========
Listing the suite is cheap and open to any authenticated user. Actually
*running* it is hundreds of LLM calls, so no route executes it — that is a
deliberate operator action via the `edupilot-evaluate` CLI.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from edupilot.api.deps import CurrentUser
from edupilot.security import rate_limited

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])


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
