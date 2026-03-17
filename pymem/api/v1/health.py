"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from pymem.api.v1.dependencies import get_engine
from pymem.api.v1.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(engine=Depends(get_engine)):
    """Check health of all backends."""
    vector_ok = await engine._vector.health_check()
    graph_ok = await engine._graph.health_check()

    cache_ok = False
    if engine.working and hasattr(engine.working, "_cache"):
        try:
            cache_ok = await engine.working._cache.health_check()
        except Exception:
            cache_ok = False

    all_ok = vector_ok and graph_ok
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        version="1.0.0",
        vector_store=vector_ok,
        graph_store=graph_ok,
        cache=cache_ok,
        relational=True,
    )
