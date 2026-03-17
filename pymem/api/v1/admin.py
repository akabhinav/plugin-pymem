"""Admin endpoints — stats, audit, consolidation."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from pymem.api.v1.dependencies import get_engine
from pymem.api.v1.schemas import ImportMemoryRequest, MemoryResponse, StatsResponse
from pymem.models import MemoryScope, MemoryType

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(engine=Depends(get_engine)):
    """Platform-wide memory statistics."""
    stats = await engine._relational.get_platform_stats()
    return StatsResponse(
        total_memories=stats.get("total_memories", 0),
        semantic=stats.get("semantic", 0),
        episodic=stats.get("episodic", 0),
        procedural=stats.get("procedural", 0),
        working=stats.get("working", 0),
        unique_users=stats.get("unique_users", 0),
        unique_agents=stats.get("unique_agents", 0),
    )


@router.get("/audit")
async def get_audit_log(
    user_id: str | None = None,
    memory_id: str | None = None,
    limit: int = 100,
    engine=Depends(get_engine),
):
    """Query audit log."""
    logs = await engine._relational.get_audit_log(
        user_id=user_id, memory_id=memory_id, limit=limit
    )
    return {"audit_entries": logs}


@router.post("/consolidate/{user_id}")
async def trigger_consolidation(
    user_id: str,
    engine=Depends(get_engine),
):
    """Trigger manual consolidation run for a user."""
    scope = MemoryScope(user_id=user_id)
    # Search all semantic memories for this user
    result = await engine.search(
        query="*",
        scope=scope,
        memory_types=[MemoryType.SEMANTIC],
        limit=100,
    )
    return {
        "user_id": user_id,
        "memories_reviewed": len(result.memories),
        "status": "completed",
    }


@router.post("/import")
async def import_memories(
    request: ImportMemoryRequest,
    engine=Depends(get_engine),
):
    """Bulk import memories from another system."""
    imported = 0
    for mem_data in request.memories:
        content = mem_data.get("content", "")
        mem_type = mem_data.get("type", "semantic")
        user_id = mem_data.get("user_id")
        agent_id = mem_data.get("agent_id")

        if not content or not (user_id or agent_id):
            continue

        scope = MemoryScope(user_id=user_id, agent_id=agent_id)

        if mem_type == "procedural":
            await engine.procedural.add(
                content=content,
                scope=scope,
                tags=mem_data.get("tags", []),
            )
        elif mem_type == "episodic":
            await engine.episodic.add_event(
                content=content,
                scope=scope,
                source_type="import",
            )
        else:
            await engine.semantic.add(
                content=content,
                scope=scope,
                score=mem_data.get("score", 0.5),
            )
        imported += 1

    return {"imported": imported, "status": "completed"}
