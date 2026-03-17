"""User memory management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from pymem.api.v1.dependencies import get_engine
from pymem.api.v1.schemas import ForgetRequest, ForgetResponse, MemoryResponse
from pymem.models import MemoryScope, MemoryType

router = APIRouter(prefix="/v1/users", tags=["users"])


@router.get("/{user_id}/memories", response_model=list[MemoryResponse])
async def list_user_memories(
    user_id: str,
    type: str | None = None,
    limit: int = 50,
    offset: int = 0,
    engine=Depends(get_engine),
):
    """Paginated list of all user memories."""
    scope = MemoryScope(user_id=user_id)
    rows = await engine._relational.list_memories(
        filters=scope.to_filter(),
        memory_type=type,
        limit=limit,
        offset=offset,
    )
    return [
        MemoryResponse(
            id=r["id"],
            memory_type=r["memory_type"],
            content=r["content"],
            score=r.get("score", 0.5),
            user_id=r.get("user_id"),
        )
        for r in rows
    ]


@router.delete("/{user_id}/memories")
async def delete_user_memories(
    user_id: str,
    type: str | None = None,
    engine=Depends(get_engine),
):
    """Bulk delete user memories."""
    scope = MemoryScope(user_id=user_id)
    result = await engine.forget(
        scope=scope,
        memory_types=[MemoryType(type)] if type else None,
    )
    return {"deleted_per_type": result.deleted_per_type}


@router.post("/{user_id}/forget", response_model=ForgetResponse)
async def forget_user(
    user_id: str,
    request: ForgetRequest,
    engine=Depends(get_engine),
):
    """GDPR-compliant forget. Wipes all backends."""
    scope = MemoryScope(user_id=user_id)
    memory_types = None
    if request.scope != "all":
        memory_types = [MemoryType(request.scope)]

    result = await engine.forget(scope=scope, memory_types=memory_types)
    return ForgetResponse(
        deleted_per_type=result.deleted_per_type,
        completed_at=result.completed_at,
        confirmation_id=result.confirmation_id,
    )


@router.get("/{user_id}/entities")
async def get_user_entities(
    user_id: str,
    engine=Depends(get_engine),
):
    """Entity graph for a user."""
    scope = MemoryScope(user_id=user_id)
    # Search for entity-type memories
    result = await engine.search(
        query="entities relationships",
        scope=scope,
        memory_types=[MemoryType.SEMANTIC],
        limit=20,
    )
    return {
        "user_id": user_id,
        "entities": [
            {
                "id": m.id,
                "content": m.content,
                "entities": m.metadata.get("entities", []),
            }
            for m in result.memories
        ],
    }
