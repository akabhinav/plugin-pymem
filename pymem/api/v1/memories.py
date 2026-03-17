"""Core memory CRUD + search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from pymem.api.v1.dependencies import get_engine
from pymem.api.v1.schemas import (
    AddMemoryRequest,
    AddMemoryResponse,
    ContextResponse,
    MemoryResponse,
    SearchMemoryRequest,
    SearchMemoryResponse,
    ShareMemoryRequest,
    UpdateMemoryRequest,
)
from pymem.models import MemoryScope, MemoryType

router = APIRouter(prefix="/v1/memory", tags=["memories"])


@router.post("/add", response_model=AddMemoryResponse)
async def add_memory(
    request: AddMemoryRequest,
    engine=Depends(get_engine),
):
    """Extract and store memories from a conversation."""
    scope = MemoryScope(
        user_id=request.user_id,
        agent_id=request.agent_id,
        session_id=request.session_id,
        org_id=request.org_id,
    )

    memory_types = None
    if request.memory_types:
        memory_types = [MemoryType(t) for t in request.memory_types]

    result = await engine.add(
        messages=request.messages,
        scope=scope,
        memory_types=memory_types,
    )

    return AddMemoryResponse(
        status=result.status,
        memories_count=result.memories_count,
        memories=[
            MemoryResponse(
                id=m.id,
                memory_type=m.memory_type.value,
                content=m.content,
                score=m.score,
                created_at=m.created_at,
                user_id=m.user_id,
                agent_id=m.agent_id,
            )
            for m in result.memories
        ],
    )


@router.post("/search", response_model=SearchMemoryResponse)
async def search_memories(
    request: SearchMemoryRequest,
    engine=Depends(get_engine),
):
    """Semantic + temporal + procedural search across memory types."""
    scope = MemoryScope(
        user_id=request.user_id,
        agent_id=request.agent_id,
        session_id=request.session_id,
        org_id=request.org_id,
    )

    memory_types = None
    if request.types:
        memory_types = [MemoryType(t) for t in request.types]

    result = await engine.search(
        query=request.query,
        scope=scope,
        memory_types=memory_types,
        limit=request.limit,
    )

    return SearchMemoryResponse(
        memories=[
            MemoryResponse(
                id=m.id,
                memory_type=m.memory_type.value,
                content=m.content,
                score=m.score,
                user_id=m.user_id,
                agent_id=m.agent_id,
            )
            for m in result.memories
        ],
        total=result.total,
    )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    engine=Depends(get_engine),
):
    """Get a specific memory record."""
    memory = await engine.get(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(
        id=memory.id,
        memory_type=memory.memory_type.value,
        content=memory.content,
        score=memory.score,
        access_count=memory.access_count,
        user_id=memory.user_id,
        agent_id=memory.agent_id,
        version=memory.version,
    )


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    request: UpdateMemoryRequest,
    user_id: str | None = None,
    engine=Depends(get_engine),
):
    """Update memory content. Creates new version."""
    scope = MemoryScope(user_id=user_id or "system")
    result = await engine.update(memory_id, request.content, scope)
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(
        id=result.id,
        memory_type=result.memory_type.value,
        content=result.content,
        version=result.version,
    )


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    hard: bool = False,
    user_id: str | None = None,
    engine=Depends(get_engine),
):
    """Delete a memory. Soft delete by default."""
    scope = MemoryScope(user_id=user_id or "system")
    await engine.delete(memory_id, scope, hard_delete=hard)
    return {"status": "deleted", "memory_id": memory_id}


@router.get("/context/assemble", response_model=ContextResponse)
async def get_context(
    user_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    query: str | None = None,
    max_tokens: int = 4000,
    engine=Depends(get_engine),
):
    """Assembled agent context from all memory types."""
    scope = MemoryScope(
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
    )
    ctx = await engine.get_context(scope, query, max_tokens)
    return ContextResponse(
        formatted_prompt=ctx.formatted,
        working_messages=ctx.working_messages,
        memories=[
            MemoryResponse(
                id=m.id,
                memory_type=m.memory_type.value,
                content=m.content,
                score=m.score,
            )
            for m in ctx.relevant_facts
        ],
        instructions=[
            MemoryResponse(
                id=m.id,
                memory_type=m.memory_type.value,
                content=m.content,
            )
            for m in ctx.agent_instructions
        ],
    )


@router.post("/share")
async def share_memory(
    request: ShareMemoryRequest,
    engine=Depends(get_engine),
):
    """Clone memory into a different scope."""
    from_scope = MemoryScope(**request.from_scope)
    to_scope = MemoryScope(**request.to_scope)
    result = await engine.share(request.memory_id, from_scope, to_scope)
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse(
        id=result.id,
        memory_type=result.memory_type.value,
        content=result.content,
    )
