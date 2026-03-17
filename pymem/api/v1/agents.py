"""Agent memory management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from pymem.api.v1.dependencies import get_engine
from pymem.api.v1.schemas import AddInstructionRequest, MemoryResponse
from pymem.models import MemoryScope

router = APIRouter(prefix="/v1/agents", tags=["agents"])


@router.get("/{agent_id}/instructions", response_model=list[MemoryResponse])
async def get_agent_instructions(
    agent_id: str,
    engine=Depends(get_engine),
):
    """All procedural memories (instructions) for this agent."""
    scope = MemoryScope(agent_id=agent_id)
    memories = await engine.procedural.get_all(scope)
    return [
        MemoryResponse(
            id=m.id,
            memory_type=m.memory_type.value,
            content=m.content,
            score=m.score,
            metadata=m.metadata,
            agent_id=m.agent_id,
        )
        for m in memories
    ]


@router.post("/{agent_id}/instructions", response_model=MemoryResponse)
async def add_agent_instruction(
    agent_id: str,
    request: AddInstructionRequest,
    engine=Depends(get_engine),
):
    """Add a procedural memory (agent instruction)."""
    scope = MemoryScope(agent_id=agent_id)
    memory = await engine.procedural.add(
        content=request.content,
        scope=scope,
        tags=request.tags,
        priority=request.priority,
    )
    return MemoryResponse(
        id=memory.id,
        memory_type=memory.memory_type.value,
        content=memory.content,
        score=memory.score,
        metadata=memory.metadata,
        agent_id=memory.agent_id,
    )
