"""Session (working) memory endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from pymem.api.v1.dependencies import get_engine
from pymem.api.v1.schemas import WorkingMemoryRequest
from pymem.models import MemoryScope

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


@router.get("/{session_id}/working")
async def get_working_memory(
    session_id: str,
    engine=Depends(get_engine),
):
    """Get current working memory window for session."""
    if not engine.working:
        raise HTTPException(status_code=503, detail="Working memory not available")

    scope = MemoryScope(session_id=session_id)
    messages = await engine.working.get_window(scope)
    return {"session_id": session_id, "messages": messages}


@router.post("/{session_id}/working")
async def append_working_memory(
    session_id: str,
    request: WorkingMemoryRequest,
    engine=Depends(get_engine),
):
    """Append messages to working memory window."""
    if not engine.working:
        raise HTTPException(status_code=503, detail="Working memory not available")

    scope = MemoryScope(session_id=session_id)
    await engine.working.append(request.messages, scope)
    length = await engine.working.get_length(session_id)
    return {"session_id": session_id, "message_count": length}


@router.delete("/{session_id}/working")
async def clear_working_memory(
    session_id: str,
    engine=Depends(get_engine),
):
    """Clear working memory (on session end)."""
    if not engine.working:
        raise HTTPException(status_code=503, detail="Working memory not available")

    await engine.working.expire_session(session_id)
    return {"session_id": session_id, "status": "cleared"}
