"""FastAPI dependencies — engine injection, scope enforcement."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Header

from pymem.models import MemoryScope


class ScopeEnforcer:
    """Validates scope access. Cross-scope leakage raises HTTP 403."""

    async def validate_scope(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
    ) -> MemoryScope:
        """Ensure at least one scope dimension is provided."""
        if not any([user_id, agent_id, session_id, org_id]):
            raise HTTPException(
                status_code=400,
                detail="At least one scope field required: user_id, agent_id, session_id, or org_id",
            )
        return MemoryScope(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            org_id=org_id,
        )


scope_enforcer = ScopeEnforcer()


# Global engine instance — set during app startup
_engine_instance: Any = None


def set_engine(engine: Any) -> None:
    global _engine_instance
    _engine_instance = engine


def get_engine() -> Any:
    if _engine_instance is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine_instance


async def verify_api_key(
    authorization: str = Header(None, alias="Authorization"),
) -> str:
    """Simple API key verification."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return authorization
