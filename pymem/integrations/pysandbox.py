"""PySandbox integration — persists workspace state across sandbox lifecycle."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pymem.models import MemoryScope

if TYPE_CHECKING:
    from pymem.engine.memory_engine import MemoryEngine


class SandboxMemoryBridge:
    """Persists workspace state to PyMem on sandbox pause. Restores on resume."""

    def __init__(self, engine: MemoryEngine) -> None:
        self._engine = engine

    async def on_sandbox_pause(
        self, sandbox_id: str, workspace_state: dict[str, Any]
    ) -> None:
        """Snapshot working memory + task state to episodic memory."""
        await self._engine.episodic.add_event(
            content=f"Sandbox {sandbox_id} paused. State: {json.dumps(workspace_state)}",
            scope=MemoryScope(session_id=sandbox_id),
            metadata={"event_type": "sandbox_pause", "state": workspace_state},
        )

    async def on_sandbox_resume(self, sandbox_id: str) -> dict[str, Any] | None:
        """Retrieve last saved state for this sandbox."""
        results = await self._engine.episodic.get_recent(
            scope=MemoryScope(session_id=sandbox_id), limit=1
        )
        if results:
            return results[0].metadata.get("state")
        return None
