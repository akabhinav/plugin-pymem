"""PyOz integration — agent lifecycle hooks for automatic memory management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymem.models import MemoryScope

if TYPE_CHECKING:
    from pymem.engine.memory_engine import MemoryEngine


class PyOzMemoryHooks:
    """Hooks called by PyOz agent lifecycle events.

    Auto-writes memories without the developer doing anything.
    """

    def __init__(self, engine: MemoryEngine) -> None:
        self._engine = engine

    async def on_session_start(
        self, agent_id: str, user_id: str, session_id: str
    ) -> str:
        """Load context and inject into agent system prompt."""
        context = await self._engine.get_context(
            scope=MemoryScope(
                user_id=user_id, agent_id=agent_id, session_id=session_id
            )
        )
        return context.formatted

    async def on_session_end(
        self,
        messages: list[dict[str, str]],
        agent_id: str,
        user_id: str,
        session_id: str,
    ) -> None:
        """Extract and store memories from completed session."""
        scope = MemoryScope(
            user_id=user_id, agent_id=agent_id, session_id=session_id
        )
        await self._engine.add(messages=messages, scope=scope)
        if self._engine.working:
            await self._engine.working.expire_session(session_id)

    async def on_task_complete(
        self,
        task_description: str,
        outcome: str,
        agent_id: str,
        user_id: str,
    ) -> None:
        """Store episodic memory of completed task."""
        await self._engine.episodic.add_event(
            content=f"Completed task: {task_description}. Outcome: {outcome}",
            scope=MemoryScope(user_id=user_id, agent_id=agent_id),
            source_type="task",
        )

    async def on_instruction_learned(
        self, instruction: str, agent_id: str
    ) -> None:
        """Store a new procedural memory."""
        await self._engine.procedural.add(
            content=instruction,
            scope=MemoryScope(agent_id=agent_id),
        )
