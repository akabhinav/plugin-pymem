"""Context assembler — builds the final context for injection into agent prompts."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pymem.models import AgentContext, MemoryRecord, MemoryScope

if TYPE_CHECKING:
    from pymem.engine.episodic_manager import EpisodicMemoryManager
    from pymem.engine.procedural_manager import ProceduralMemoryManager
    from pymem.engine.semantic_manager import SemanticMemoryManager
    from pymem.engine.working_manager import WorkingMemoryManager
    from pymem.storage.base import EmbeddingPlugin


class ContextAssembler:
    """Assembles multi-type context for agents.

    Parallel fetch across all four memory types, then format into
    a structured prompt block ready for injection.
    """

    def __init__(
        self,
        working: WorkingMemoryManager,
        semantic: SemanticMemoryManager,
        episodic: EpisodicMemoryManager,
        procedural: ProceduralMemoryManager,
        embedding_provider: EmbeddingPlugin,
    ) -> None:
        self._working = working
        self._semantic = semantic
        self._episodic = episodic
        self._procedural = procedural
        self._embedder = embedding_provider

    async def assemble(
        self,
        scope: MemoryScope,
        query: str | None = None,
        max_tokens: int = 4000,
    ) -> AgentContext:
        """Parallel fetch across all four types, then format."""
        query_embedding = None
        if query:
            query_embedding = await self._embedder.embed(query)

        working_msgs, semantic, episodic, procedural = await asyncio.gather(
            self._working.get_window(scope, max_messages=20),
            self._semantic.search(query_embedding, scope, limit=8),
            self._episodic.get_recent(scope, limit=5),
            self._procedural.get_all(scope),
        )

        formatted = self._format_prompt_block(
            working_msgs, semantic, episodic, procedural
        )

        return AgentContext(
            working_messages=working_msgs,
            relevant_facts=semantic,
            recent_events=episodic,
            agent_instructions=procedural,
            formatted=formatted,
        )

    @staticmethod
    def _format_prompt_block(
        working: list[dict[str, Any]],
        semantic: list[MemoryRecord],
        episodic: list[MemoryRecord],
        procedural: list[MemoryRecord],
    ) -> str:
        """Format all memory types into a structured prompt block."""
        sections: list[str] = []

        if procedural:
            lines = "\n".join(f"- {m.content}" for m in procedural)
            sections.append(f"## Agent instructions (always follow)\n{lines}")

        if semantic:
            lines = "\n".join(f"- {m.content}" for m in semantic)
            sections.append(f"## What I know about you\n{lines}")

        if episodic:
            lines = "\n".join(f"- {m.content}" for m in episodic)
            sections.append(f"## Recent events\n{lines}")

        if working:
            lines = "\n".join(
                f"- {m.get('role', 'user')}: {m.get('content', '')}"
                for m in working[-5:]
            )
            sections.append(f"## Current conversation\n{lines}")

        return "\n\n".join(sections)
