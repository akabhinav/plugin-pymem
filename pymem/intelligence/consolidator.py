"""Memory consolidator — detects and resolves duplicate or conflicting memories."""

from __future__ import annotations

from pymem.intelligence.pygate_client import PyGateClient
from pymem.models import ConsolidationResult, MemoryRecord

CONSOLIDATION_PROMPT = """You are a memory deduplication specialist.

Given a NEW memory and a list of EXISTING similar memories, decide:
- "keep_new": new memory adds information not in existing ones
- "merge": combine new and existing into one improved memory
- "discard": new memory is already covered by existing ones
- "supersede": new memory contradicts/updates an existing one

Return JSON: {"action": "keep_new"|"merge"|"discard"|"supersede",
              "merged_content": "...",
              "superseded_id": "...",
              "reason": "..."}"""


class MemoryConsolidator:
    """Detects and resolves duplicate or conflicting memories.

    Lightweight mode: compare new memory against top-5 similar existing ones.
    If similarity > 0.92: merge or discard.
    If conflict detected: mark old as superseded.
    """

    def __init__(self, pygate: PyGateClient) -> None:
        self._pygate = pygate

    async def consolidate(
        self,
        new_memory: MemoryRecord,
        existing_similar: list[MemoryRecord],
    ) -> ConsolidationResult:
        """Decide how to handle a new memory given similar existing ones."""
        if not existing_similar:
            return ConsolidationResult(action="keep_new", memory=new_memory)

        existing_text = "\n".join(
            f"[{m.id}] {m.content}" for m in existing_similar
        )

        result = await self._pygate.complete_json(
            system=CONSOLIDATION_PROMPT,
            user=f"NEW MEMORY: {new_memory.content}\n\nEXISTING:\n{existing_text}",
        )

        return ConsolidationResult.from_llm_response(
            result, new_memory, existing_similar
        )

    async def consolidate_simple(
        self,
        new_memory: MemoryRecord,
        existing_similar: list[MemoryRecord],
    ) -> ConsolidationResult:
        """Simple consolidation without LLM — keeps new if not exact duplicate."""
        if not existing_similar:
            return ConsolidationResult(action="keep_new", memory=new_memory)

        for existing in existing_similar:
            if existing.content.strip().lower() == new_memory.content.strip().lower():
                return ConsolidationResult(
                    action="discard",
                    memory=new_memory,
                    reason="Exact duplicate",
                )

        return ConsolidationResult(action="keep_new", memory=new_memory)
