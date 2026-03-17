"""Procedural memory manager — agent instructions, learned behaviours."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from pymem.models import MemoryRecord, MemoryScope, MemoryType
from pymem.storage.base import EmbeddingPlugin


class ProceduralMemoryManager:
    """Manages procedural memories — how to do things, agent instructions.

    Storage: Relational store (structured rows, priority-ordered).
    Retrieval: Exact match / tag-based + priority ordering.
    TTL: Never expires (manual deletion only).
    """

    def __init__(
        self,
        relational_store: Any,
        embedding_provider: EmbeddingPlugin,
    ) -> None:
        self._relational = relational_store
        self._embedder = embedding_provider

    async def add(
        self,
        content: str,
        scope: MemoryScope,
        tags: list[str] | None = None,
        priority: int = 5,
    ) -> MemoryRecord:
        """Add a procedural memory (agent instruction, workflow step)."""
        memory_id = str(uuid4())
        now = datetime.utcnow()
        scope_dict = scope.to_dict()

        record = MemoryRecord(
            id=memory_id,
            memory_type=MemoryType.PROCEDURAL,
            content=content,
            metadata={"tags": tags or [], "priority": priority},
            score=1.0,  # Procedural never decays
            created_at=now,
            updated_at=now,
            **scope_dict,
        )

        # Store in procedural table
        await self._relational.save_procedural({
            "id": memory_id,
            "content": content,
            "tags": tags or [],
            "priority": priority,
            "created_at": now.isoformat(),
            **scope_dict,
        })

        # Also store in general metadata table
        await self._relational.save_memory({
            "id": memory_id,
            "memory_type": "procedural",
            "content": content,
            "score": 1.0,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "metadata": {"tags": tags or [], "priority": priority},
            **scope_dict,
        })

        return record

    async def get_all(
        self,
        scope: MemoryScope,
        tags: list[str] | None = None,
    ) -> list[MemoryRecord]:
        """Get all procedural memories for a scope, ordered by priority."""
        scope_dict = scope.to_filter()
        rows = await self._relational.get_procedural(scope_dict, tags=tags)

        return [
            MemoryRecord(
                id=r["id"],
                memory_type=MemoryType.PROCEDURAL,
                content=r["content"],
                metadata={
                    "tags": json.loads(r["tags"]) if isinstance(r["tags"], str) else r.get("tags", []),
                    "priority": r.get("priority", 5),
                },
                score=1.0,
                agent_id=r.get("agent_id"),
                org_id=r.get("org_id"),
                user_id=r.get("user_id"),
            )
            for r in rows
        ]

    async def search(
        self,
        query: str,
        scope: MemoryScope,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        """Search procedural memories by content match."""
        all_memories = await self.get_all(scope)
        query_lower = query.lower()

        # Score by keyword overlap
        scored = []
        for mem in all_memories:
            content_lower = mem.content.lower()
            words = query_lower.split()
            matches = sum(1 for w in words if w in content_lower)
            score = matches / max(len(words), 1)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]

    async def delete_by_scope(self, scope: MemoryScope) -> int:
        """Delete all procedural memories for a scope."""
        scope_dict = scope.to_filter()
        proc_deleted = await self._relational.delete_procedural_by_scope(scope_dict)
        rel_deleted = await self._relational.delete_by_scope(
            scope_dict, memory_type="procedural"
        )
        return max(proc_deleted, rel_deleted)
