"""Episodic memory manager — temporal events stored in graph + relational."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pymem.models import MemoryRecord, MemoryScope, MemoryType
from pymem.storage.base import EmbeddingPlugin, GraphNode, GraphStorePlugin


class EpisodicMemoryManager:
    """Manages episodic memories — what happened, when, with whom.

    Storage: Graph store (temporal chains) + relational (metadata).
    Retrieval: Temporal query + graph traversal.
    TTL: Configurable (default 90 days).
    """

    def __init__(
        self,
        graph_store: GraphStorePlugin,
        relational_store: Any,
        embedding_provider: EmbeddingPlugin,
    ) -> None:
        self._graph = graph_store
        self._relational = relational_store
        self._embedder = embedding_provider

    async def add_event(
        self,
        content: str,
        scope: MemoryScope,
        source_type: str = "conversation",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Add a timestamped event to episodic memory."""
        memory_id = str(uuid4())
        now = datetime.utcnow()
        scope_dict = scope.to_dict()

        record = MemoryRecord(
            id=memory_id,
            memory_type=MemoryType.EPISODIC,
            content=content,
            metadata=metadata or {},
            score=0.7,
            created_at=now,
            updated_at=now,
            source_type=source_type,
            **scope_dict,
        )

        # Store in graph as Event node
        node = GraphNode(
            id=memory_id,
            label="Event",
            properties={
                "content": content,
                "created_at": now.isoformat(),
                **(metadata or {}),
            },
            scope=scope_dict,
        )
        await self._graph.upsert_node(node)

        # Store metadata in relational
        await self._relational.save_memory({
            "id": memory_id,
            "memory_type": "episodic",
            "content": content,
            "score": record.score,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "source_type": source_type,
            "graph_indexed": True,
            "metadata": metadata or {},
            **scope_dict,
        })

        return record

    async def search(
        self,
        query_embedding: list[float] | None,
        scope: MemoryScope,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        """Search episodic memories by timeline."""
        scope_dict = scope.to_dict()
        events = await self._graph.get_episodic_timeline(
            scope=scope_dict, limit=limit
        )
        return [
            MemoryRecord(
                id=event.id,
                memory_type=MemoryType.EPISODIC,
                content=event.properties.get("content", ""),
                metadata=event.properties,
                score=0.7,
                **{k: v for k, v in event.scope.items() if v},
            )
            for event in events
        ]

    async def get_recent(
        self,
        scope: MemoryScope,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Get most recent episodic memories."""
        return await self.search(None, scope, limit)

    async def delete_by_scope(self, scope: MemoryScope) -> int:
        """Delete all episodic memories for a scope."""
        scope_dict = scope.to_dict()
        graph_deleted = await self._graph.delete_by_scope(scope_dict)
        rel_deleted = await self._relational.delete_by_scope(
            scope_dict, memory_type="episodic"
        )
        return max(graph_deleted, rel_deleted)
