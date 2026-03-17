"""Semantic memory manager — facts, preferences, entities stored in vector + graph."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pymem.models import MemoryRecord, MemoryScope, MemoryType
from pymem.storage.base import (
    EmbeddingPlugin,
    GraphStorePlugin,
    VectorStorePlugin,
)


class SemanticMemoryManager:
    """Manages semantic memories — facts, beliefs, preferences, entities.

    Storage: Vector store (embeddings) + graph store (entity relationships).
    Retrieval: Similarity search + reranking.
    TTL: Decay by access frequency.
    """

    def __init__(
        self,
        vector_store: VectorStorePlugin,
        graph_store: GraphStorePlugin,
        relational_store: Any,
        embedding_provider: EmbeddingPlugin,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._relational = relational_store
        self._embedder = embedding_provider

    async def add(
        self,
        content: str,
        scope: MemoryScope,
        score: float = 0.5,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> MemoryRecord:
        """Add a semantic memory (fact, preference, entity)."""
        memory_id = memory_id or str(uuid4())
        now = datetime.utcnow()
        scope_dict = scope.to_dict()

        # Compute embedding
        embedding = await self._embedder.embed(content)

        record = MemoryRecord(
            id=memory_id,
            memory_type=MemoryType.SEMANTIC,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            score=score,
            created_at=now,
            updated_at=now,
            source_type="conversation",
            **scope_dict,
        )

        # Store in vector DB
        await self._vector.upsert(
            memory_id=memory_id,
            embedding=embedding,
            content=content,
            metadata={**scope_dict, **(metadata or {})},
        )

        # Store metadata in relational
        await self._relational.save_memory({
            "id": memory_id,
            "memory_type": "semantic",
            "content": content,
            "score": score,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "vector_indexed": True,
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
        """Search semantic memories by vector similarity."""
        if query_embedding is None:
            # Fall back to relational listing
            rows = await self._relational.list_memories(
                filters=scope.to_filter(),
                memory_type="semantic",
                limit=limit,
            )
            return [self._row_to_record(r) for r in rows]

        results = await self._vector.search(
            query_embedding=query_embedding,
            filters=scope.to_filter(),
            limit=limit,
            min_score=-1.0,  # Allow negative cosine scores (hash-based embeddings)
        )

        # Fallback to relational listing if vector search returns nothing
        if not results:
            rows = await self._relational.list_memories(
                filters=scope.to_filter(),
                memory_type="semantic",
                limit=limit,
            )
            return [self._row_to_record(r) for r in rows]

        return [
            MemoryRecord(
                id=r.memory_id,
                memory_type=MemoryType.SEMANTIC,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
                **{k: r.metadata.get(k) for k in ("user_id", "agent_id", "session_id", "org_id") if r.metadata.get(k)},
            )
            for r in results
        ]

    async def get_similar(
        self,
        content: str,
        scope: MemoryScope,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Find memories similar to given content."""
        embedding = await self._embedder.embed(content)
        return await self.search(embedding, scope, limit)

    async def delete_by_scope(self, scope: MemoryScope) -> int:
        """Delete all semantic memories for a scope."""
        scope_dict = scope.to_filter()
        vector_deleted = await self._vector.delete_by_filter(scope_dict)
        rel_deleted = await self._relational.delete_by_scope(
            scope_dict, memory_type="semantic"
        )
        return max(vector_deleted, rel_deleted)

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            memory_type=MemoryType.SEMANTIC,
            content=row["content"],
            score=row.get("score", 0.5),
            metadata=row.get("metadata", {}),
            user_id=row.get("user_id"),
            agent_id=row.get("agent_id"),
            session_id=row.get("session_id"),
            org_id=row.get("org_id"),
        )
