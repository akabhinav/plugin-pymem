"""In-memory vector store — development fallback when pgvector is unavailable."""

from __future__ import annotations

from typing import Any

from pymem.storage.base import VectorSearchResult, VectorStorePlugin
from pymem.storage.registry import register_vector_store

_SCOPE_FIELDS = ("user_id", "agent_id", "session_id", "org_id")


@register_vector_store("memory")
class InMemoryVectorStore(VectorStorePlugin):
    """Simple in-memory vector store for dev/test. Brute-force cosine similarity."""

    def __init__(self, **kwargs: Any) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        pass

    async def upsert(
        self,
        memory_id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        self._records[memory_id] = {
            "id": memory_id,
            "embedding": embedding,
            "content": content,
            "metadata": metadata,
        }

    async def search(
        self,
        query_embedding: list[float],
        filters: dict[str, Any],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        results: list[tuple[float, dict]] = []

        for record in self._records.values():
            # Check scope filters
            meta = record["metadata"]
            if not self._matches_filters(meta, filters):
                continue

            score = self._cosine_similarity(query_embedding, record["embedding"])
            if score >= min_score:
                results.append((score, record))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            VectorSearchResult(
                memory_id=r["id"],
                content=r["content"],
                score=score,
                metadata=r["metadata"],
            )
            for score, r in results[:limit]
        ]

    async def delete(self, memory_id: str) -> None:
        self._records.pop(memory_id, None)

    async def delete_by_filter(self, filters: dict[str, Any]) -> int:
        to_delete = [
            mid
            for mid, r in self._records.items()
            if self._matches_filters(r["metadata"], filters)
        ]
        for mid in to_delete:
            del self._records[mid]
        return len(to_delete)

    async def get(self, memory_id: str) -> VectorSearchResult | None:
        record = self._records.get(memory_id)
        if not record:
            return None
        return VectorSearchResult(
            memory_id=record["id"],
            content=record["content"],
            score=1.0,
            metadata=record["metadata"],
        )

    async def count(self, filters: dict[str, Any]) -> int:
        return sum(
            1
            for r in self._records.values()
            if self._matches_filters(r["metadata"], filters)
        )

    async def health_check(self) -> bool:
        return True

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        for field in _SCOPE_FIELDS:
            if field in filters and filters[field] is not None:
                if metadata.get(field) != filters[field]:
                    return False
        return True

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
