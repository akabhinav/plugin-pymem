"""Qdrant vector store plugin."""

from __future__ import annotations

from typing import Any

from pymem.storage.base import VectorSearchResult, VectorStorePlugin
from pymem.storage.registry import register_vector_store

_SCOPE_FIELDS = ("user_id", "agent_id", "session_id", "org_id")


@register_vector_store("qdrant")
class QdrantStore(VectorStorePlugin):
    """Qdrant cloud or self-hosted. Collection: pymem."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "pymem",
        dimension: int = 384,
    ) -> None:
        self._url = url
        self._collection = collection_name
        self._dimension = dimension
        self._client: Any = None

    async def initialize(self) -> None:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._client = AsyncQdrantClient(url=self._url)

        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimension, distance=Distance.COSINE
                ),
            )

    async def upsert(
        self,
        memory_id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        from qdrant_client.models import PointStruct

        await self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={"content": content, **metadata},
                )
            ],
        )

    async def search(
        self,
        query_embedding: list[float],
        filters: dict[str, Any],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        for field in _SCOPE_FIELDS:
            if field in filters and filters[field] is not None:
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=filters[field]))
                )

        qdrant_filter = Filter(must=conditions) if conditions else None

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=limit,
            score_threshold=min_score,
        )

        return [
            VectorSearchResult(
                memory_id=str(r.id),
                content=r.payload.get("content", ""),
                score=r.score,
                metadata=r.payload,
            )
            for r in results
        ]

    async def delete(self, memory_id: str) -> None:
        from qdrant_client.models import PointIdsList

        await self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[memory_id]),
        )

    async def delete_by_filter(self, filters: dict[str, Any]) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        for field in _SCOPE_FIELDS:
            if field in filters and filters[field] is not None:
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=filters[field]))
                )

        if not conditions:
            return 0

        before = await self.count(filters)
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(must=conditions),
        )
        return before

    async def get(self, memory_id: str) -> VectorSearchResult | None:
        results = await self._client.retrieve(
            collection_name=self._collection,
            ids=[memory_id],
            with_payload=True,
        )
        if not results:
            return None
        r = results[0]
        return VectorSearchResult(
            memory_id=str(r.id),
            content=r.payload.get("content", ""),
            score=1.0,
            metadata=r.payload,
        )

    async def count(self, filters: dict[str, Any]) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        for field in _SCOPE_FIELDS:
            if field in filters and filters[field] is not None:
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=filters[field]))
                )

        qdrant_filter = Filter(must=conditions) if conditions else None
        result = await self._client.count(
            collection_name=self._collection, count_filter=qdrant_filter
        )
        return result.count

    async def health_check(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()
