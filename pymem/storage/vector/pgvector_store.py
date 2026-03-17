"""pgvector — Default vector store using PostgreSQL + pgvector extension."""

from __future__ import annotations

import json
from typing import Any

from pymem.storage.base import VectorSearchResult, VectorStorePlugin
from pymem.storage.registry import register_vector_store

_SCOPE_FIELDS = ("user_id", "agent_id", "session_id", "org_id")


@register_vector_store("pgvector")
class PgVectorStore(VectorStorePlugin):
    """PostgreSQL + pgvector. Table: pymem_vectors.

    Index: ivfflat on embedding for approximate nearest neighbour.
    Scope filtering via dedicated columns + metadata jsonb.
    """

    def __init__(self, dsn: str | None = None, pool: Any = None) -> None:
        self._dsn = dsn
        self._pool = pool

    async def initialize(self) -> None:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)

        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS pymem_vectors (
                    id TEXT PRIMARY KEY,
                    embedding vector(384),
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    user_id TEXT,
                    agent_id TEXT,
                    session_id TEXT,
                    org_id TEXT,
                    is_deleted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pymem_vectors_embedding
                ON pymem_vectors USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            for field in _SCOPE_FIELDS:
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_pymem_vectors_{field} "
                    f"ON pymem_vectors ({field})"
                )

    async def upsert(
        self,
        memory_id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO pymem_vectors (id, embedding, content, metadata,
                    user_id, agent_id, session_id, org_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                memory_id,
                str(embedding),
                content,
                json.dumps(metadata),
                metadata.get("user_id"),
                metadata.get("agent_id"),
                metadata.get("session_id"),
                metadata.get("org_id"),
            )

    async def search(
        self,
        query_embedding: list[float],
        filters: dict[str, Any],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        where_parts = ["is_deleted = FALSE"]
        params: list[Any] = [str(query_embedding)]
        idx = 2

        for field in _SCOPE_FIELDS:
            if field in filters and filters[field] is not None:
                where_parts.append(f"{field} = ${idx}")
                params.append(filters[field])
                idx += 1

        params.append(limit)
        where_clause = " AND ".join(where_parts)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, content, metadata,
                       1 - (embedding <=> $1) AS score
                FROM pymem_vectors
                WHERE {where_clause}
                ORDER BY embedding <=> $1
                LIMIT ${idx}
                """,
                *params,
            )

        return [
            VectorSearchResult(
                memory_id=r["id"],
                content=r["content"],
                score=float(r["score"]),
                metadata=json.loads(r["metadata"]) if r["metadata"] else {},
            )
            for r in rows
            if float(r["score"]) >= min_score
        ]

    async def delete(self, memory_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM pymem_vectors WHERE id = $1", memory_id)

    async def delete_by_filter(self, filters: dict[str, Any]) -> int:
        where_parts: list[str] = []
        params: list[Any] = []
        idx = 1

        for field in _SCOPE_FIELDS:
            if field in filters and filters[field] is not None:
                where_parts.append(f"{field} = ${idx}")
                params.append(filters[field])
                idx += 1

        if not where_parts:
            return 0

        where_clause = " AND ".join(where_parts)
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM pymem_vectors WHERE {where_clause}", *params
            )
            return int(result.split()[-1])

    async def get(self, memory_id: str) -> VectorSearchResult | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, content, metadata FROM pymem_vectors WHERE id = $1",
                memory_id,
            )
        if not row:
            return None
        return VectorSearchResult(
            memory_id=row["id"],
            content=row["content"],
            score=1.0,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def count(self, filters: dict[str, Any]) -> int:
        where_parts = ["is_deleted = FALSE"]
        params: list[Any] = []
        idx = 1

        for field in _SCOPE_FIELDS:
            if field in filters and filters[field] is not None:
                where_parts.append(f"{field} = ${idx}")
                params.append(filters[field])
                idx += 1

        where_clause = " AND ".join(where_parts)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT COUNT(*) as cnt FROM pymem_vectors WHERE {where_clause}",
                *params,
            )
            return row["cnt"] if row else 0

    async def health_check(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        if self._pool:
            await self._pool.close()
