"""SQLite relational store — default for dev/embedded use."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import aiosqlite


class SQLiteRelationalStore:
    """Lightweight async SQLite store for metadata, procedural memory, and audit."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._create_tables()

    async def _create_tables(self) -> None:
        assert self._conn
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS pymem_memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                score REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed_at TEXT,
                expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                parent_id TEXT,
                is_deleted INTEGER DEFAULT 0,
                deleted_at TEXT,
                source_type TEXT DEFAULT 'conversation',
                source_ref TEXT,
                user_id TEXT,
                agent_id TEXT,
                session_id TEXT,
                org_id TEXT,
                vector_indexed INTEGER DEFAULT 0,
                graph_indexed INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_pymem_memories_user
                ON pymem_memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_pymem_memories_agent
                ON pymem_memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_pymem_memories_type
                ON pymem_memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_pymem_memories_session
                ON pymem_memories(session_id);

            CREATE TABLE IF NOT EXISTS pymem_procedural (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                priority INTEGER DEFAULT 5,
                agent_id TEXT,
                org_id TEXT,
                user_id TEXT,
                created_at TEXT NOT NULL,
                is_deleted INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS pymem_audit (
                id TEXT PRIMARY KEY,
                memory_id TEXT,
                operation TEXT NOT NULL,
                actor_id TEXT,
                scope TEXT DEFAULT '{}',
                before_state TEXT,
                after_state TEXT,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pymem_user_stats (
                user_id TEXT PRIMARY KEY,
                org_id TEXT,
                total_memories INTEGER DEFAULT 0,
                semantic_count INTEGER DEFAULT 0,
                episodic_count INTEGER DEFAULT 0,
                procedural_count INTEGER DEFAULT 0,
                working_count INTEGER DEFAULT 0,
                last_extraction_at TEXT,
                updated_at TEXT
            );
        """)
        await self._conn.commit()

    # --- Memory metadata CRUD ---

    async def save_memory(self, record: dict[str, Any]) -> None:
        assert self._conn
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO pymem_memories
            (id, memory_type, content, score, access_count, last_accessed_at,
             expires_at, created_at, updated_at, version, parent_id,
             is_deleted, deleted_at, source_type, source_ref,
             user_id, agent_id, session_id, org_id,
             vector_indexed, graph_indexed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["memory_type"],
                record["content"],
                record.get("score", 0.5),
                record.get("access_count", 0),
                record.get("last_accessed_at"),
                record.get("expires_at"),
                record.get("created_at", datetime.utcnow().isoformat()),
                record.get("updated_at", datetime.utcnow().isoformat()),
                record.get("version", 1),
                record.get("parent_id"),
                int(record.get("is_deleted", False)),
                record.get("deleted_at"),
                record.get("source_type", "conversation"),
                record.get("source_ref"),
                record.get("user_id"),
                record.get("agent_id"),
                record.get("session_id"),
                record.get("org_id"),
                int(record.get("vector_indexed", False)),
                int(record.get("graph_indexed", False)),
                json.dumps(record.get("metadata", {})),
            ),
        )
        await self._conn.commit()

    async def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        assert self._conn
        cursor = await self._conn.execute(
            "SELECT * FROM pymem_memories WHERE id = ? AND is_deleted = 0",
            (memory_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(row)

    async def list_memories(
        self,
        filters: dict[str, Any],
        memory_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        assert self._conn
        where_parts = ["is_deleted = 0"]
        params: list[Any] = []

        for field in ("user_id", "agent_id", "session_id", "org_id"):
            if field in filters and filters[field] is not None:
                where_parts.append(f"{field} = ?")
                params.append(filters[field])

        if memory_type:
            where_parts.append("memory_type = ?")
            params.append(memory_type)

        where_clause = " AND ".join(where_parts)
        params.extend([limit, offset])

        cursor = await self._conn.execute(
            f"SELECT * FROM pymem_memories WHERE {where_clause} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def soft_delete_memory(self, memory_id: str) -> None:
        assert self._conn
        await self._conn.execute(
            "UPDATE pymem_memories SET is_deleted = 1, deleted_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), memory_id),
        )
        await self._conn.commit()

    async def hard_delete_memory(self, memory_id: str) -> None:
        assert self._conn
        await self._conn.execute("DELETE FROM pymem_memories WHERE id = ?", (memory_id,))
        await self._conn.commit()

    async def delete_by_scope(
        self, filters: dict[str, Any], memory_type: str | None = None
    ) -> int:
        assert self._conn
        where_parts: list[str] = []
        params: list[Any] = []

        for field in ("user_id", "agent_id", "session_id", "org_id"):
            if field in filters and filters[field] is not None:
                where_parts.append(f"{field} = ?")
                params.append(filters[field])

        if memory_type:
            where_parts.append("memory_type = ?")
            params.append(memory_type)

        if not where_parts:
            return 0

        where_clause = " AND ".join(where_parts)
        cursor = await self._conn.execute(
            f"SELECT COUNT(*) FROM pymem_memories WHERE {where_clause}", params
        )
        row = await cursor.fetchone()
        count = row[0] if row else 0

        await self._conn.execute(
            f"DELETE FROM pymem_memories WHERE {where_clause}", params
        )
        await self._conn.commit()
        return count

    async def update_access(self, memory_id: str) -> None:
        assert self._conn
        await self._conn.execute(
            """
            UPDATE pymem_memories
            SET access_count = access_count + 1,
                last_accessed_at = ?
            WHERE id = ?
            """,
            (datetime.utcnow().isoformat(), memory_id),
        )
        await self._conn.commit()

    async def update_score(self, memory_id: str, new_score: float) -> None:
        assert self._conn
        await self._conn.execute(
            "UPDATE pymem_memories SET score = ?, updated_at = ? WHERE id = ?",
            (new_score, datetime.utcnow().isoformat(), memory_id),
        )
        await self._conn.commit()

    # --- Procedural memory ---

    async def save_procedural(self, record: dict[str, Any]) -> None:
        assert self._conn
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO pymem_procedural
            (id, content, tags, priority, agent_id, org_id, user_id, created_at, is_deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                record["id"],
                record["content"],
                json.dumps(record.get("tags", [])),
                record.get("priority", 5),
                record.get("agent_id"),
                record.get("org_id"),
                record.get("user_id"),
                record.get("created_at", datetime.utcnow().isoformat()),
            ),
        )
        await self._conn.commit()

    async def get_procedural(
        self, filters: dict[str, Any], tags: list[str] | None = None
    ) -> list[dict[str, Any]]:
        assert self._conn
        where_parts = ["is_deleted = 0"]
        params: list[Any] = []

        for field in ("agent_id", "org_id", "user_id"):
            if field in filters and filters[field] is not None:
                where_parts.append(f"{field} = ?")
                params.append(filters[field])

        where_clause = " AND ".join(where_parts)
        cursor = await self._conn.execute(
            f"SELECT * FROM pymem_procedural WHERE {where_clause} ORDER BY priority ASC",
            params,
        )
        rows = await cursor.fetchall()
        results = [dict(r) for r in rows]

        if tags:
            results = [
                r
                for r in results
                if any(t in json.loads(r.get("tags", "[]")) for t in tags)
            ]
        return results

    async def delete_procedural_by_scope(self, filters: dict[str, Any]) -> int:
        assert self._conn
        where_parts: list[str] = []
        params: list[Any] = []

        for field in ("agent_id", "org_id", "user_id"):
            if field in filters and filters[field] is not None:
                where_parts.append(f"{field} = ?")
                params.append(filters[field])

        if not where_parts:
            return 0

        where_clause = " AND ".join(where_parts)
        cursor = await self._conn.execute(
            f"SELECT COUNT(*) FROM pymem_procedural WHERE {where_clause}", params
        )
        row = await cursor.fetchone()
        count = row[0] if row else 0
        await self._conn.execute(
            f"DELETE FROM pymem_procedural WHERE {where_clause}", params
        )
        await self._conn.commit()
        return count

    # --- Audit log ---

    async def log_audit(
        self,
        audit_id: str,
        memory_id: str,
        operation: str,
        actor_id: str,
        scope: dict[str, Any],
        before_state: str | None = None,
        after_state: str | None = None,
    ) -> None:
        assert self._conn
        await self._conn.execute(
            """
            INSERT INTO pymem_audit
            (id, memory_id, operation, actor_id, scope, before_state, after_state, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                memory_id,
                operation,
                actor_id,
                json.dumps(scope),
                before_state,
                after_state,
                datetime.utcnow().isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_audit_log(
        self,
        user_id: str | None = None,
        memory_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        assert self._conn
        where_parts: list[str] = []
        params: list[Any] = []

        if user_id:
            where_parts.append("scope LIKE ?")
            params.append(f'%"user_id": "{user_id}"%')
        if memory_id:
            where_parts.append("memory_id = ?")
            params.append(memory_id)

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        params.append(limit)

        cursor = await self._conn.execute(
            f"SELECT * FROM pymem_audit WHERE {where_clause} ORDER BY timestamp DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # --- User stats ---

    async def update_user_stats(self, user_id: str, memory_type: str, delta: int = 1) -> None:
        assert self._conn
        type_col = f"{memory_type}_count"
        await self._conn.execute(
            f"""
            INSERT INTO pymem_user_stats (user_id, total_memories, {type_col}, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                total_memories = total_memories + ?,
                {type_col} = {type_col} + ?,
                updated_at = ?
            """,
            (
                user_id,
                delta,
                delta,
                datetime.utcnow().isoformat(),
                delta,
                delta,
                datetime.utcnow().isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_user_stats(self, user_id: str) -> dict[str, Any] | None:
        assert self._conn
        cursor = await self._conn.execute(
            "SELECT * FROM pymem_user_stats WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_platform_stats(self) -> dict[str, Any]:
        assert self._conn
        cursor = await self._conn.execute(
            """
            SELECT
                COUNT(*) as total_memories,
                SUM(CASE WHEN memory_type='semantic' THEN 1 ELSE 0 END) as semantic,
                SUM(CASE WHEN memory_type='episodic' THEN 1 ELSE 0 END) as episodic,
                SUM(CASE WHEN memory_type='procedural' THEN 1 ELSE 0 END) as procedural,
                SUM(CASE WHEN memory_type='working' THEN 1 ELSE 0 END) as working,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT agent_id) as unique_agents
            FROM pymem_memories WHERE is_deleted = 0
            """
        )
        row = await cursor.fetchone()
        return dict(row) if row else {}

    async def shutdown(self) -> None:
        if self._conn:
            await self._conn.close()
