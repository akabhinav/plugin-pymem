"""Integration test for SQLite relational store."""

import pytest

from pymem.storage.relational.sqlite_store import SQLiteRelationalStore


@pytest.mark.asyncio
class TestSQLiteRelationalStore:
    async def test_save_and_get_memory(self, relational_store):
        await relational_store.save_memory({
            "id": "test-1",
            "memory_type": "semantic",
            "content": "Test memory content",
            "score": 0.8,
            "user_id": "u1",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        })

        result = await relational_store.get_memory("test-1")
        assert result is not None
        assert result["content"] == "Test memory content"
        assert result["memory_type"] == "semantic"

    async def test_list_memories_by_user(self, relational_store):
        for i in range(5):
            await relational_store.save_memory({
                "id": f"list-{i}",
                "memory_type": "semantic",
                "content": f"Memory {i}",
                "user_id": "list_user",
                "created_at": "2024-01-01",
                "updated_at": "2024-01-01",
            })

        results = await relational_store.list_memories(
            filters={"user_id": "list_user"}, limit=3
        )
        assert len(results) == 3

    async def test_soft_delete(self, relational_store):
        await relational_store.save_memory({
            "id": "del-1",
            "memory_type": "semantic",
            "content": "To be deleted",
            "user_id": "u1",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        })

        await relational_store.soft_delete_memory("del-1")
        result = await relational_store.get_memory("del-1")
        assert result is None  # Soft deleted, not visible

    async def test_procedural_memory(self, relational_store):
        await relational_store.save_procedural({
            "id": "proc-1",
            "content": "Always test first",
            "tags": ["testing"],
            "priority": 1,
            "agent_id": "a1",
            "created_at": "2024-01-01",
        })

        results = await relational_store.get_procedural(
            filters={"agent_id": "a1"}
        )
        assert len(results) == 1
        assert results[0]["content"] == "Always test first"

    async def test_audit_log(self, relational_store):
        await relational_store.log_audit(
            audit_id="audit-1",
            memory_id="mem-1",
            operation="add",
            actor_id="user-1",
            scope={"user_id": "user-1"},
        )

        logs = await relational_store.get_audit_log(memory_id="mem-1")
        assert len(logs) == 1
        assert logs[0]["operation"] == "add"

    async def test_platform_stats(self, relational_store):
        await relational_store.save_memory({
            "id": "stat-1",
            "memory_type": "semantic",
            "content": "Test",
            "user_id": "u1",
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        })

        stats = await relational_store.get_platform_stats()
        assert stats["total_memories"] >= 1
