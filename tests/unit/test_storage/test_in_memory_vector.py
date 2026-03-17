"""Tests for in-memory vector store."""

import pytest

from pymem.storage.vector.in_memory_vector import InMemoryVectorStore


@pytest.fixture
async def store():
    s = InMemoryVectorStore()
    await s.initialize()
    return s


@pytest.mark.asyncio
class TestInMemoryVectorStore:
    async def test_upsert_and_get(self, store):
        await store.upsert(
            memory_id="m1",
            embedding=[0.1, 0.2, 0.3],
            content="test content",
            metadata={"user_id": "u1"},
        )
        result = await store.get("m1")
        assert result is not None
        assert result.memory_id == "m1"
        assert result.content == "test content"

    async def test_get_nonexistent(self, store):
        result = await store.get("nonexistent")
        assert result is None

    async def test_search_by_similarity(self, store):
        await store.upsert("m1", [1.0, 0.0, 0.0], "python", {"user_id": "u1"})
        await store.upsert("m2", [0.0, 1.0, 0.0], "java", {"user_id": "u1"})
        await store.upsert("m3", [0.9, 0.1, 0.0], "python 3", {"user_id": "u1"})

        results = await store.search(
            query_embedding=[1.0, 0.0, 0.0],
            filters={"user_id": "u1"},
            limit=2,
        )
        assert len(results) == 2
        assert results[0].memory_id == "m1"  # Exact match
        assert results[0].score > results[1].score

    async def test_search_with_scope_filter(self, store):
        await store.upsert("m1", [1.0, 0.0], "content", {"user_id": "u1"})
        await store.upsert("m2", [1.0, 0.0], "content", {"user_id": "u2"})

        results = await store.search([1.0, 0.0], filters={"user_id": "u1"})
        assert len(results) == 1
        assert results[0].memory_id == "m1"

    async def test_delete(self, store):
        await store.upsert("m1", [1.0], "test", {"user_id": "u1"})
        await store.delete("m1")
        result = await store.get("m1")
        assert result is None

    async def test_delete_by_filter(self, store):
        await store.upsert("m1", [1.0], "a", {"user_id": "u1"})
        await store.upsert("m2", [1.0], "b", {"user_id": "u1"})
        await store.upsert("m3", [1.0], "c", {"user_id": "u2"})

        deleted = await store.delete_by_filter({"user_id": "u1"})
        assert deleted == 2
        assert await store.count({"user_id": "u1"}) == 0
        assert await store.count({"user_id": "u2"}) == 1

    async def test_count(self, store):
        await store.upsert("m1", [1.0], "a", {"user_id": "u1"})
        await store.upsert("m2", [1.0], "b", {"user_id": "u1"})
        assert await store.count({"user_id": "u1"}) == 2
        assert await store.count({"user_id": "u2"}) == 0

    async def test_health_check(self, store):
        assert await store.health_check() is True
