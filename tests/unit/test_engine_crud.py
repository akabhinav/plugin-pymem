"""Unit tests for MemoryEngine — get, update, delete, share operations."""

from __future__ import annotations

import pytest

from pymem.models import MemoryScope, MemoryType


@pytest.mark.asyncio
class TestEngineGet:
    async def test_get_existing_memory(self, engine, user_scope):
        """Engine.get() returns a MemoryRecord for a stored memory."""
        mem = await engine.semantic.add(
            content="User prefers dark mode",
            scope=user_scope,
            score=0.8,
        )
        result = await engine.get(mem.id)
        assert result is not None
        assert result.id == mem.id
        assert result.content == "User prefers dark mode"
        assert result.memory_type == MemoryType.SEMANTIC

    async def test_get_nonexistent_returns_none(self, engine):
        """Engine.get() returns None for unknown ID."""
        result = await engine.get("nonexistent-id-12345")
        assert result is None

    async def test_get_deleted_memory_returns_none(self, engine, user_scope):
        """Soft-deleted memories should not be returned by get()."""
        mem = await engine.semantic.add(
            content="Temporary fact",
            scope=user_scope,
        )
        await engine.delete(mem.id, user_scope)
        result = await engine.get(mem.id)
        assert result is None

    async def test_get_preserves_scope_fields(self, engine):
        """Engine.get() should return scope fields correctly."""
        scope = MemoryScope(user_id="u1", agent_id="a1")
        mem = await engine.semantic.add(
            content="Scoped memory",
            scope=scope,
            score=0.9,
        )
        result = await engine.get(mem.id)
        assert result is not None
        assert result.user_id == "u1"
        assert result.agent_id == "a1"


@pytest.mark.asyncio
class TestEngineUpdate:
    async def test_update_creates_new_version(self, engine, user_scope):
        """Update should create a new version with incremented version number."""
        original = await engine.semantic.add(
            content="Python is best",
            scope=user_scope,
        )
        updated = await engine.update(original.id, "Python and Rust are best", user_scope)
        assert updated is not None
        assert updated.id != original.id
        assert updated.content == "Python and Rust are best"
        assert updated.version == 2
        assert updated.parent_id == original.id

    async def test_update_soft_deletes_old_version(self, engine, user_scope):
        """Old version should be soft-deleted after update."""
        original = await engine.semantic.add(
            content="Old content",
            scope=user_scope,
        )
        await engine.update(original.id, "New content", user_scope)
        old = await engine.get(original.id)
        assert old is None  # soft-deleted

    async def test_update_nonexistent_returns_none(self, engine, user_scope):
        """Updating a nonexistent memory should return None."""
        result = await engine.update("nonexistent-id", "New content", user_scope)
        assert result is None

    async def test_update_semantic_updates_vector_store(self, engine, user_scope):
        """Updating a semantic memory should update the vector store."""
        original = await engine.semantic.add(
            content="Original semantic content",
            scope=user_scope,
        )
        updated = await engine.update(original.id, "Updated semantic content", user_scope)
        assert updated is not None

        # New ID should be in vector store
        vec_result = await engine._vector.get(updated.id)
        assert vec_result is not None
        assert vec_result.content == "Updated semantic content"

        # Old ID should be removed from vector store
        old_vec = await engine._vector.get(original.id)
        assert old_vec is None


@pytest.mark.asyncio
class TestEngineDelete:
    async def test_soft_delete(self, engine, user_scope):
        """Soft delete should make memory invisible to get()."""
        mem = await engine.semantic.add(content="To delete", scope=user_scope)
        await engine.delete(mem.id, user_scope, hard_delete=False)
        result = await engine.get(mem.id)
        assert result is None

    async def test_hard_delete_removes_from_all_backends(self, engine, user_scope):
        """Hard delete should remove from relational, vector, and graph."""
        mem = await engine.semantic.add(content="Hard delete target", scope=user_scope)
        await engine.delete(mem.id, user_scope, hard_delete=True)

        # Relational
        assert await engine.get(mem.id) is None
        # Vector
        assert await engine._vector.get(mem.id) is None
        # Graph
        assert await engine._graph.get_node(mem.id) is None

    async def test_delete_nonexistent_no_error(self, engine, user_scope):
        """Deleting a nonexistent memory should not raise."""
        await engine.delete("nonexistent-id", user_scope, hard_delete=True)


@pytest.mark.asyncio
class TestEngineShare:
    async def test_share_clones_to_new_scope(self, engine):
        """Share should clone memory into a different scope."""
        from_scope = MemoryScope(user_id="user_a")
        to_scope = MemoryScope(user_id="user_b")

        original = await engine.semantic.add(
            content="Shared knowledge",
            scope=from_scope,
        )

        shared = await engine.share(original.id, from_scope, to_scope)
        assert shared is not None
        assert shared.id != original.id
        assert shared.content == "Shared knowledge"
        assert shared.user_id == "user_b"

    async def test_share_nonexistent_returns_none(self, engine):
        """Sharing a nonexistent memory returns None."""
        result = await engine.share(
            "nonexistent",
            MemoryScope(user_id="a"),
            MemoryScope(user_id="b"),
        )
        assert result is None

    async def test_share_preserves_original(self, engine):
        """Original memory should still exist after sharing."""
        from_scope = MemoryScope(user_id="user_x")
        to_scope = MemoryScope(user_id="user_y")

        original = await engine.semantic.add(
            content="Still here after share",
            scope=from_scope,
        )
        await engine.share(original.id, from_scope, to_scope)

        # Original still accessible
        result = await engine.get(original.id)
        assert result is not None
        assert result.content == "Still here after share"
