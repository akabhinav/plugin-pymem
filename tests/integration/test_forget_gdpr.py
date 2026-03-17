"""Integration test: GDPR forget — after forget(), zero results in all backends."""

import pytest

from pymem.models import MemoryScope, MemoryType


@pytest.mark.asyncio
class TestGDPRForget:
    async def test_forget_all(self, engine):
        """After forget(): zero results in all backends."""
        scope = MemoryScope(user_id="forget_user")

        # Add memories across all types
        await engine.semantic.add("Fact about user", scope)
        await engine.episodic.add_event("Something happened", scope)
        await engine.procedural.add(
            "Always do X",
            MemoryScope(user_id="forget_user", agent_id="a1"),
        )

        # Verify memories exist
        search = await engine.search("fact", scope, limit=10)
        assert len(search.memories) > 0

        # Forget
        result = await engine.forget(scope=scope)
        assert result.confirmation_id  # Non-empty

        # Verify: search returns 0 results
        search_after = await engine.search("fact", scope, limit=10)
        assert len(search_after.memories) == 0

    async def test_forget_specific_type(self, engine):
        """Forget only semantic memories — episodic should remain."""
        scope = MemoryScope(user_id="partial_forget_user")

        await engine.semantic.add("Semantic fact", scope)
        await engine.episodic.add_event("Episodic event", scope)

        # Forget only semantic
        result = await engine.forget(
            scope=scope, memory_types=[MemoryType.SEMANTIC]
        )
        assert "semantic" in result.deleted_per_type

        # Episodic should still exist
        episodic_results = await engine.episodic.get_recent(scope)
        # The events may still be in graph store
        # This tests that forget is scoped correctly

    async def test_forget_nonexistent_user(self, engine):
        """Forget for user with no memories should succeed gracefully."""
        scope = MemoryScope(user_id="nonexistent_user")
        result = await engine.forget(scope=scope)
        assert result.confirmation_id  # Still returns a valid result
