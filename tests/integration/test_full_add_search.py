"""Integration test: full add → search → verify pipeline."""

import pytest

from pymem.models import MemoryScope, MemoryType


@pytest.mark.asyncio
class TestFullAddSearch:
    async def test_add_and_search_semantic(self, engine, user_scope):
        """Add messages → extract → search → verify retrieval."""
        messages = [
            {"role": "user", "content": "I have 15 years of Java experience"},
            {"role": "assistant", "content": "That's impressive!"},
            {"role": "user", "content": "I prefer Python over Java for new projects"},
        ]

        result = await engine.add(messages=messages, scope=user_scope)
        assert result.status == "stored"
        assert result.memories_count > 0

        # Search should find related memories
        search_result = await engine.search(
            query="programming experience",
            scope=user_scope,
            limit=5,
        )
        assert len(search_result.memories) > 0

    async def test_add_and_search_with_agent_scope(self, engine):
        """Memories scoped to agent are retrievable by agent."""
        scope = MemoryScope(user_id="u1", agent_id="agent_1")
        messages = [
            {"role": "user", "content": "Always format output as markdown tables"},
        ]

        await engine.add(messages=messages, scope=scope)

        # Search with same scope
        result = await engine.search(
            query="formatting",
            scope=scope,
            limit=5,
        )
        assert len(result.memories) > 0

    async def test_different_users_isolated(self, engine):
        """Memories from user1 should not appear in user2's search."""
        scope1 = MemoryScope(user_id="user_1")
        scope2 = MemoryScope(user_id="user_2")

        await engine.semantic.add(
            content="User 1 likes pizza",
            scope=scope1,
        )
        await engine.semantic.add(
            content="User 2 likes sushi",
            scope=scope2,
        )

        result1 = await engine.search("food preference", scope=scope1, limit=5)
        result2 = await engine.search("food preference", scope=scope2, limit=5)

        # Each user should only see their own memories
        for mem in result1.memories:
            assert "sushi" not in mem.content.lower()
        for mem in result2.memories:
            assert "pizza" not in mem.content.lower()


@pytest.mark.asyncio
class TestFourMemoryTypes:
    async def test_semantic_memory(self, engine, user_scope):
        mem = await engine.semantic.add(
            content="User prefers Python over Go",
            scope=user_scope,
            score=0.8,
        )
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.content == "User prefers Python over Go"

    async def test_episodic_memory(self, engine, user_scope):
        mem = await engine.episodic.add_event(
            content="User asked about Kafka on March 14",
            scope=user_scope,
        )
        assert mem.memory_type == MemoryType.EPISODIC

    async def test_procedural_memory(self, engine):
        scope = MemoryScope(agent_id="test_agent")
        mem = await engine.procedural.add(
            content="Always run tests after each file change",
            scope=scope,
            tags=["testing"],
        )
        assert mem.memory_type == MemoryType.PROCEDURAL
        assert mem.score == 1.0  # Never decays

        # Retrieve by agent
        all_instructions = await engine.procedural.get_all(scope)
        assert len(all_instructions) >= 1
        assert any("tests" in m.content for m in all_instructions)

    async def test_working_memory(self, engine):
        scope = MemoryScope(session_id="test_session")
        messages = [
            {"role": "user", "content": "What is Kafka?"},
            {"role": "assistant", "content": "Kafka is a distributed streaming platform."},
        ]

        await engine.working.append(messages, scope)
        window = await engine.working.get_window(scope)
        assert len(window) == 2
        assert window[0]["content"] == "What is Kafka?"

        # Expire session
        await engine.working.expire_session("test_session")
        window = await engine.working.get_window(scope)
        assert len(window) == 0
