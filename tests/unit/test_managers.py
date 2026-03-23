"""Unit tests for all four memory type managers."""

from __future__ import annotations

import pytest

from pymem.models import MemoryScope, MemoryType


# ──────────────────────────────────────────────────────────────
# Semantic Memory Manager
# ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestSemanticManager:
    async def test_add_returns_record(self, engine, user_scope):
        mem = await engine.semantic.add(
            content="User knows Python",
            scope=user_scope,
            score=0.9,
        )
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.content == "User knows Python"
        assert mem.score == 0.9
        assert mem.id is not None

    async def test_add_stores_in_vector(self, engine, user_scope):
        mem = await engine.semantic.add(content="Vector check", scope=user_scope)
        vec = await engine._vector.get(mem.id)
        assert vec is not None
        assert vec.content == "Vector check"

    async def test_add_stores_in_relational(self, engine, user_scope):
        mem = await engine.semantic.add(content="Relational check", scope=user_scope)
        row = await engine._relational.get_memory(mem.id)
        assert row is not None
        assert row["content"] == "Relational check"

    async def test_add_with_metadata(self, engine, user_scope):
        mem = await engine.semantic.add(
            content="Tagged memory",
            scope=user_scope,
            metadata={"tags": ["python", "pref"]},
        )
        assert mem.metadata["tags"] == ["python", "pref"]

    async def test_search_by_embedding(self, engine, user_scope):
        await engine.semantic.add(content="User prefers dark mode", scope=user_scope)
        await engine.semantic.add(content="User works at TechCorp", scope=user_scope)

        embedding = await engine._embedder.embed("dark theme preference")
        results = await engine.semantic.search(embedding, user_scope, limit=5)
        assert len(results) > 0
        assert all(r.memory_type == MemoryType.SEMANTIC for r in results)

    async def test_search_without_embedding_falls_back(self, engine, user_scope):
        await engine.semantic.add(content="Fallback test", scope=user_scope)
        results = await engine.semantic.search(None, user_scope, limit=5)
        assert len(results) > 0

    async def test_search_respects_scope(self, engine):
        scope_a = MemoryScope(user_id="sem_user_a")
        scope_b = MemoryScope(user_id="sem_user_b")
        await engine.semantic.add(content="A's secret", scope=scope_a)
        await engine.semantic.add(content="B's secret", scope=scope_b)

        emb = await engine._embedder.embed("secret")
        results_a = await engine.semantic.search(emb, scope_a, limit=10)
        for r in results_a:
            assert "B's secret" != r.content or r.user_id != "sem_user_b"

    async def test_get_similar(self, engine, user_scope):
        await engine.semantic.add(content="User likes Python", scope=user_scope)
        similar = await engine.semantic.get_similar("Python preference", user_scope, limit=5)
        assert len(similar) > 0

    async def test_delete_by_scope(self, engine):
        scope = MemoryScope(user_id="delete_sem_user")
        await engine.semantic.add(content="To be deleted", scope=scope)
        count = await engine.semantic.delete_by_scope(scope)
        assert count >= 1

        results = await engine.semantic.search(None, scope, limit=10)
        assert len(results) == 0


# ──────────────────────────────────────────────────────────────
# Episodic Memory Manager
# ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestEpisodicManager:
    async def test_add_event_returns_record(self, engine, user_scope):
        mem = await engine.episodic.add_event(
            content="User completed code review",
            scope=user_scope,
        )
        assert mem.memory_type == MemoryType.EPISODIC
        assert mem.content == "User completed code review"
        assert mem.score == 0.7

    async def test_add_event_stores_in_graph(self, engine, user_scope):
        mem = await engine.episodic.add_event(
            content="Graph node check",
            scope=user_scope,
        )
        node = await engine._graph.get_node(mem.id)
        assert node is not None
        assert node.properties["content"] == "Graph node check"

    async def test_add_event_stores_in_relational(self, engine, user_scope):
        mem = await engine.episodic.add_event(
            content="Relational event check",
            scope=user_scope,
        )
        row = await engine._relational.get_memory(mem.id)
        assert row is not None
        assert row["memory_type"] == "episodic"

    async def test_add_event_with_metadata(self, engine, user_scope):
        mem = await engine.episodic.add_event(
            content="Meeting with Bob",
            scope=user_scope,
            metadata={"attendees": ["Bob"], "location": "Zoom"},
        )
        assert mem.metadata["attendees"] == ["Bob"]

    async def test_search_returns_events(self, engine, user_scope):
        await engine.episodic.add_event(content="Event alpha", scope=user_scope)
        await engine.episodic.add_event(content="Event beta", scope=user_scope)

        emb = await engine._embedder.embed("event")
        results = await engine.episodic.search(emb, user_scope, limit=10)
        assert len(results) >= 2
        assert all(r.memory_type == MemoryType.EPISODIC for r in results)

    async def test_get_recent(self, engine):
        scope = MemoryScope(user_id="recent_user")
        await engine.episodic.add_event(content="First event", scope=scope)
        await engine.episodic.add_event(content="Second event", scope=scope)

        recent = await engine.episodic.get_recent(scope, limit=5)
        assert len(recent) >= 2

    async def test_delete_by_scope(self, engine):
        scope = MemoryScope(user_id="delete_epi_user")
        await engine.episodic.add_event(content="To be deleted", scope=scope)
        count = await engine.episodic.delete_by_scope(scope)
        assert count >= 1


# ──────────────────────────────────────────────────────────────
# Procedural Memory Manager
# ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestProceduralManager:
    async def test_add_returns_record(self, engine):
        scope = MemoryScope(agent_id="proc_agent")
        mem = await engine.procedural.add(
            content="Always run tests",
            scope=scope,
            tags=["testing"],
            priority=1,
        )
        assert mem.memory_type == MemoryType.PROCEDURAL
        assert mem.content == "Always run tests"
        assert mem.score == 1.0  # Never decays

    async def test_add_stores_with_priority(self, engine):
        scope = MemoryScope(agent_id="prio_agent")
        mem = await engine.procedural.add(
            content="High priority instruction",
            scope=scope,
            priority=1,
        )
        assert mem.metadata["priority"] == 1

    async def test_get_all_ordered_by_priority(self, engine):
        scope = MemoryScope(agent_id="order_agent")
        await engine.procedural.add(content="Low prio", scope=scope, priority=10)
        await engine.procedural.add(content="High prio", scope=scope, priority=1)
        await engine.procedural.add(content="Mid prio", scope=scope, priority=5)

        all_procs = await engine.procedural.get_all(scope)
        assert len(all_procs) >= 3
        priorities = [m.metadata["priority"] for m in all_procs]
        assert priorities == sorted(priorities)

    async def test_get_all_with_tag_filter(self, engine):
        scope = MemoryScope(agent_id="tag_agent")
        await engine.procedural.add(content="Tagged A", scope=scope, tags=["coding"])
        await engine.procedural.add(content="Tagged B", scope=scope, tags=["style"])

        results = await engine.procedural.get_all(scope, tags=["coding"])
        assert any("Tagged A" in m.content for m in results)

    async def test_search_by_keyword(self, engine):
        scope = MemoryScope(agent_id="search_proc_agent")
        await engine.procedural.add(content="Always use TypeScript", scope=scope)
        await engine.procedural.add(content="Format code with prettier", scope=scope)

        results = await engine.procedural.search("TypeScript", scope, limit=5)
        assert len(results) > 0
        assert results[0].content == "Always use TypeScript"

    async def test_delete_by_scope(self, engine):
        scope = MemoryScope(agent_id="delete_proc_agent")
        await engine.procedural.add(content="To be deleted", scope=scope)
        count = await engine.procedural.delete_by_scope(scope)
        assert count >= 1

        remaining = await engine.procedural.get_all(scope)
        assert len(remaining) == 0


# ──────────────────────────────────────────────────────────────
# Working Memory Manager
# ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestWorkingMemoryManager:
    async def test_append_and_get_window(self, engine):
        scope = MemoryScope(session_id="wm_session_1")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        await engine.working.append(messages, scope)
        window = await engine.working.get_window(scope)
        assert len(window) == 2
        assert window[0]["content"] == "Hello"
        assert window[1]["content"] == "Hi there!"

    async def test_append_multiple_times(self, engine):
        scope = MemoryScope(session_id="wm_session_2")
        await engine.working.append(
            [{"role": "user", "content": "First"}], scope
        )
        await engine.working.append(
            [{"role": "user", "content": "Second"}], scope
        )
        window = await engine.working.get_window(scope)
        assert len(window) == 2

    async def test_get_window_empty_session(self, engine):
        scope = MemoryScope(session_id="wm_empty")
        window = await engine.working.get_window(scope)
        assert window == []

    async def test_get_window_no_session_id(self, engine):
        scope = MemoryScope(user_id="no_session")
        window = await engine.working.get_window(scope)
        assert window == []

    async def test_append_no_session_id_noop(self, engine):
        scope = MemoryScope(user_id="no_session")
        await engine.working.append(
            [{"role": "user", "content": "Ignored"}], scope
        )
        # No error

    async def test_expire_session_clears_window(self, engine):
        scope = MemoryScope(session_id="wm_expire")
        await engine.working.append(
            [{"role": "user", "content": "To be cleared"}], scope
        )
        await engine.working.expire_session("wm_expire")
        window = await engine.working.get_window(scope)
        assert len(window) == 0

    async def test_get_length(self, engine):
        scope = MemoryScope(session_id="wm_len")
        await engine.working.append(
            [
                {"role": "user", "content": "One"},
                {"role": "assistant", "content": "Two"},
                {"role": "user", "content": "Three"},
            ],
            scope,
        )
        length = await engine.working.get_length("wm_len")
        assert length == 3

    async def test_get_length_empty(self, engine):
        length = await engine.working.get_length("wm_nonexistent")
        assert length == 0
