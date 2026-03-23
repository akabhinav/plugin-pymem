"""E2E workflow tests — complete memory lifecycle scenarios."""

from __future__ import annotations

import pytest

from pymem.models import MemoryScope, MemoryType


@pytest.mark.asyncio
class TestFullMemoryLifecycle:
    """Add → search → update → search again → delete → verify gone."""

    async def test_semantic_lifecycle(self, engine):
        scope = MemoryScope(user_id="lifecycle_user")

        # 1. Add
        mem = await engine.semantic.add(
            content="User prefers vim over VS Code",
            scope=scope,
            score=0.9,
        )
        assert mem.id

        # 2. Search
        result = await engine.search("editor preference", scope, limit=5)
        assert any("vim" in m.content for m in result.memories)

        # 3. Get
        fetched = await engine.get(mem.id)
        assert fetched is not None
        assert fetched.content == "User prefers vim over VS Code"

        # 4. Update
        updated = await engine.update(
            mem.id, "User now prefers VS Code over vim", scope
        )
        assert updated is not None
        assert updated.version == 2

        # 5. Search again — should find updated content
        result2 = await engine.search("editor preference", scope, limit=5)
        assert any("VS Code" in m.content for m in result2.memories)

        # 6. Delete
        await engine.delete(updated.id, scope, hard_delete=True)

        # 7. Verify gone
        assert await engine.get(updated.id) is None

    async def test_episodic_lifecycle(self, engine):
        scope = MemoryScope(user_id="epi_lifecycle_user")

        # Add events
        ev1 = await engine.episodic.add_event(
            content="Started the new project today",
            scope=scope,
        )
        ev2 = await engine.episodic.add_event(
            content="Completed code review for PR #42",
            scope=scope,
        )

        # Get recent
        recent = await engine.episodic.get_recent(scope, limit=10)
        assert len(recent) >= 2

        # Search
        result = await engine.search(
            "project activities",
            scope,
            memory_types=[MemoryType.EPISODIC],
            limit=5,
        )
        assert len(result.memories) >= 1

        # Delete
        await engine.delete(ev1.id, scope, hard_delete=True)
        await engine.delete(ev2.id, scope, hard_delete=True)

    async def test_procedural_lifecycle(self, engine):
        scope = MemoryScope(agent_id="proc_lifecycle_agent")

        # Add instructions
        inst1 = await engine.procedural.add(
            content="Always format code with black",
            scope=scope,
            tags=["formatting"],
            priority=1,
        )
        inst2 = await engine.procedural.add(
            content="Use pytest for testing",
            scope=scope,
            tags=["testing"],
            priority=3,
        )

        # Get all
        all_insts = await engine.procedural.get_all(scope)
        assert len(all_insts) >= 2
        # Priority 1 should come first
        assert all_insts[0].metadata["priority"] <= all_insts[1].metadata["priority"]

        # Search
        results = await engine.procedural.search("formatting", scope)
        assert results[0].content == "Always format code with black"

        # Clean up
        await engine.procedural.delete_by_scope(scope)
        remaining = await engine.procedural.get_all(scope)
        assert len(remaining) == 0

    async def test_working_memory_lifecycle(self, engine):
        scope = MemoryScope(session_id="wm_lifecycle_session")

        # Append messages
        await engine.working.append(
            [
                {"role": "user", "content": "What is Redis?"},
                {"role": "assistant", "content": "Redis is an in-memory data store."},
            ],
            scope,
        )

        # Verify
        window = await engine.working.get_window(scope)
        assert len(window) == 2

        # Get length
        length = await engine.working.get_length("wm_lifecycle_session")
        assert length == 2

        # Expire
        await engine.working.expire_session("wm_lifecycle_session")
        window = await engine.working.get_window(scope)
        assert len(window) == 0


@pytest.mark.asyncio
class TestMultiUserIsolation:
    """Memories from different users should NEVER leak across scopes."""

    async def test_semantic_isolation(self, engine):
        scope_a = MemoryScope(user_id="iso_user_a")
        scope_b = MemoryScope(user_id="iso_user_b")

        await engine.semantic.add(content="A's salary is 150k", scope=scope_a)
        await engine.semantic.add(content="B's salary is 200k", scope=scope_b)

        # Search as user A
        result_a = await engine.search("salary", scope_a, limit=10)
        for m in result_a.memories:
            assert "B's salary" not in m.content

        # Search as user B
        result_b = await engine.search("salary", scope_b, limit=10)
        for m in result_b.memories:
            assert "A's salary" not in m.content

    async def test_working_memory_isolation(self, engine):
        scope_a = MemoryScope(session_id="iso_session_a")
        scope_b = MemoryScope(session_id="iso_session_b")

        await engine.working.append(
            [{"role": "user", "content": "A's private message"}], scope_a
        )
        await engine.working.append(
            [{"role": "user", "content": "B's private message"}], scope_b
        )

        window_a = await engine.working.get_window(scope_a)
        window_b = await engine.working.get_window(scope_b)

        assert all("B's" not in m["content"] for m in window_a)
        assert all("A's" not in m["content"] for m in window_b)

    async def test_procedural_isolation(self, engine):
        scope_a = MemoryScope(agent_id="iso_agent_a")
        scope_b = MemoryScope(agent_id="iso_agent_b")

        await engine.procedural.add(content="Agent A rule", scope=scope_a)
        await engine.procedural.add(content="Agent B rule", scope=scope_b)

        insts_a = await engine.procedural.get_all(scope_a)
        insts_b = await engine.procedural.get_all(scope_b)

        assert all("Agent B" not in i.content for i in insts_a)
        assert all("Agent A" not in i.content for i in insts_b)


@pytest.mark.asyncio
class TestGDPRCompliance:
    """GDPR forget must wipe ALL data from ALL backends."""

    async def test_full_gdpr_forget_workflow(self, engine):
        scope = MemoryScope(user_id="gdpr_full_user", session_id="gdpr_session")

        # Create memories of every type
        await engine.semantic.add(content="GDPR semantic fact", scope=scope)
        await engine.episodic.add_event(content="GDPR episodic event", scope=scope)
        await engine.procedural.add(
            content="GDPR procedural instruction",
            scope=MemoryScope(user_id="gdpr_full_user", agent_id="gdpr_agent"),
        )
        await engine.working.append(
            [{"role": "user", "content": "GDPR working message"}], scope
        )

        # Verify data exists
        search = await engine.search("GDPR", scope, limit=20)
        assert len(search.memories) > 0

        # Forget everything
        result = await engine.forget(scope=MemoryScope(user_id="gdpr_full_user"))
        assert result.confirmation_id
        assert sum(result.deleted_per_type.values()) >= 1

        # Verify: search returns empty
        search_after = await engine.search("GDPR", scope, limit=20)
        assert len(search_after.memories) == 0


@pytest.mark.asyncio
class TestContextAssemblyWorkflow:
    """Test full context assembly from multiple memory types."""

    async def test_assemble_all_types(self, engine):
        scope = MemoryScope(
            user_id="ctx_wf_user",
            agent_id="ctx_wf_agent",
            session_id="ctx_wf_session",
        )

        # Populate all types
        await engine.semantic.add(
            content="User is a senior backend engineer",
            scope=scope,
            score=0.9,
        )
        await engine.episodic.add_event(
            content="User deployed microservice v2.0",
            scope=scope,
        )
        await engine.procedural.add(
            content="Always use structured logging",
            scope=scope,
            tags=["logging"],
        )
        await engine.working.append(
            [{"role": "user", "content": "Help me debug the auth service"}],
            scope,
        )

        # Assemble context
        ctx = await engine.get_context(
            scope=scope,
            query="backend engineering",
            max_tokens=4000,
        )

        # Verify all types present
        assert len(ctx.relevant_facts) > 0
        assert len(ctx.agent_instructions) >= 1
        assert ctx.formatted  # Non-empty formatted block

    async def test_context_with_query_relevance(self, engine):
        scope = MemoryScope(user_id="ctx_rel_user", agent_id="ctx_rel_agent")

        await engine.semantic.add(
            content="User loves pizza and Italian food",
            scope=scope,
        )
        await engine.semantic.add(
            content="User is expert in Kubernetes and Docker",
            scope=scope,
        )

        # Query about tech should surface tech memory
        ctx = await engine.get_context(scope=scope, query="container orchestration")
        assert ctx.relevant_facts  # Should have results


@pytest.mark.asyncio
class TestMemoryVersioning:
    """Test memory versioning through updates."""

    async def test_multiple_updates_create_version_chain(self, engine):
        scope = MemoryScope(user_id="version_user")

        # Version 1
        v1 = await engine.semantic.add(
            content="Version 1 content",
            scope=scope,
        )

        # Version 2
        v2 = await engine.update(v1.id, "Version 2 content", scope)
        assert v2.version == 2
        assert v2.parent_id == v1.id

        # Version 3
        v3 = await engine.update(v2.id, "Version 3 content", scope)
        assert v3.version == 3
        assert v3.parent_id == v2.id

        # Only latest version accessible
        assert await engine.get(v1.id) is None  # soft-deleted
        assert await engine.get(v2.id) is None  # soft-deleted
        latest = await engine.get(v3.id)
        assert latest is not None
        assert latest.content == "Version 3 content"


@pytest.mark.asyncio
class TestShareWorkflow:
    """Test memory sharing between scopes."""

    async def test_share_makes_memory_accessible_in_new_scope(self, engine):
        from_scope = MemoryScope(user_id="share_from")
        to_scope = MemoryScope(user_id="share_to")

        original = await engine.semantic.add(
            content="Shared team knowledge base",
            scope=from_scope,
        )

        shared = await engine.share(original.id, from_scope, to_scope)
        assert shared is not None

        # Shared memory should be searchable in new scope
        results = await engine.search("knowledge base", to_scope, limit=5)
        assert any("team knowledge" in m.content for m in results.memories)

    async def test_share_does_not_remove_from_original(self, engine):
        from_scope = MemoryScope(user_id="share_orig")
        to_scope = MemoryScope(user_id="share_dest")

        original = await engine.semantic.add(
            content="Original stays after share",
            scope=from_scope,
        )
        await engine.share(original.id, from_scope, to_scope)

        # Original still accessible
        result = await engine.search("stays after share", from_scope, limit=5)
        assert any("Original stays" in m.content for m in result.memories)
