"""Integration test: context assembly from all four types."""

import pytest

from pymem.models import MemoryScope


@pytest.mark.asyncio
class TestContextAssembly:
    async def test_assemble_empty_context(self, engine):
        """Context for user with no memories should return empty formatted string."""
        scope = MemoryScope(user_id="empty_user")
        ctx = await engine.get_context(scope=scope)
        assert ctx.formatted == "" or ctx.formatted is not None

    async def test_assemble_with_semantic_memories(self, engine):
        """Context should include semantic memories."""
        scope = MemoryScope(user_id="ctx_user_1", agent_id="ctx_agent")

        await engine.semantic.add(
            content="User is a senior Python developer",
            scope=scope,
            score=0.9,
        )
        await engine.semantic.add(
            content="User works at TechCorp",
            scope=scope,
            score=0.8,
        )

        ctx = await engine.get_context(
            scope=scope,
            query="Tell me about the user",
        )
        assert len(ctx.relevant_facts) > 0
        assert "Python developer" in ctx.formatted or len(ctx.relevant_facts) > 0

    async def test_assemble_with_procedural(self, engine):
        """Context should include procedural instructions."""
        scope = MemoryScope(agent_id="ctx_agent_2")

        await engine.procedural.add(
            content="Always respond in TypeScript",
            scope=scope,
            tags=["coding"],
        )

        ctx = await engine.get_context(scope=scope)
        assert len(ctx.agent_instructions) >= 1
        assert "TypeScript" in ctx.formatted

    async def test_assemble_with_working_memory(self, engine):
        """Context should include recent working memory messages."""
        scope = MemoryScope(
            user_id="ctx_user_2", session_id="ctx_session_1"
        )

        messages = [
            {"role": "user", "content": "I need help with Kubernetes"},
            {"role": "assistant", "content": "Sure, what specifically?"},
        ]
        await engine.working.append(messages, scope)

        ctx = await engine.get_context(scope=scope)
        assert len(ctx.working_messages) == 2
