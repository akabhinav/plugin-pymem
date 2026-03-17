"""Tests for context assembler — prompt formatting."""

from pymem.engine.context_assembler import ContextAssembler
from pymem.models import MemoryRecord, MemoryType


class TestContextAssembler:
    def test_format_prompt_block_empty(self):
        result = ContextAssembler._format_prompt_block([], [], [], [])
        assert result == ""

    def test_format_prompt_block_procedural(self):
        procedural = [
            MemoryRecord(
                id="p1",
                memory_type=MemoryType.PROCEDURAL,
                content="Always use TypeScript",
                user_id="u1",
            )
        ]
        result = ContextAssembler._format_prompt_block([], [], [], procedural)
        assert "Agent instructions" in result
        assert "Always use TypeScript" in result

    def test_format_prompt_block_semantic(self):
        semantic = [
            MemoryRecord(
                id="s1",
                memory_type=MemoryType.SEMANTIC,
                content="User prefers dark mode",
                user_id="u1",
            )
        ]
        result = ContextAssembler._format_prompt_block([], semantic, [], [])
        assert "What I know about you" in result
        assert "dark mode" in result

    def test_format_prompt_block_all_types(self):
        procedural = [
            MemoryRecord(
                id="p1",
                memory_type=MemoryType.PROCEDURAL,
                content="Be concise",
                user_id="u1",
            )
        ]
        semantic = [
            MemoryRecord(
                id="s1",
                memory_type=MemoryType.SEMANTIC,
                content="User likes Python",
                user_id="u1",
            )
        ]
        episodic = [
            MemoryRecord(
                id="e1",
                memory_type=MemoryType.EPISODIC,
                content="Completed Kafka setup",
                user_id="u1",
            )
        ]
        working = [{"role": "user", "content": "Hello there"}]

        result = ContextAssembler._format_prompt_block(
            working, semantic, episodic, procedural
        )

        assert "Agent instructions" in result
        assert "What I know about you" in result
        assert "Recent events" in result
        assert "Current conversation" in result
