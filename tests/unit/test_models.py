"""Tests for core domain models."""

import pytest

from pymem.models import (
    AddMemoryResult,
    AgentContext,
    ConsolidationResult,
    ForgetResult,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    SearchResult,
)


class TestMemoryScope:
    def test_requires_at_least_one_field(self):
        with pytest.raises(ValueError, match="at least one"):
            MemoryScope()

    def test_user_scope(self):
        scope = MemoryScope(user_id="u1")
        assert scope.user_id == "u1"
        assert scope.agent_id is None
        assert scope.to_dict() == {"user_id": "u1"}

    def test_full_scope(self):
        scope = MemoryScope(
            user_id="u1", agent_id="a1", session_id="s1", org_id="o1"
        )
        d = scope.to_dict()
        assert len(d) == 4
        assert d["user_id"] == "u1"

    def test_to_filter(self):
        scope = MemoryScope(user_id="u1", agent_id="a1")
        f = scope.to_filter()
        assert f == {"user_id": "u1", "agent_id": "a1"}


class TestMemoryType:
    def test_all_types(self):
        assert MemoryType.WORKING == "working"
        assert MemoryType.EPISODIC == "episodic"
        assert MemoryType.SEMANTIC == "semantic"
        assert MemoryType.PROCEDURAL == "procedural"

    def test_from_string(self):
        assert MemoryType("semantic") == MemoryType.SEMANTIC


class TestMemoryRecord:
    def test_basic_creation(self):
        record = MemoryRecord(
            id="test-id",
            memory_type=MemoryType.SEMANTIC,
            content="User prefers Python",
            user_id="u1",
        )
        assert record.id == "test-id"
        assert record.memory_type == MemoryType.SEMANTIC
        assert record.content == "User prefers Python"
        assert record.version == 1
        assert record.is_deleted is False

    def test_scope_property(self):
        record = MemoryRecord(
            id="test",
            memory_type=MemoryType.SEMANTIC,
            content="test",
            user_id="u1",
            agent_id="a1",
        )
        scope = record.scope
        assert scope.user_id == "u1"
        assert scope.agent_id == "a1"

    def test_scope_dict(self):
        record = MemoryRecord(
            id="test",
            memory_type=MemoryType.SEMANTIC,
            content="test",
            user_id="u1",
        )
        assert record.scope_dict() == {"user_id": "u1"}


class TestConsolidationResult:
    def test_keep_new(self):
        record = MemoryRecord(
            id="new",
            memory_type=MemoryType.SEMANTIC,
            content="new fact",
            user_id="u1",
        )
        result = ConsolidationResult(action="keep_new", memory=record)
        assert result.action == "keep_new"

    def test_from_llm_response_merge(self):
        new = MemoryRecord(
            id="new",
            memory_type=MemoryType.SEMANTIC,
            content="Python expert",
            score=0.7,
            user_id="u1",
        )
        existing = [
            MemoryRecord(
                id="old",
                memory_type=MemoryType.SEMANTIC,
                content="Knows Python",
                score=0.8,
                user_id="u1",
            )
        ]
        response = {
            "action": "merge",
            "merged_content": "Expert Python developer",
            "reason": "Combined info",
        }
        result = ConsolidationResult.from_llm_response(response, new, existing)
        assert result.action == "merge"
        assert result.memory is not None
        assert result.memory.content == "Expert Python developer"
        assert result.memory.score == 0.8  # Max of scores

    def test_from_llm_response_discard(self):
        new = MemoryRecord(
            id="new",
            memory_type=MemoryType.SEMANTIC,
            content="test",
            user_id="u1",
        )
        result = ConsolidationResult.from_llm_response(
            {"action": "discard", "reason": "duplicate"}, new, []
        )
        assert result.action == "discard"


class TestAddMemoryResult:
    def test_queued(self):
        result = AddMemoryResult(status="queued")
        assert result.status == "queued"
        assert result.memories_count == 0

    def test_stored_with_memories(self):
        result = AddMemoryResult(status="stored", memories_count=3)
        assert result.memories_count == 3


class TestSearchResult:
    def test_empty(self):
        result = SearchResult()
        assert result.memories == []
        assert result.total == 0


class TestForgetResult:
    def test_basic(self):
        result = ForgetResult(
            deleted_per_type={"semantic": 5, "episodic": 3},
            confirmation_id="abc",
        )
        assert result.deleted_per_type["semantic"] == 5


class TestAgentContext:
    def test_empty(self):
        ctx = AgentContext()
        assert ctx.formatted == ""
        assert ctx.working_messages == []
