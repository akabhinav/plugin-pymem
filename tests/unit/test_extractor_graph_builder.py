"""Unit tests for MemoryExtractor and GraphBuilder."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from pymem.intelligence.extractor import MemoryExtractor
from pymem.intelligence.graph_builder import GraphBuilder
from pymem.models import MemoryRecord, MemoryScope, MemoryType


# ──────────────────────────────────────────────────────────────
# MemoryExtractor
# ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestMemoryExtractor:
    def _make_pygate_mock(self, response: dict) -> AsyncMock:
        mock = AsyncMock()
        mock.complete_json = AsyncMock(return_value=response)
        return mock

    async def test_extract_returns_memories(self):
        """Extraction should return typed MemoryRecords."""
        pygate = self._make_pygate_mock({
            "memories": [
                {
                    "content": "User prefers Python",
                    "memory_type": "semantic",
                    "score": 0.9,
                    "entities": ["Python"],
                    "temporal": False,
                },
                {
                    "content": "User completed the onboarding on March 1",
                    "memory_type": "episodic",
                    "score": 0.7,
                    "entities": [],
                    "temporal": True,
                },
            ]
        })
        extractor = MemoryExtractor(pygate, min_score=0.3)
        scope = MemoryScope(user_id="test_user")
        messages = [{"role": "user", "content": "I prefer Python"}]

        result = await extractor.extract(messages, scope)
        assert len(result) == 2
        assert result[0].memory_type == MemoryType.SEMANTIC
        assert result[0].content == "User prefers Python"
        assert result[1].memory_type == MemoryType.EPISODIC

    async def test_extract_filters_by_min_score(self):
        """Memories below min_score should be filtered out."""
        pygate = self._make_pygate_mock({
            "memories": [
                {"content": "Important fact", "memory_type": "semantic", "score": 0.8},
                {"content": "Trivial filler", "memory_type": "semantic", "score": 0.1},
            ]
        })
        extractor = MemoryExtractor(pygate, min_score=0.3)
        scope = MemoryScope(user_id="u1")

        result = await extractor.extract(
            [{"role": "user", "content": "test"}], scope
        )
        assert len(result) == 1
        assert result[0].content == "Important fact"

    async def test_extract_handles_invalid_memory_type(self):
        """Invalid memory_type should default to SEMANTIC."""
        pygate = self._make_pygate_mock({
            "memories": [
                {"content": "Unknown type", "memory_type": "unknown_type", "score": 0.8},
            ]
        })
        extractor = MemoryExtractor(pygate, min_score=0.3)
        scope = MemoryScope(user_id="u1")

        result = await extractor.extract(
            [{"role": "user", "content": "test"}], scope
        )
        assert len(result) == 1
        assert result[0].memory_type == MemoryType.SEMANTIC

    async def test_extract_empty_result(self):
        """Empty extraction should return empty list."""
        pygate = self._make_pygate_mock({"memories": []})
        extractor = MemoryExtractor(pygate, min_score=0.3)
        scope = MemoryScope(user_id="u1")

        result = await extractor.extract(
            [{"role": "user", "content": "hello"}], scope
        )
        assert result == []

    async def test_extract_sets_scope_on_records(self):
        """Extracted records should carry scope fields."""
        pygate = self._make_pygate_mock({
            "memories": [
                {"content": "Scoped memory", "memory_type": "semantic", "score": 0.8},
            ]
        })
        extractor = MemoryExtractor(pygate, min_score=0.3)
        scope = MemoryScope(user_id="u1", agent_id="a1")

        result = await extractor.extract(
            [{"role": "user", "content": "test"}], scope
        )
        assert result[0].user_id == "u1"
        assert result[0].agent_id == "a1"

    async def test_extract_simple_creates_from_user_messages(self):
        """Simple extraction should create one memory per user message > 10 chars."""
        pygate = self._make_pygate_mock({})
        extractor = MemoryExtractor(pygate, min_score=0.3)
        scope = MemoryScope(user_id="u1")

        messages = [
            {"role": "user", "content": "I prefer Python over Java for backend work"},
            {"role": "assistant", "content": "Got it!"},
            {"role": "user", "content": "Short"},  # < 10 chars, skipped
            {"role": "user", "content": "I've been using Kafka for three years now"},
        ]

        result = await extractor.extract_simple(messages, scope)
        assert len(result) == 2
        assert all(r.memory_type == MemoryType.SEMANTIC for r in result)
        assert all(r.score == 0.5 for r in result)

    async def test_extract_procedural_type(self):
        """Extractor should handle procedural memory type."""
        pygate = self._make_pygate_mock({
            "memories": [
                {"content": "Always use TypeScript", "memory_type": "procedural", "score": 0.9},
            ]
        })
        extractor = MemoryExtractor(pygate, min_score=0.3)
        scope = MemoryScope(agent_id="a1")

        result = await extractor.extract(
            [{"role": "user", "content": "use typescript"}], scope
        )
        assert result[0].memory_type == MemoryType.PROCEDURAL


# ──────────────────────────────────────────────────────────────
# GraphBuilder
# ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
class TestGraphBuilder:
    def _make_pygate_mock(self, response: dict) -> AsyncMock:
        mock = AsyncMock()
        mock.complete_json = AsyncMock(return_value=response)
        return mock

    async def test_build_with_llm(self):
        """LLM-based graph building should return nodes and edges."""
        pygate = self._make_pygate_mock({
            "entities": [
                {"name": "Python", "type": "Tool", "description": "Programming language"},
                {"name": "User", "type": "Person", "description": "The user"},
            ],
            "relationships": [
                {
                    "source": "User",
                    "target": "Python",
                    "type": "USES",
                    "description": "User uses Python",
                },
            ],
        })

        builder = GraphBuilder(pygate)
        scope = {"user_id": "u1"}
        memories = [
            MemoryRecord(
                id="m1",
                memory_type=MemoryType.SEMANTIC,
                content="User uses Python",
            ),
        ]

        nodes, edges = await builder.build_from_memories(memories, scope)
        assert len(nodes) == 2
        assert len(edges) == 1
        assert edges[0].relationship == "USES"

    async def test_build_from_metadata_without_llm(self):
        """Without LLM, should extract entities from metadata."""
        builder = GraphBuilder(pygate=None)
        scope = {"user_id": "u1"}
        memories = [
            MemoryRecord(
                id="m1",
                memory_type=MemoryType.SEMANTIC,
                content="User knows Python and Kafka",
                metadata={"entities": ["Python", "Kafka"]},
            ),
        ]

        nodes, edges = await builder.build_from_memories(memories, scope)
        assert len(nodes) == 2
        names = {n.properties["name"] for n in nodes}
        assert names == {"Python", "Kafka"}
        # Co-occurring entities should have an edge
        assert len(edges) == 1

    async def test_build_from_metadata_no_entities(self):
        """Memories with no entities should produce empty graph."""
        builder = GraphBuilder(pygate=None)
        scope = {"user_id": "u1"}
        memories = [
            MemoryRecord(
                id="m1",
                memory_type=MemoryType.SEMANTIC,
                content="Some fact",
                metadata={},
            ),
        ]

        nodes, edges = await builder.build_from_memories(memories, scope)
        assert len(nodes) == 0
        assert len(edges) == 0

    async def test_build_deduplicates_entities(self):
        """Same entity appearing in multiple memories should only create one node."""
        builder = GraphBuilder(pygate=None)
        scope = {"user_id": "u1"}
        memories = [
            MemoryRecord(
                id="m1", memory_type=MemoryType.SEMANTIC,
                content="A", metadata={"entities": ["Python"]},
            ),
            MemoryRecord(
                id="m2", memory_type=MemoryType.SEMANTIC,
                content="B", metadata={"entities": ["Python"]},
            ),
        ]

        nodes, edges = await builder.build_from_memories(memories, scope)
        assert len(nodes) == 1
        assert nodes[0].properties["name"] == "Python"
