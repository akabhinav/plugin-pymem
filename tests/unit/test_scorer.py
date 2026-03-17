"""Tests for memory scorer — heuristic scoring."""

import pytest

from pymem.intelligence.scorer import MemoryScorer
from pymem.models import MemoryRecord, MemoryType


class TestMemoryScorer:
    def setup_method(self):
        self.scorer = MemoryScorer()

    @pytest.mark.asyncio
    async def test_preference_boost(self):
        mem = MemoryRecord(
            id="m1",
            memory_type=MemoryType.SEMANTIC,
            content="I always prefer dark mode",
            user_id="u1",
        )
        score = await self.scorer.score(mem)
        assert score > 0.5  # Preference words boost

    @pytest.mark.asyncio
    async def test_short_content_penalty(self):
        mem = MemoryRecord(
            id="m1",
            memory_type=MemoryType.SEMANTIC,
            content="ok",
            user_id="u1",
        )
        score = await self.scorer.score(mem)
        assert score < 0.5

    @pytest.mark.asyncio
    async def test_entity_boost(self):
        mem = MemoryRecord(
            id="m1",
            memory_type=MemoryType.SEMANTIC,
            content="Some content here in this memory",
            metadata={"entities": ["Python", "Java"]},
            user_id="u1",
        )
        score = await self.scorer.score(mem)
        assert score >= 0.6

    @pytest.mark.asyncio
    async def test_score_bounded(self):
        mem = MemoryRecord(
            id="m1",
            memory_type=MemoryType.SEMANTIC,
            content="I always prefer and love this important thing which is never bad",
            metadata={"entities": ["A", "B", "C"]},
            user_id="u1",
        )
        score = await self.scorer.score(mem)
        assert 0.0 <= score <= 1.0
