"""Tests for memory consolidator — dedup logic."""

import pytest

from pymem.intelligence.consolidator import MemoryConsolidator
from pymem.models import MemoryRecord, MemoryType


@pytest.mark.asyncio
class TestConsolidatorSimple:
    async def test_keep_new_when_no_existing(self):
        consolidator = MemoryConsolidator(pygate=None)  # type: ignore
        new = MemoryRecord(
            id="new",
            memory_type=MemoryType.SEMANTIC,
            content="User prefers Python",
            user_id="u1",
        )
        result = await consolidator.consolidate_simple(new, [])
        assert result.action == "keep_new"

    async def test_discard_exact_duplicate(self):
        consolidator = MemoryConsolidator(pygate=None)  # type: ignore
        new = MemoryRecord(
            id="new",
            memory_type=MemoryType.SEMANTIC,
            content="User prefers Python",
            user_id="u1",
        )
        existing = [
            MemoryRecord(
                id="old",
                memory_type=MemoryType.SEMANTIC,
                content="User prefers Python",
                user_id="u1",
            )
        ]
        result = await consolidator.consolidate_simple(new, existing)
        assert result.action == "discard"

    async def test_keep_new_when_different(self):
        consolidator = MemoryConsolidator(pygate=None)  # type: ignore
        new = MemoryRecord(
            id="new",
            memory_type=MemoryType.SEMANTIC,
            content="User prefers Python",
            user_id="u1",
        )
        existing = [
            MemoryRecord(
                id="old",
                memory_type=MemoryType.SEMANTIC,
                content="User knows Java",
                user_id="u1",
            )
        ]
        result = await consolidator.consolidate_simple(new, existing)
        assert result.action == "keep_new"
