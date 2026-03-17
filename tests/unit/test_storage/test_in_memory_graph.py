"""Tests for in-memory graph store."""

import pytest

from pymem.storage.graph.in_memory_graph import InMemoryGraphStore
from pymem.storage.base import GraphNode, GraphEdge


@pytest.fixture
async def store():
    s = InMemoryGraphStore()
    await s.initialize()
    return s


@pytest.mark.asyncio
class TestInMemoryGraphStore:
    async def test_upsert_and_get_node(self, store):
        node = GraphNode(
            id="n1",
            label="Person",
            properties={"name": "Alice"},
            scope={"user_id": "u1"},
        )
        await store.upsert_node(node)
        result = await store.get_node("n1")
        assert result is not None
        assert result.label == "Person"

    async def test_get_nonexistent_node(self, store):
        result = await store.get_node("nonexistent")
        assert result is None

    async def test_upsert_edge(self, store):
        await store.upsert_node(
            GraphNode(id="n1", label="Person", scope={"user_id": "u1"})
        )
        await store.upsert_node(
            GraphNode(id="n2", label="Topic", scope={"user_id": "u1"})
        )
        await store.upsert_edge(
            GraphEdge(source_id="n1", target_id="n2", relationship="KNOWS")
        )

        result = await store.search_neighbors("n1")
        assert len(result.nodes) > 0

    async def test_delete_node(self, store):
        await store.upsert_node(
            GraphNode(id="n1", label="Test", scope={"user_id": "u1"})
        )
        await store.delete_node("n1")
        assert await store.get_node("n1") is None

    async def test_delete_node_removes_edges(self, store):
        await store.upsert_node(
            GraphNode(id="n1", label="A", scope={"user_id": "u1"})
        )
        await store.upsert_node(
            GraphNode(id="n2", label="B", scope={"user_id": "u1"})
        )
        await store.upsert_edge(
            GraphEdge(source_id="n1", target_id="n2", relationship="REL")
        )
        await store.delete_node("n1")
        # Edge should be removed too
        result = await store.search_neighbors("n2")
        edges = result.edges
        assert len(edges) == 0

    async def test_episodic_timeline(self, store):
        for i in range(5):
            await store.upsert_node(
                GraphNode(
                    id=f"e{i}",
                    label="Event",
                    properties={"content": f"Event {i}", "created_at": f"2024-01-{i+1:02d}"},
                    scope={"user_id": "u1"},
                )
            )

        events = await store.get_episodic_timeline(
            scope={"user_id": "u1"}, limit=3
        )
        assert len(events) == 3

    async def test_delete_by_scope(self, store):
        await store.upsert_node(
            GraphNode(id="n1", label="A", scope={"user_id": "u1"})
        )
        await store.upsert_node(
            GraphNode(id="n2", label="B", scope={"user_id": "u1"})
        )
        await store.upsert_node(
            GraphNode(id="n3", label="C", scope={"user_id": "u2"})
        )

        deleted = await store.delete_by_scope({"user_id": "u1"})
        assert deleted == 2
        assert await store.get_node("n3") is not None

    async def test_health_check(self, store):
        assert await store.health_check() is True
