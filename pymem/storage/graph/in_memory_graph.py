"""In-memory graph store — default for development and testing."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymem.storage.base import GraphEdge, GraphNode, GraphSearchResult, GraphStorePlugin
from pymem.storage.registry import register_graph_store


@register_graph_store("memory")
class InMemoryGraphStore(GraphStorePlugin):
    """Simple in-memory graph for dev/test. No external dependencies."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    async def initialize(self) -> None:
        pass

    async def upsert_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node

    async def upsert_edge(self, edge: GraphEdge) -> None:
        # Replace existing edge between same source/target/relationship
        self._edges = [
            e
            for e in self._edges
            if not (
                e.source_id == edge.source_id
                and e.target_id == edge.target_id
                and e.relationship == edge.relationship
            )
        ]
        self._edges.append(edge)

    async def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    async def delete_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._edges = [
            e for e in self._edges if e.source_id != node_id and e.target_id != node_id
        ]

    async def search_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None = None,
        depth: int = 2,
        limit: int = 20,
    ) -> GraphSearchResult:
        visited: set[str] = set()
        result_nodes: list[GraphNode] = []
        result_edges: list[GraphEdge] = []
        frontier = {node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                if nid in self._nodes:
                    result_nodes.append(self._nodes[nid])

                for edge in self._edges:
                    if relationship_types and edge.relationship not in relationship_types:
                        continue
                    if edge.source_id == nid and edge.target_id not in visited:
                        result_edges.append(edge)
                        next_frontier.add(edge.target_id)
                    elif edge.target_id == nid and edge.source_id not in visited:
                        result_edges.append(edge)
                        next_frontier.add(edge.source_id)

            frontier = next_frontier
            if len(result_nodes) >= limit:
                break

        return GraphSearchResult(
            nodes=result_nodes[:limit], edges=result_edges, paths=[]
        )

    async def get_episodic_timeline(
        self,
        scope: dict[str, str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        events = [
            n
            for n in self._nodes.values()
            if n.label == "Event" and self._matches_scope(n.scope, scope)
        ]
        events.sort(
            key=lambda n: n.properties.get("created_at", ""), reverse=True
        )
        return events[:limit]

    async def get_entity_relationships(
        self, entity_name: str, scope: dict[str, str]
    ) -> GraphSearchResult:
        matching = [
            n
            for n in self._nodes.values()
            if n.properties.get("name", "").lower() == entity_name.lower()
            and self._matches_scope(n.scope, scope)
        ]
        if not matching:
            return GraphSearchResult()

        result = GraphSearchResult(nodes=list(matching), edges=[], paths=[])
        for node in matching:
            for edge in self._edges:
                if edge.source_id == node.id or edge.target_id == node.id:
                    result.edges.append(edge)
        return result

    async def delete_by_scope(self, scope: dict[str, str]) -> int:
        to_delete = [
            nid
            for nid, n in self._nodes.items()
            if self._matches_scope(n.scope, scope)
        ]
        for nid in to_delete:
            await self.delete_node(nid)
        return len(to_delete)

    async def health_check(self) -> bool:
        return True

    @staticmethod
    def _matches_scope(node_scope: dict[str, str], filter_scope: dict[str, str]) -> bool:
        return all(node_scope.get(k) == v for k, v in filter_scope.items())
