"""Neo4j graph store plugin — full Neo4j for production."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymem.storage.base import GraphEdge, GraphNode, GraphSearchResult, GraphStorePlugin
from pymem.storage.registry import register_graph_store


@register_graph_store("neo4j")
class Neo4jStore(GraphStorePlugin):
    """Full Neo4j for production graph memory. Uses async neo4j driver."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Any = None

    async def initialize(self) -> None:
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        async with self._driver.session() as session:
            await session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:MemoryNode) ON (n.id)"
            )
            await session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:MemoryNode) ON (n.user_id)"
            )

    async def upsert_node(self, node: GraphNode) -> None:
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (n:MemoryNode {id: $id})
                SET n += $props, n.updated_at = datetime()
                """,
                id=node.id,
                props={
                    **node.properties,
                    **node.scope,
                    "label": node.label,
                },
            )

    async def upsert_edge(self, edge: GraphEdge) -> None:
        async with self._driver.session() as session:
            await session.run(
                f"""
                MATCH (a:MemoryNode {{id: $src}}), (b:MemoryNode {{id: $tgt}})
                MERGE (a)-[r:{edge.relationship}]->(b)
                SET r.weight = $weight, r.updated_at = datetime()
                """,
                src=edge.source_id,
                tgt=edge.target_id,
                weight=edge.weight,
            )

    async def get_node(self, node_id: str) -> GraphNode | None:
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (n:MemoryNode {id: $id}) RETURN n", id=node_id
            )
            record = await result.single()
            if not record:
                return None
            props = dict(record["n"])
            return GraphNode(
                id=props.pop("id"),
                label=props.pop("label", ""),
                properties=props,
                scope={
                    k: props.get(k, "")
                    for k in ("user_id", "agent_id", "org_id")
                    if props.get(k)
                },
            )

    async def delete_node(self, node_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                "MATCH (n:MemoryNode {id: $id}) DETACH DELETE n", id=node_id
            )

    async def search_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None = None,
        depth: int = 2,
        limit: int = 20,
    ) -> GraphSearchResult:
        rel_filter = "|".join(relationship_types) if relationship_types else ""
        rel_pattern = f"[r:{rel_filter}]" if rel_filter else "[r]"

        async with self._driver.session() as session:
            result = await session.run(
                f"""
                MATCH path = (start:MemoryNode {{id: $id}})-{rel_pattern}*1..{depth}-(n)
                RETURN DISTINCT n.id as nid, n.label as label, n as props
                LIMIT $limit
                """,
                id=node_id,
                limit=limit,
            )
            nodes = []
            async for record in result:
                nodes.append(
                    GraphNode(
                        id=record["nid"],
                        label=record["label"] or "",
                        properties=dict(record["props"]),
                    )
                )
        return GraphSearchResult(nodes=nodes)

    async def get_episodic_timeline(
        self,
        scope: dict[str, str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        user_id = scope.get("user_id", "")
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (n:MemoryNode)
                WHERE n.user_id = $user_id AND n.label = 'Event'
                RETURN n ORDER BY n.created_at DESC LIMIT $limit
                """,
                user_id=user_id,
                limit=limit,
            )
            nodes = []
            async for record in result:
                props = dict(record["n"])
                nodes.append(
                    GraphNode(
                        id=props.get("id", ""),
                        label="Event",
                        properties=props,
                        scope=scope,
                    )
                )
        return nodes

    async def get_entity_relationships(
        self, entity_name: str, scope: dict[str, str]
    ) -> GraphSearchResult:
        user_id = scope.get("user_id", "")
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (n:MemoryNode)-[r]-(m:MemoryNode)
                WHERE n.name = $name AND n.user_id = $user_id
                RETURN n, r, m
                """,
                name=entity_name,
                user_id=user_id,
            )
            nodes = []
            edges = []
            async for record in result:
                n_props = dict(record["n"])
                m_props = dict(record["m"])
                nodes.append(GraphNode(id=n_props["id"], label=n_props.get("label", "")))
                nodes.append(GraphNode(id=m_props["id"], label=m_props.get("label", "")))
                edges.append(
                    GraphEdge(
                        source_id=n_props["id"],
                        target_id=m_props["id"],
                        relationship=record["r"].type,
                    )
                )
        return GraphSearchResult(nodes=nodes, edges=edges)

    async def delete_by_scope(self, scope: dict[str, str]) -> int:
        user_id = scope.get("user_id", "")
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (n:MemoryNode) WHERE n.user_id = $user_id
                WITH n, count(n) as cnt
                DETACH DELETE n
                RETURN cnt
                """,
                user_id=user_id,
            )
            record = await result.single()
            return record["cnt"] if record else 0

    async def health_check(self) -> bool:
        try:
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        if self._driver:
            await self._driver.close()
