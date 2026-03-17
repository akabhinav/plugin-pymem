"""Kuzu graph store — embeddable, zero-server, Apache-2.0."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from pymem.storage.base import GraphEdge, GraphNode, GraphSearchResult, GraphStorePlugin
from pymem.storage.registry import register_graph_store


@register_graph_store("kuzu")
class KuzuStore(GraphStorePlugin):
    """Kuzu: embeddable graph DB, zero external server."""

    def __init__(self, db_path: str = "./data/kuzu") -> None:
        self._db_path = db_path
        self._db: Any = None
        self._conn: Any = None

    async def initialize(self) -> None:
        import kuzu

        self._db = kuzu.Database(self._db_path)
        self._conn = kuzu.Connection(self._db)
        await asyncio.to_thread(self._init_schema)

    def _init_schema(self) -> None:
        try:
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS MemoryNode(
                    id STRING, label STRING, content STRING, name STRING,
                    user_id STRING, agent_id STRING, org_id STRING,
                    created_at TIMESTAMP, PRIMARY KEY(id))
            """)
        except Exception:
            pass
        try:
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS MemoryEdge(
                    FROM MemoryNode TO MemoryNode,
                    relationship STRING, weight DOUBLE, created_at TIMESTAMP)
            """)
        except Exception:
            pass

    async def upsert_node(self, node: GraphNode) -> None:
        await asyncio.to_thread(
            self._conn.execute,
            """
            MERGE (n:MemoryNode {id: $id})
            SET n.label = $label, n.content = $content, n.name = $name,
                n.user_id = $user_id, n.agent_id = $agent_id,
                n.org_id = $org_id, n.created_at = $ts
            """,
            {
                "id": node.id,
                "label": node.label,
                "content": node.properties.get("content", ""),
                "name": node.properties.get("name", ""),
                "user_id": node.scope.get("user_id", ""),
                "agent_id": node.scope.get("agent_id", ""),
                "org_id": node.scope.get("org_id", ""),
                "ts": datetime.utcnow(),
            },
        )

    async def upsert_edge(self, edge: GraphEdge) -> None:
        await asyncio.to_thread(
            self._conn.execute,
            """
            MATCH (a:MemoryNode {id: $src}), (b:MemoryNode {id: $tgt})
            CREATE (a)-[:MemoryEdge {relationship: $rel, weight: $w, created_at: $ts}]->(b)
            """,
            {
                "src": edge.source_id,
                "tgt": edge.target_id,
                "rel": edge.relationship,
                "w": edge.weight,
                "ts": datetime.utcnow(),
            },
        )

    async def get_node(self, node_id: str) -> GraphNode | None:
        result = await asyncio.to_thread(
            self._conn.execute,
            "MATCH (n:MemoryNode {id: $id}) RETURN n.id, n.label, n.content, n.user_id, n.agent_id, n.org_id",
            {"id": node_id},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        return GraphNode(
            id=row[0],
            label=row[1],
            properties={"content": row[2]},
            scope={"user_id": row[3], "agent_id": row[4], "org_id": row[5]},
        )

    async def delete_node(self, node_id: str) -> None:
        await asyncio.to_thread(
            self._conn.execute,
            "MATCH (n:MemoryNode {id: $id}) DETACH DELETE n",
            {"id": node_id},
        )

    async def search_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None = None,
        depth: int = 2,
        limit: int = 20,
    ) -> GraphSearchResult:
        result = await asyncio.to_thread(
            self._conn.execute,
            f"""
            MATCH (start:MemoryNode {{id: $id}})-[r:MemoryEdge*1..{depth}]-(n:MemoryNode)
            RETURN DISTINCT n.id, n.label, n.content, r
            LIMIT $limit
            """,
            {"id": node_id, "limit": limit},
        )
        nodes = []
        while result.has_next():
            row = result.get_next()
            nodes.append(GraphNode(id=row[0], label=row[1], properties={"content": row[2]}))
        return GraphSearchResult(nodes=nodes)

    async def get_episodic_timeline(
        self,
        scope: dict[str, str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        user_id = scope.get("user_id", "")
        result = await asyncio.to_thread(
            self._conn.execute,
            """
            MATCH (n:MemoryNode)
            WHERE n.user_id = $user_id AND n.label = 'Event'
            RETURN n.id, n.label, n.content, n.created_at
            ORDER BY n.created_at DESC
            LIMIT $limit
            """,
            {"user_id": user_id, "limit": limit},
        )
        nodes = []
        while result.has_next():
            row = result.get_next()
            nodes.append(
                GraphNode(
                    id=row[0],
                    label=row[1],
                    properties={"content": row[2], "created_at": str(row[3])},
                    scope=scope,
                )
            )
        return nodes

    async def get_entity_relationships(
        self, entity_name: str, scope: dict[str, str]
    ) -> GraphSearchResult:
        user_id = scope.get("user_id", "")
        result = await asyncio.to_thread(
            self._conn.execute,
            """
            MATCH (n:MemoryNode)-[r:MemoryEdge]-(m:MemoryNode)
            WHERE n.name = $name AND n.user_id = $user_id
            RETURN n.id, n.label, m.id, m.label, r.relationship
            """,
            {"name": entity_name, "user_id": user_id},
        )
        nodes = []
        edges = []
        while result.has_next():
            row = result.get_next()
            nodes.append(GraphNode(id=row[0], label=row[1], scope=scope))
            nodes.append(GraphNode(id=row[2], label=row[3], scope=scope))
            edges.append(GraphEdge(source_id=row[0], target_id=row[2], relationship=row[4]))
        return GraphSearchResult(nodes=nodes, edges=edges)

    async def delete_by_scope(self, scope: dict[str, str]) -> int:
        user_id = scope.get("user_id", "")
        result = await asyncio.to_thread(
            self._conn.execute,
            """
            MATCH (n:MemoryNode)
            WHERE n.user_id = $user_id
            RETURN count(n)
            """,
            {"user_id": user_id},
        )
        count = 0
        if result.has_next():
            count = result.get_next()[0]
        await asyncio.to_thread(
            self._conn.execute,
            "MATCH (n:MemoryNode) WHERE n.user_id = $user_id DETACH DELETE n",
            {"user_id": user_id},
        )
        return count

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(self._conn.execute, "RETURN 1")
            return True
        except Exception:
            return False
