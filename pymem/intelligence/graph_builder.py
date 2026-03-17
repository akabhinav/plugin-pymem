"""Graph builder — extracts entities and relationships for the graph store."""

from __future__ import annotations

from uuid import uuid4

from pymem.intelligence.pygate_client import PyGateClient
from pymem.models import MemoryRecord
from pymem.storage.base import GraphEdge, GraphNode

GRAPH_EXTRACTION_PROMPT = """You are an entity and relationship extraction specialist.

Given a list of memories, extract entities and relationships.
Return JSON: {
    "entities": [
        {"name": "...", "type": "Person|Topic|Tool|Company|Concept", "description": "..."}
    ],
    "relationships": [
        {"source": "entity_name", "target": "entity_name", "type": "KNOWS|PREFERS|USES|WORKED_ON", "description": "..."}
    ]
}"""


class GraphBuilder:
    """Extracts entities and relationships from memories for graph storage."""

    def __init__(self, pygate: PyGateClient | None = None) -> None:
        self._pygate = pygate

    async def build_from_memories(
        self,
        memories: list[MemoryRecord],
        scope: dict[str, str],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Extract graph nodes and edges from a list of memories."""
        if self._pygate:
            return await self._build_with_llm(memories, scope)
        return self._build_from_metadata(memories, scope)

    async def _build_with_llm(
        self,
        memories: list[MemoryRecord],
        scope: dict[str, str],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        memory_text = "\n".join(f"- {m.content}" for m in memories)

        result = await self._pygate.complete_json(
            system=GRAPH_EXTRACTION_PROMPT,
            user=f"Memories:\n{memory_text}",
        )

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        name_to_id: dict[str, str] = {}

        for entity in result.get("entities", []):
            node_id = str(uuid4())
            name = entity["name"]
            name_to_id[name] = node_id
            nodes.append(
                GraphNode(
                    id=node_id,
                    label=entity.get("type", "Concept"),
                    properties={
                        "name": name,
                        "content": entity.get("description", ""),
                    },
                    scope=scope,
                )
            )

        for rel in result.get("relationships", []):
            src_id = name_to_id.get(rel["source"])
            tgt_id = name_to_id.get(rel["target"])
            if src_id and tgt_id:
                edges.append(
                    GraphEdge(
                        source_id=src_id,
                        target_id=tgt_id,
                        relationship=rel.get("type", "RELATED_TO"),
                        properties={"description": rel.get("description", "")},
                    )
                )

        return nodes, edges

    @staticmethod
    def _build_from_metadata(
        memories: list[MemoryRecord],
        scope: dict[str, str],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Extract graph data from memory metadata entities without LLM."""
        nodes: list[GraphNode] = []
        name_to_id: dict[str, str] = {}

        for memory in memories:
            entities = memory.metadata.get("entities", [])
            for entity_name in entities:
                if entity_name not in name_to_id:
                    node_id = str(uuid4())
                    name_to_id[entity_name] = node_id
                    nodes.append(
                        GraphNode(
                            id=node_id,
                            label="Concept",
                            properties={
                                "name": entity_name,
                                "content": memory.content,
                            },
                            scope=scope,
                        )
                    )

        # Create edges between entities co-occurring in the same memory
        edges: list[GraphEdge] = []
        for memory in memories:
            entities = memory.metadata.get("entities", [])
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    if e1 in name_to_id and e2 in name_to_id:
                        edges.append(
                            GraphEdge(
                                source_id=name_to_id[e1],
                                target_id=name_to_id[e2],
                                relationship="RELATED_TO",
                            )
                        )

        return nodes, edges
