"""Abstract plugin interfaces — THE CONTRACT.

Every storage backend implements one of these interfaces.
The engine never calls any DB client directly — only these interfaces.

Rules:
- All methods are async.
- Scoping is handled via metadata filters.
- Embeddings are pre-computed before calling vector methods.
- Plugins never call PyGate or any LLM directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# --- Vector Store ---


@dataclass
class VectorSearchResult:
    """A single result from a vector similarity search."""

    memory_id: str
    content: str
    score: float  # Cosine similarity (0.0 – 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStorePlugin(ABC):
    """Interface for all vector store backends (pgvector, Qdrant, Chroma, Milvus)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Set up connections, create tables/collections if needed."""

    @abstractmethod
    async def upsert(
        self,
        memory_id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Insert or update a vector record."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        filters: dict[str, Any],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Return top-k similar memories matching scope filter."""

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """Hard-delete a vector record."""

    @abstractmethod
    async def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """Bulk delete matching scope. Returns count deleted."""

    @abstractmethod
    async def get(self, memory_id: str) -> VectorSearchResult | None:
        """Retrieve a single record by ID."""

    @abstractmethod
    async def count(self, filters: dict[str, Any]) -> int:
        """Count records matching scope filter."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if backend is reachable and healthy."""

    async def shutdown(self) -> None:
        """Clean up connections."""


# --- Graph Store ---


@dataclass
class GraphNode:
    """A node in the memory graph."""

    id: str
    label: str  # Person, Topic, Preference, Event
    properties: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, str] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge between two graph nodes."""

    source_id: str
    target_id: str
    relationship: str  # KNOWS, PREFERS, WORKED_ON, HAPPENED_BEFORE
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class GraphSearchResult:
    """Result of a graph traversal or search."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)


class GraphStorePlugin(ABC):
    """Interface for all graph store backends (Kuzu, Neo4j, Memgraph)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Set up schema, indexes, connections."""

    @abstractmethod
    async def upsert_node(self, node: GraphNode) -> None:
        ...

    @abstractmethod
    async def upsert_edge(self, edge: GraphEdge) -> None:
        ...

    @abstractmethod
    async def get_node(self, node_id: str) -> GraphNode | None:
        ...

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        ...

    @abstractmethod
    async def search_neighbors(
        self,
        node_id: str,
        relationship_types: list[str] | None = None,
        depth: int = 2,
        limit: int = 20,
    ) -> GraphSearchResult:
        ...

    @abstractmethod
    async def get_episodic_timeline(
        self,
        scope: dict[str, str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Get time-ordered events for episodic memory."""

    @abstractmethod
    async def get_entity_relationships(
        self,
        entity_name: str,
        scope: dict[str, str],
    ) -> GraphSearchResult:
        """Get all relationships for a named entity."""

    @abstractmethod
    async def delete_by_scope(self, scope: dict[str, str]) -> int:
        """Delete all graph data matching scope. Returns count."""

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    async def shutdown(self) -> None:
        """Clean up connections."""


# --- Embedding Provider ---


class EmbeddingPlugin(ABC):
    """Interface for embedding providers (local or remote via PyGate)."""

    dimension: int = 384

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    def get_dimension(self) -> int:
        return self.dimension
