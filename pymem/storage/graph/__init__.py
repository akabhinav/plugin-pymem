"""Graph store plugins — import to trigger registration."""

from pymem.storage.graph.in_memory_graph import InMemoryGraphStore

__all__ = ["InMemoryGraphStore"]

try:
    from pymem.storage.graph.kuzu_store import KuzuStore  # noqa: F401
except ImportError:
    pass

try:
    from pymem.storage.graph.neo4j_store import Neo4jStore  # noqa: F401
except ImportError:
    pass
