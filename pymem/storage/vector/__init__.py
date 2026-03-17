"""Vector store plugins — import to trigger registration."""

from pymem.storage.vector.pgvector_store import PgVectorStore

__all__ = ["PgVectorStore"]

# Optional backends — import only if dependencies available
try:
    from pymem.storage.vector.qdrant_store import QdrantStore  # noqa: F401
except ImportError:
    pass
