"""Plugin registry — decorators for registering storage backends.

Adding a new backend:
1. Create the plugin class implementing the ABC
2. Apply @register_*("name") decorator
3. Set the env var to the name
4. Zero other changes
"""

from __future__ import annotations

from typing import Any

from pymem.storage.base import EmbeddingPlugin, GraphStorePlugin, VectorStorePlugin

_vector_backends: dict[str, type[VectorStorePlugin]] = {}
_graph_backends: dict[str, type[GraphStorePlugin]] = {}
_embed_providers: dict[str, type[EmbeddingPlugin]] = {}


def register_vector_store(name: str):
    """Decorator to register a vector store plugin."""

    def decorator(cls: type[VectorStorePlugin]):
        _vector_backends[name] = cls
        return cls

    return decorator


def register_graph_store(name: str):
    """Decorator to register a graph store plugin."""

    def decorator(cls: type[GraphStorePlugin]):
        _graph_backends[name] = cls
        return cls

    return decorator


def register_embedding_provider(name: str):
    """Decorator to register an embedding provider plugin."""

    def decorator(cls: type[EmbeddingPlugin]):
        _embed_providers[name] = cls
        return cls

    return decorator


def get_vector_store(name: str, **kwargs: Any) -> VectorStorePlugin:
    """Instantiate a registered vector store by name."""
    if name not in _vector_backends:
        available = ", ".join(_vector_backends.keys()) or "none"
        raise ValueError(f"Unknown vector store '{name}'. Available: {available}")
    return _vector_backends[name](**kwargs)


def get_graph_store(name: str, **kwargs: Any) -> GraphStorePlugin:
    """Instantiate a registered graph store by name."""
    if name not in _graph_backends:
        available = ", ".join(_graph_backends.keys()) or "none"
        raise ValueError(f"Unknown graph store '{name}'. Available: {available}")
    return _graph_backends[name](**kwargs)


def get_embedding_provider(name: str, **kwargs: Any) -> EmbeddingPlugin:
    """Instantiate a registered embedding provider by name."""
    if name not in _embed_providers:
        available = ", ".join(_embed_providers.keys()) or "none"
        raise ValueError(f"Unknown embedding provider '{name}'. Available: {available}")
    return _embed_providers[name](**kwargs)


def list_vector_stores() -> list[str]:
    return list(_vector_backends.keys())


def list_graph_stores() -> list[str]:
    return list(_graph_backends.keys())


def list_embedding_providers() -> list[str]:
    return list(_embed_providers.keys())
