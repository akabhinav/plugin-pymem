"""Engine factory for Streamlit UI — creates engine on first call, caches it."""

from __future__ import annotations

import asyncio
from typing import Any

import streamlit as st


@st.cache_resource
def get_or_create_engine() -> Any:
    """Create and cache the memory engine for the Streamlit session."""
    loop = asyncio.new_event_loop()
    engine = loop.run_until_complete(_create_engine())
    return engine


async def _create_engine():
    """Build a minimal engine for the Streamlit UI."""
    from pymem.cache.redis_cache import InMemoryCache
    from pymem.engine.memory_engine import MemoryEngine
    from pymem.storage.graph.in_memory_graph import InMemoryGraphStore
    from pymem.storage.relational.sqlite_store import SQLiteRelationalStore
    from pymem.storage.vector.in_memory_vector import InMemoryVectorStore

    # Import to register plugins
    import pymem.embeddings.local_embeddings  # noqa: F401
    import pymem.storage.vector.in_memory_vector  # noqa: F401
    import pymem.storage.graph.in_memory_graph  # noqa: F401

    from pymem.storage.registry import get_embedding_provider

    # Use in-memory backends for the UI (no external deps needed)
    vector = InMemoryVectorStore()
    await vector.initialize()

    graph = InMemoryGraphStore()
    await graph.initialize()

    relational = SQLiteRelationalStore("./data/pymem_ui.db")
    await relational.initialize()

    embedding = get_embedding_provider("local")

    cache = InMemoryCache()
    await cache.initialize()

    return MemoryEngine(
        vector_store=vector,
        graph_store=graph,
        relational_store=relational,
        embedding_provider=embedding,
        cache=cache,
    )
