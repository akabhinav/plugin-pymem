"""Celery task for async memory extraction."""

from __future__ import annotations

import asyncio
import logging

from pymem.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="pymem.tasks.extraction_task.extract_memories",
    bind=True,
    max_retries=3,
)
def extract_memories(self, messages: list[dict], scope: dict, types: list[str]):
    """Async memory extraction via Celery.

    Runs the full extraction pipeline without blocking the API response.
    """
    try:
        asyncio.run(_run_extraction(messages, scope, types))
    except Exception as exc:
        logger.exception("Extraction task failed")
        self.retry(exc=exc, countdown=60)


async def _run_extraction(messages: list[dict], scope: dict, types: list[str]):
    """Run the extraction pipeline asynchronously."""
    from pymem.models import MemoryScope, MemoryType

    # Build a minimal engine for the worker
    from pymem.cache.redis_cache import InMemoryCache
    from pymem.storage.graph.in_memory_graph import InMemoryGraphStore
    from pymem.storage.relational.sqlite_store import SQLiteRelationalStore
    from pymem.storage.vector.in_memory_vector import InMemoryVectorStore
    from pymem.storage.registry import get_embedding_provider
    from pymem.engine.memory_engine import MemoryEngine

    vector = InMemoryVectorStore()
    await vector.initialize()
    graph = InMemoryGraphStore()
    await graph.initialize()
    relational = SQLiteRelationalStore("./data/pymem.db")
    await relational.initialize()
    embedding = get_embedding_provider("local")
    cache = InMemoryCache()

    engine = MemoryEngine(
        vector_store=vector,
        graph_store=graph,
        relational_store=relational,
        embedding_provider=embedding,
        cache=cache,
    )

    memory_scope = MemoryScope(**{k: v for k, v in scope.items() if v})
    memory_types = [MemoryType(t) for t in types] if types else None

    await engine.add(messages, memory_scope, memory_types)
    await relational.shutdown()
