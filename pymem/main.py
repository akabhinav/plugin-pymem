"""FastAPI app factory + lifespan — the entry point for PyMem API."""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import PlainTextResponse

from pymem.api.v1 import admin, agents, health, memories, sessions, users
from pymem.api.v1.dependencies import set_engine
from pymem.cache.redis_cache import InMemoryCache
from pymem.config.settings import get_settings
from pymem.engine.memory_engine import MemoryEngine
from pymem.storage.graph.in_memory_graph import InMemoryGraphStore
from pymem.storage.relational.sqlite_store import SQLiteRelationalStore

# Import to trigger plugin registration
import pymem.embeddings.local_embeddings  # noqa: F401
import pymem.storage.graph.in_memory_graph  # noqa: F401

logger = structlog.get_logger()


async def _build_engine() -> MemoryEngine:
    """Build the memory engine with configured backends."""
    settings = get_settings()

    # Vector store
    vector_store = await _build_vector_store(settings)

    # Graph store
    graph_store = await _build_graph_store(settings)

    # Relational store
    relational = SQLiteRelationalStore(db_path="./data/pymem.db")
    await relational.initialize()

    # Embedding provider
    from pymem.storage.registry import get_embedding_provider

    embedding = get_embedding_provider(settings.EMBEDDING_PROVIDER)

    # Cache
    cache = InMemoryCache()
    await cache.initialize()

    # Intelligence layer (optional — depends on PyGate availability)
    extractor = None
    consolidator = None
    graph_builder = None

    try:
        from pymem.intelligence.pygate_client import PyGateClient
        from pymem.intelligence.extractor import MemoryExtractor
        from pymem.intelligence.consolidator import MemoryConsolidator
        from pymem.intelligence.graph_builder import GraphBuilder

        pygate = PyGateClient(
            url=settings.PYGATE_URL,
            api_key=settings.PYGATE_API_KEY.get_secret_value(),
            model=settings.EXTRACTION_MODEL,
        )
        extractor = MemoryExtractor(pygate, min_score=settings.MIN_MEMORY_SCORE)
        consolidator = MemoryConsolidator(pygate)
        graph_builder = GraphBuilder(pygate)
    except Exception:
        logger.warning("PyGate unavailable — running without LLM intelligence")

    return MemoryEngine(
        vector_store=vector_store,
        graph_store=graph_store,
        relational_store=relational,
        embedding_provider=embedding,
        extractor=extractor,
        consolidator=consolidator,
        graph_builder=graph_builder,
        cache=cache,
    )


async def _build_vector_store(settings):
    """Build vector store based on config. Falls back to in-memory graph for search."""
    backend = settings.VECTOR_STORE_BACKEND

    if backend == "pgvector":
        try:
            from pymem.storage.registry import get_vector_store

            store = get_vector_store(
                "pgvector",
                dsn=settings.DATABASE_URL.get_secret_value().replace(
                    "postgresql+asyncpg://", "postgresql://"
                ),
            )
            await store.initialize()
            return store
        except Exception:
            logger.warning("pgvector unavailable, using in-memory vector store")

    # Fallback: simple in-memory vector store
    from pymem.storage.vector.in_memory_vector import InMemoryVectorStore

    store = InMemoryVectorStore()
    await store.initialize()
    return store


async def _build_graph_store(settings):
    """Build graph store based on config."""
    backend = settings.GRAPH_STORE_BACKEND

    if backend == "kuzu":
        try:
            from pymem.storage.registry import get_graph_store

            store = get_graph_store("kuzu", db_path=settings.KUZU_DB_PATH)
            await store.initialize()
            return store
        except Exception:
            logger.warning("Kuzu unavailable, using in-memory graph store")

    store = InMemoryGraphStore()
    await store.initialize()
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize and tear down resources."""
    logger.info("Starting PyMem", version="1.0.0")
    engine = await _build_engine()
    set_engine(engine)
    logger.info("PyMem engine initialized")

    yield

    logger.info("Shutting down PyMem")
    await engine._vector.shutdown()
    await engine._graph.shutdown()
    await engine._relational.shutdown()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="PyMem",
        description="Four-type memory platform for AI agents",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics endpoint
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        try:
            from prometheus_client import generate_latest

            return generate_latest()
        except ImportError:
            return PlainTextResponse("prometheus_client not installed", status_code=501)

    # Register routers
    app.include_router(health.router)
    app.include_router(memories.router)
    app.include_router(users.router)
    app.include_router(agents.router)
    app.include_router(sessions.router)
    app.include_router(admin.router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("pymem.main:app", host="0.0.0.0", port=8001, reload=True)
