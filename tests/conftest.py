"""Shared test fixtures for PyMem tests."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from pymem.cache.redis_cache import InMemoryCache
from pymem.engine.memory_engine import MemoryEngine
from pymem.models import MemoryScope
from pymem.storage.graph.in_memory_graph import InMemoryGraphStore
from pymem.storage.relational.sqlite_store import SQLiteRelationalStore
from pymem.storage.vector.in_memory_vector import InMemoryVectorStore

# Ensure plugins are registered
import pymem.embeddings.local_embeddings  # noqa: F401
import pymem.storage.graph.in_memory_graph  # noqa: F401
import pymem.storage.vector.in_memory_vector  # noqa: F401

from pymem.storage.registry import get_embedding_provider


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def vector_store():
    store = InMemoryVectorStore()
    await store.initialize()
    yield store


@pytest_asyncio.fixture
async def graph_store():
    store = InMemoryGraphStore()
    await store.initialize()
    yield store


@pytest_asyncio.fixture
async def relational_store():
    store = SQLiteRelationalStore(":memory:")
    await store.initialize()
    yield store
    await store.shutdown()


@pytest_asyncio.fixture
async def cache():
    c = InMemoryCache()
    await c.initialize()
    yield c
    await c.shutdown()


@pytest_asyncio.fixture
def embedding_provider():
    return get_embedding_provider("local")


@pytest_asyncio.fixture
async def engine(vector_store, graph_store, relational_store, embedding_provider, cache):
    eng = MemoryEngine(
        vector_store=vector_store,
        graph_store=graph_store,
        relational_store=relational_store,
        embedding_provider=embedding_provider,
        cache=cache,
    )
    yield eng


@pytest.fixture
def user_scope():
    return MemoryScope(user_id="test_user_1")


@pytest.fixture
def agent_scope():
    return MemoryScope(agent_id="test_agent_1")


@pytest.fixture
def session_scope():
    return MemoryScope(session_id="test_session_1")


@pytest.fixture
def full_scope():
    return MemoryScope(
        user_id="test_user_1",
        agent_id="test_agent_1",
        session_id="test_session_1",
    )
