"""E2E tests for all FastAPI API endpoints via TestClient."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from pymem.api.v1 import admin, agents, health, memories, sessions, users
from pymem.api.v1.dependencies import set_engine
from pymem.cache.redis_cache import InMemoryCache
from pymem.engine.memory_engine import MemoryEngine
from pymem.storage.graph.in_memory_graph import InMemoryGraphStore
from pymem.storage.relational.sqlite_store import SQLiteRelationalStore
from pymem.storage.vector.in_memory_vector import InMemoryVectorStore

# Ensure plugins are registered
import pymem.embeddings.local_embeddings  # noqa: F401
import pymem.storage.graph.in_memory_graph  # noqa: F401
import pymem.storage.vector.in_memory_vector  # noqa: F401

from pymem.storage.registry import get_embedding_provider


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _create_test_app(engine: MemoryEngine) -> FastAPI:
    """Create a FastAPI app for testing — no lifespan that calls _build_engine."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        set_engine(engine)
        yield

    app = FastAPI(title="PyMem Test", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(memories.router)
    app.include_router(users.router)
    app.include_router(agents.router)
    app.include_router(sessions.router)
    app.include_router(admin.router)
    return app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a real in-memory engine."""
    async def _setup():
        vector = InMemoryVectorStore()
        await vector.initialize()
        graph = InMemoryGraphStore()
        await graph.initialize()
        relational = SQLiteRelationalStore(":memory:")
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

    engine = _run(_setup())
    app = _create_test_app(engine)

    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["version"] == "1.0.0"
        assert "vector_store" in data
        assert "graph_store" in data


# ──────────────────────────────────────────────────────────────
# Memory CRUD
# ──────────────────────────────────────────────────────────────
class TestMemoryEndpoints:
    def test_add_memory(self, client):
        resp = client.post("/v1/memory/add", json={
            "messages": [
                {"role": "user", "content": "I have 10 years of Python experience"},
            ],
            "user_id": "api_user_1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"
        assert data["memories_count"] >= 1

    def test_add_memory_requires_messages(self, client):
        resp = client.post("/v1/memory/add", json={
            "messages": [],
            "user_id": "api_user_1",
        })
        assert resp.status_code == 422

    def test_search_memory(self, client):
        # First add something
        client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I love using Kubernetes for deployments"}],
            "user_id": "search_api_user",
        })

        resp = client.post("/v1/memory/search", json={
            "query": "container orchestration",
            "user_id": "search_api_user",
            "limit": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "memories" in data
        assert "total" in data

    def test_search_memory_requires_query(self, client):
        resp = client.post("/v1/memory/search", json={
            "query": "",
            "user_id": "u1",
        })
        assert resp.status_code == 422

    def test_get_memory(self, client):
        # Add a memory first
        add_resp = client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I work at Acme Corp as a senior engineer"}],
            "user_id": "get_api_user",
        })
        memories = add_resp.json()["memories"]
        if memories:
            mem_id = memories[0]["id"]
            resp = client.get(f"/v1/memory/{mem_id}")
            assert resp.status_code == 200
            assert resp.json()["content"] is not None

    def test_get_memory_not_found(self, client):
        resp = client.get("/v1/memory/nonexistent-id-12345")
        assert resp.status_code == 404

    def test_update_memory(self, client):
        # Add then update
        add_resp = client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I prefer tabs over spaces for indentation"}],
            "user_id": "update_api_user",
        })
        memories = add_resp.json()["memories"]
        if memories:
            mem_id = memories[0]["id"]
            resp = client.put(
                f"/v1/memory/{mem_id}?user_id=update_api_user",
                json={"content": "I now prefer spaces over tabs"},
            )
            assert resp.status_code == 200
            assert resp.json()["version"] == 2

    def test_update_memory_not_found(self, client):
        resp = client.put(
            "/v1/memory/nonexistent-id?user_id=u1",
            json={"content": "new"},
        )
        assert resp.status_code == 404

    def test_delete_memory_soft(self, client):
        add_resp = client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I enjoy cooking Italian food on weekends"}],
            "user_id": "delete_api_user",
        })
        memories = add_resp.json()["memories"]
        if memories:
            mem_id = memories[0]["id"]
            resp = client.delete(f"/v1/memory/{mem_id}?user_id=delete_api_user")
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"

            # Verify it's gone
            get_resp = client.get(f"/v1/memory/{mem_id}")
            assert get_resp.status_code == 404

    def test_delete_memory_hard(self, client):
        add_resp = client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I have been learning Rust programming recently"}],
            "user_id": "hard_del_api_user",
        })
        memories = add_resp.json()["memories"]
        if memories:
            mem_id = memories[0]["id"]
            resp = client.delete(
                f"/v1/memory/{mem_id}?user_id=hard_del_api_user&hard=true"
            )
            assert resp.status_code == 200

    def test_share_memory(self, client):
        # Add a memory for user_a
        add_resp = client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I prefer functional programming paradigms"}],
            "user_id": "share_user_a",
        })
        memories = add_resp.json()["memories"]
        if memories:
            mem_id = memories[0]["id"]
            resp = client.post("/v1/memory/share", json={
                "memory_id": mem_id,
                "from_scope": {"user_id": "share_user_a"},
                "to_scope": {"user_id": "share_user_b"},
            })
            assert resp.status_code == 200
            assert resp.json()["id"] != mem_id

    def test_share_memory_not_found(self, client):
        resp = client.post("/v1/memory/share", json={
            "memory_id": "nonexistent",
            "from_scope": {"user_id": "a"},
            "to_scope": {"user_id": "b"},
        })
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────
# Context Assembly
# ──────────────────────────────────────────────────────────────
class TestContextEndpoint:
    def test_assemble_context(self, client):
        # Add some data first
        client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I specialize in distributed systems design"}],
            "user_id": "ctx_api_user",
            "agent_id": "ctx_api_agent",
            "session_id": "ctx_session",
        })

        resp = client.get(
            "/v1/memory/context/assemble",
            params={
                "user_id": "ctx_api_user",
                "agent_id": "ctx_api_agent",
                "query": "tech background",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "formatted_prompt" in data
        assert "memories" in data
        assert "instructions" in data


# ──────────────────────────────────────────────────────────────
# Users
# ──────────────────────────────────────────────────────────────
class TestUserEndpoints:
    def test_list_user_memories(self, client):
        client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I enjoy hiking in the mountains every weekend"}],
            "user_id": "list_user",
        })
        resp = client.get("/v1/users/list_user/memories")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_user_memories_with_type_filter(self, client):
        resp = client.get("/v1/users/list_user/memories?type=semantic")
        assert resp.status_code == 200

    def test_list_user_memories_pagination(self, client):
        resp = client.get("/v1/users/list_user/memories?limit=1&offset=0")
        assert resp.status_code == 200

    def test_delete_user_memories(self, client):
        client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I am training for a marathon this year"}],
            "user_id": "bulk_del_user",
        })
        resp = client.delete("/v1/users/bulk_del_user/memories")
        assert resp.status_code == 200
        assert "deleted_per_type" in resp.json()

    def test_forget_user(self, client):
        client.post("/v1/memory/add", json={
            "messages": [{"role": "user", "content": "I have a golden retriever named Max at home"}],
            "user_id": "forget_api_user",
        })
        resp = client.post(
            "/v1/users/forget_api_user/forget",
            json={"scope": "all"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "confirmation_id" in data
        assert "completed_at" in data

    def test_forget_user_specific_type(self, client):
        resp = client.post(
            "/v1/users/forget_api_user/forget",
            json={"scope": "semantic"},
        )
        assert resp.status_code == 200

    def test_get_user_entities(self, client):
        resp = client.get("/v1/users/list_user/entities")
        assert resp.status_code == 200
        data = resp.json()
        assert "user_id" in data
        assert "entities" in data


# ──────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────
class TestAgentEndpoints:
    def test_add_agent_instruction(self, client):
        resp = client.post("/v1/agents/api_agent_1/instructions", json={
            "content": "Always respond in TypeScript",
            "priority": 2,
            "tags": ["coding", "language"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Always respond in TypeScript"
        assert data["memory_type"] == "procedural"

    def test_get_agent_instructions(self, client):
        # Add first
        client.post("/v1/agents/api_agent_2/instructions", json={
            "content": "Use functional programming patterns",
            "priority": 1,
        })
        resp = client.get("/v1/agents/api_agent_2/instructions")
        assert resp.status_code == 200
        instructions = resp.json()
        assert isinstance(instructions, list)
        assert len(instructions) >= 1

    def test_get_agent_instructions_empty(self, client):
        resp = client.get("/v1/agents/nonexistent_agent/instructions")
        assert resp.status_code == 200
        assert resp.json() == []


# ──────────────────────────────────────────────────────────────
# Sessions / Working Memory
# ──────────────────────────────────────────────────────────────
class TestSessionEndpoints:
    def test_append_working_memory(self, client):
        resp = client.post("/v1/sessions/api_session_1/working", json={
            "messages": [
                {"role": "user", "content": "What is Docker?"},
                {"role": "assistant", "content": "Docker is a containerization platform."},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "api_session_1"
        assert data["message_count"] >= 2

    def test_get_working_memory(self, client):
        # Append first
        client.post("/v1/sessions/api_session_2/working", json={
            "messages": [{"role": "user", "content": "Hello world"}],
        })
        resp = client.get("/v1/sessions/api_session_2/working")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "api_session_2"
        assert len(data["messages"]) >= 1

    def test_clear_working_memory(self, client):
        client.post("/v1/sessions/api_session_3/working", json={
            "messages": [{"role": "user", "content": "To be cleared"}],
        })
        resp = client.delete("/v1/sessions/api_session_3/working")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"

        # Verify cleared
        get_resp = client.get("/v1/sessions/api_session_3/working")
        assert get_resp.json()["messages"] == []


# ──────────────────────────────────────────────────────────────
# Admin
# ──────────────────────────────────────────────────────────────
class TestAdminEndpoints:
    def test_get_stats(self, client):
        resp = client.get("/v1/admin/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_memories" in data
        assert "semantic" in data
        assert "unique_users" in data

    def test_get_audit_log(self, client):
        resp = client.get("/v1/admin/audit")
        assert resp.status_code == 200
        assert "audit_entries" in resp.json()

    def test_get_audit_log_with_filters(self, client):
        resp = client.get("/v1/admin/audit?user_id=api_user_1&limit=10")
        assert resp.status_code == 200

    def test_trigger_consolidation(self, client):
        resp = client.post("/v1/admin/consolidate/api_user_1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "memories_reviewed" in data

    def test_bulk_import(self, client):
        resp = client.post("/v1/admin/import", json={
            "memories": [
                {"content": "Imported fact 1", "type": "semantic", "user_id": "import_user"},
                {"content": "Imported fact 2", "type": "episodic", "user_id": "import_user"},
                {"content": "Imported instruction", "type": "procedural", "agent_id": "import_agent"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["imported"] == 3
        assert data["status"] == "completed"

    def test_bulk_import_skips_invalid(self, client):
        resp = client.post("/v1/admin/import", json={
            "memories": [
                {"content": "", "type": "semantic", "user_id": "u1"},  # empty content
                {"content": "No scope", "type": "semantic"},  # no user/agent
                {"content": "Valid one", "type": "semantic", "user_id": "u1"},
            ],
        })
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1
