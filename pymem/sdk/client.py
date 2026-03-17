"""Async Python SDK — thin wrapper around the PyMem REST API.

Usage:
    memory = PyMem(api_key="sk-...")
    await memory.add(messages, user_id="u1", agent_id="a1")
    facts = await memory.search("kafka experience", user_id="u1")
    ctx = await memory.get_context(user_id="u1", query="banking project")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx


@dataclass
class Memory:
    """A single memory record from the API."""

    id: str
    memory_type: str
    content: str
    score: float = 0.5
    access_count: int = 0
    created_at: datetime | None = None
    user_id: str | None = None
    agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AddResult:
    status: str
    memories_count: int = 0
    extraction_id: str | None = None
    memories: list[Memory] = field(default_factory=list)


@dataclass
class AgentContext:
    formatted: str = ""
    working_messages: list[dict[str, Any]] = field(default_factory=list)
    memories: list[Memory] = field(default_factory=list)
    instructions: list[Memory] = field(default_factory=list)


@dataclass
class ForgetResult:
    deleted_per_type: dict[str, int] = field(default_factory=dict)
    completed_at: datetime | None = None
    confirmation_id: str = ""


class PyMem:
    """Async Python SDK. Mirrors the API surface exactly.

    Six operations:
    1. add() — Store memories from a conversation
    2. search() — Search across all memory types
    3. get_context() — Get assembled agent context
    4. update() — Update a specific memory
    5. delete() — Delete a memory
    6. forget() — GDPR-compliant wipe
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8001",
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
        memory_types: list[str] | None = None,
        async_extraction: bool = False,
    ) -> AddResult:
        """Store memories from a conversation."""
        resp = await self._client.post(
            "/v1/memory/add",
            json={
                "messages": messages,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "org_id": org_id,
                "memory_types": memory_types,
                "async_extraction": async_extraction,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return AddResult(
            status=data["status"],
            memories_count=data.get("memories_count", 0),
            memories=[Memory(**m) for m in data.get("memories", [])],
        )

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        """Search across all memory types."""
        resp = await self._client.post(
            "/v1/memory/search",
            json={
                "query": query,
                "user_id": user_id,
                "agent_id": agent_id,
                "types": types,
                "limit": limit,
            },
        )
        resp.raise_for_status()
        return [Memory(**m) for m in resp.json()["memories"]]

    async def get_context(
        self,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        query: str | None = None,
        max_tokens: int = 4000,
    ) -> AgentContext:
        """Get assembled agent context for prompt injection."""
        resp = await self._client.get(
            "/v1/memory/context/assemble",
            params={
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "query": query,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return AgentContext(
            formatted=data.get("formatted_prompt", ""),
            working_messages=data.get("working_messages", []),
            memories=[Memory(**m) for m in data.get("memories", [])],
            instructions=[Memory(**m) for m in data.get("instructions", [])],
        )

    async def update(
        self,
        memory_id: str,
        content: str,
        user_id: str,
    ) -> Memory:
        """Update a specific memory (creates new version)."""
        resp = await self._client.put(
            f"/v1/memory/{memory_id}",
            json={"content": content},
            params={"user_id": user_id},
        )
        resp.raise_for_status()
        return Memory(**resp.json())

    async def delete(
        self,
        memory_id: str,
        user_id: str,
        hard: bool = False,
    ) -> None:
        """Delete a memory (soft by default)."""
        resp = await self._client.delete(
            f"/v1/memory/{memory_id}",
            params={"user_id": user_id, "hard": hard},
        )
        resp.raise_for_status()

    async def forget(
        self,
        user_id: str,
        scope: str = "all",
    ) -> ForgetResult:
        """GDPR-compliant forget — wipes all backends."""
        resp = await self._client.post(
            f"/v1/users/{user_id}/forget",
            json={"scope": scope},
        )
        resp.raise_for_status()
        data = resp.json()
        return ForgetResult(
            deleted_per_type=data.get("deleted_per_type", {}),
            confirmation_id=data.get("confirmation_id", ""),
        )

    async def share(
        self,
        memory_id: str,
        from_scope: dict[str, str],
        to_scope: dict[str, str],
    ) -> Memory:
        """Clone memory into a different scope."""
        resp = await self._client.post(
            "/v1/memory/share",
            json={
                "memory_id": memory_id,
                "from_scope": from_scope,
                "to_scope": to_scope,
            },
        )
        resp.raise_for_status()
        return Memory(**resp.json())

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> PyMem:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
