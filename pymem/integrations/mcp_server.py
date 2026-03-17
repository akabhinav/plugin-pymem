"""MCP server — exposes PyMem as tools to any MCP-compatible agent."""

from __future__ import annotations

from typing import Any


class PyMemMCPServer:
    """MCP tool server for PyMem memory operations.

    Exposes five tools:
    - pymem_add_memory
    - pymem_search
    - pymem_get_context
    - pymem_remember_instruction
    - pymem_forget
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def get_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "pymem_add_memory",
                "description": "Add memories from a conversation to long-term storage",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                            },
                        },
                        "user_id": {"type": "string"},
                        "agent_id": {"type": "string"},
                    },
                    "required": ["messages", "user_id"],
                },
            },
            {
                "name": "pymem_search",
                "description": "Search memories by semantic query across all memory types",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "user_id": {"type": "string"},
                        "types": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query", "user_id"],
                },
            },
            {
                "name": "pymem_get_context",
                "description": "Get assembled agent context from all memory types",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["user_id"],
                },
            },
            {
                "name": "pymem_remember_instruction",
                "description": "Store a procedural instruction for this agent permanently",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["instruction", "agent_id"],
                },
            },
            {
                "name": "pymem_forget",
                "description": "Delete memories matching a scope (GDPR forget)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "scope": {
                            "type": "string",
                            "enum": ["all", "semantic", "episodic", "procedural"],
                        },
                    },
                    "required": ["user_id"],
                },
            },
        ]

    async def handle_tool_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle an MCP tool call."""
        from pymem.models import MemoryScope, MemoryType

        if tool_name == "pymem_add_memory":
            scope = MemoryScope(
                user_id=arguments["user_id"],
                agent_id=arguments.get("agent_id"),
            )
            result = await self._engine.add(
                messages=arguments["messages"], scope=scope
            )
            return {"status": result.status, "count": result.memories_count}

        elif tool_name == "pymem_search":
            scope = MemoryScope(user_id=arguments["user_id"])
            types = (
                [MemoryType(t) for t in arguments["types"]]
                if "types" in arguments
                else None
            )
            result = await self._engine.search(
                query=arguments["query"],
                scope=scope,
                memory_types=types,
                limit=arguments.get("limit", 10),
            )
            return {
                "memories": [
                    {"content": m.content, "type": m.memory_type.value, "score": m.score}
                    for m in result.memories
                ]
            }

        elif tool_name == "pymem_get_context":
            scope = MemoryScope(
                user_id=arguments["user_id"],
                agent_id=arguments.get("agent_id"),
            )
            ctx = await self._engine.get_context(
                scope=scope, query=arguments.get("query")
            )
            return {"context": ctx.formatted}

        elif tool_name == "pymem_remember_instruction":
            scope = MemoryScope(agent_id=arguments["agent_id"])
            await self._engine.procedural.add(
                content=arguments["instruction"],
                scope=scope,
                tags=arguments.get("tags", []),
            )
            return {"status": "stored"}

        elif tool_name == "pymem_forget":
            scope = MemoryScope(user_id=arguments["user_id"])
            scope_type = arguments.get("scope", "all")
            types = [MemoryType(scope_type)] if scope_type != "all" else None
            result = await self._engine.forget(scope=scope, memory_types=types)
            return {"deleted": result.deleted_per_type}

        return {"error": f"Unknown tool: {tool_name}"}
