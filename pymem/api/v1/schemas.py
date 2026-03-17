"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Request schemas ---


class AddMemoryRequest(BaseModel):
    messages: list[dict[str, str]] = Field(..., min_length=1)
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    org_id: str | None = None
    memory_types: list[str] | None = None
    async_extraction: bool = False


class SearchMemoryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    org_id: str | None = None
    types: list[str] | None = None
    limit: int = Field(10, ge=1, le=100)
    min_score: float = 0.0
    include_context: bool = False


class UpdateMemoryRequest(BaseModel):
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class ForgetRequest(BaseModel):
    scope: str = "all"  # all | semantic | episodic | procedural
    before: datetime | None = None


class AddInstructionRequest(BaseModel):
    content: str = Field(..., min_length=1)
    priority: int = Field(5, ge=1, le=10)
    tags: list[str] = Field(default_factory=list)


class WorkingMemoryRequest(BaseModel):
    messages: list[dict[str, str]] = Field(..., min_length=1)


class ShareMemoryRequest(BaseModel):
    memory_id: str
    from_scope: dict[str, str]
    to_scope: dict[str, str]


class ImportMemoryRequest(BaseModel):
    memories: list[dict[str, Any]] = Field(..., min_length=1)


# --- Response schemas ---


class MemoryResponse(BaseModel):
    id: str
    memory_type: str
    content: str
    score: float = 0.5
    access_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    org_id: str | None = None
    version: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddMemoryResponse(BaseModel):
    status: str  # stored | queued
    memories_count: int = 0
    extraction_id: str | None = None
    memories: list[MemoryResponse] = Field(default_factory=list)


class SearchMemoryResponse(BaseModel):
    memories: list[MemoryResponse] = Field(default_factory=list)
    total: int = 0
    context_string: str | None = None


class ContextResponse(BaseModel):
    formatted_prompt: str = ""
    working_messages: list[dict[str, Any]] = Field(default_factory=list)
    memories: list[MemoryResponse] = Field(default_factory=list)
    instructions: list[MemoryResponse] = Field(default_factory=list)


class ForgetResponse(BaseModel):
    deleted_per_type: dict[str, int] = Field(default_factory=dict)
    completed_at: datetime
    confirmation_id: str


class StatsResponse(BaseModel):
    total_memories: int = 0
    semantic: int = 0
    episodic: int = 0
    procedural: int = 0
    working: int = 0
    unique_users: int = 0
    unique_agents: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store: bool = False
    graph_store: bool = False
    cache: bool = False
    relational: bool = False
