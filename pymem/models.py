"""Core domain models — the heart of PyMem's type system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class MemoryType(StrEnum):
    """The four cognitive memory types — never collapse into one."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class SourceType(StrEnum):
    CONVERSATION = "conversation"
    TASK = "task"
    MANUAL = "manual"
    IMPORT = "import"


@dataclass
class MemoryScope:
    """Every memory is scoped to at least one dimension. Cross-scope leakage is a hard error."""

    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    org_id: str | None = None

    def __post_init__(self) -> None:
        if not any([self.user_id, self.agent_id, self.session_id, self.org_id]):
            raise ValueError("MemoryScope requires at least one of: user_id, agent_id, session_id, org_id")

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def to_filter(self) -> dict[str, str]:
        """Returns scope fields suitable for storage filters."""
        return self.to_dict()


@dataclass
class MemoryRecord:
    """Universal memory envelope — every memory type uses this."""

    id: str
    memory_type: MemoryType
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.5
    access_count: int = 0
    last_accessed_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    # Scope
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    org_id: str | None = None

    # Versioning
    version: int = 1
    parent_id: str | None = None

    # Soft delete
    is_deleted: bool = False
    deleted_at: datetime | None = None

    # Source
    source_type: str = "conversation"
    source_ref: str | None = None

    @property
    def scope(self) -> MemoryScope:
        return MemoryScope(
            user_id=self.user_id,
            agent_id=self.agent_id,
            session_id=self.session_id,
            org_id=self.org_id,
        )

    def scope_dict(self) -> dict[str, str]:
        d: dict[str, str] = {}
        if self.user_id:
            d["user_id"] = self.user_id
        if self.agent_id:
            d["agent_id"] = self.agent_id
        if self.session_id:
            d["session_id"] = self.session_id
        if self.org_id:
            d["org_id"] = self.org_id
        return d


@dataclass
class ConsolidationResult:
    """Result of memory deduplication/consolidation."""

    action: str  # keep_new | merge | discard | supersede
    memory: MemoryRecord | None = None
    merged_content: str | None = None
    superseded_id: str | None = None
    reason: str = ""

    @classmethod
    def from_llm_response(
        cls,
        response: dict[str, Any],
        new_memory: MemoryRecord,
        existing: list[MemoryRecord],
    ) -> ConsolidationResult:
        action = response.get("action", "keep_new")
        result = cls(action=action, memory=new_memory, reason=response.get("reason", ""))

        if action == "merge":
            result.merged_content = response.get("merged_content", new_memory.content)
            result.memory = MemoryRecord(
                id=new_memory.id,
                memory_type=new_memory.memory_type,
                content=result.merged_content or new_memory.content,
                score=max(new_memory.score, max((m.score for m in existing), default=0)),
                user_id=new_memory.user_id,
                agent_id=new_memory.agent_id,
                session_id=new_memory.session_id,
                org_id=new_memory.org_id,
                source_type=new_memory.source_type,
            )
        elif action == "supersede":
            result.superseded_id = response.get("superseded_id")

        return result


@dataclass
class AddMemoryResult:
    """Result of a memory add operation."""

    status: str  # stored | queued
    memories_count: int = 0
    extraction_id: str | None = None
    memories: list[MemoryRecord] = field(default_factory=list)


@dataclass
class SearchResult:
    """Result of a multi-type memory search."""

    memories: list[MemoryRecord] = field(default_factory=list)
    total: int = 0
    context_string: str | None = None


@dataclass
class ForgetResult:
    """Result of a GDPR forget operation."""

    deleted_per_type: dict[str, int] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)
    confirmation_id: str = ""


@dataclass
class AgentContext:
    """Assembled context for injection into an agent's system prompt."""

    working_messages: list[dict[str, Any]] = field(default_factory=list)
    relevant_facts: list[MemoryRecord] = field(default_factory=list)
    recent_events: list[MemoryRecord] = field(default_factory=list)
    agent_instructions: list[MemoryRecord] = field(default_factory=list)
    formatted: str = ""
