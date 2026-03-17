"""Memory extractor — extracts structured memories from conversations via PyGate."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pymem.intelligence.pygate_client import PyGateClient
from pymem.models import MemoryRecord, MemoryScope, MemoryType

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction specialist for an AI agent platform.

Given a conversation, extract factual, useful memories that should be persisted across sessions.
Return a JSON object with key "memories" containing an array.

For each memory extract:
- content: The memory as a clear, self-contained statement (no pronouns)
- memory_type: one of "episodic" | "semantic" | "procedural"
- score: importance 0.0–1.0 (0.8+ = highly important, 0.3 = borderline useful)
- entities: list of named entities mentioned (people, tools, companies, concepts)
- temporal: true if this describes a specific event in time

Rules:
- Semantic: facts, preferences, beliefs ("User prefers Python over Java")
- Episodic: timestamped events ("User completed the payment service on March 14")
- Procedural: how-to knowledge, agent instructions ("Always run tests after editing")
- Score < 0.3: discard (trivial, redundant, or conversational filler)
- Make statements self-contained — include subject explicitly
- Deduplicate: do not extract the same fact twice in different phrasings
- Maximum 15 memories per extraction call"""


class MemoryExtractor:
    """Extracts structured memories from raw conversation messages.

    Pipeline:
    1. Receive list of messages (role + content)
    2. Call LLM via PyGate to extract facts
    3. Filter by min_score threshold
    4. Return typed MemoryRecord objects (not yet persisted)
    """

    def __init__(self, pygate: PyGateClient, min_score: float = 0.3) -> None:
        self._pygate = pygate
        self._min_score = min_score

    async def extract(
        self,
        messages: list[dict[str, str]],
        scope: MemoryScope,
        existing_summary: str | None = None,
    ) -> list[MemoryRecord]:
        """Extract memories from conversation messages."""
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages[-20:]
        )

        context = f"Existing context:\n{existing_summary}\n\n" if existing_summary else ""
        user_prompt = f"{context}Conversation to extract memories from:\n{conversation_text}"

        result = await self._pygate.complete_json(
            system=EXTRACTION_SYSTEM_PROMPT,
            user=user_prompt,
        )

        records: list[MemoryRecord] = []
        scope_dict = scope.to_dict()

        for item in result.get("memories", []):
            score = item.get("score", 0)
            if score < self._min_score:
                continue

            memory_type_str = item.get("memory_type", "semantic")
            try:
                memory_type = MemoryType(memory_type_str)
            except ValueError:
                memory_type = MemoryType.SEMANTIC

            records.append(
                MemoryRecord(
                    id=str(uuid4()),
                    memory_type=memory_type,
                    content=item["content"],
                    metadata={
                        "entities": item.get("entities", []),
                        "temporal": item.get("temporal", False),
                        "extracted_from": "conversation",
                    },
                    score=score,
                    access_count=0,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    source_type="conversation",
                    **scope_dict,
                )
            )

        return records

    async def extract_simple(
        self,
        messages: list[dict[str, str]],
        scope: MemoryScope,
    ) -> list[MemoryRecord]:
        """Simple extraction without LLM — creates one memory per user message.

        Used as fallback when PyGate is unavailable.
        """
        records: list[MemoryRecord] = []
        scope_dict = scope.to_dict()

        for msg in messages:
            if msg.get("role") == "user" and len(msg.get("content", "")) > 10:
                records.append(
                    MemoryRecord(
                        id=str(uuid4()),
                        memory_type=MemoryType.SEMANTIC,
                        content=msg["content"],
                        metadata={"extracted_from": "simple"},
                        score=0.5,
                        source_type="conversation",
                        **scope_dict,
                    )
                )

        return records
