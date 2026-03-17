"""Memory Engine — THE central orchestrator for all memory operations.

Delegates to the four type managers. Never touches storage backends directly.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from pymem.engine.context_assembler import ContextAssembler
from pymem.engine.decay_engine import DecayEngine
from pymem.engine.episodic_manager import EpisodicMemoryManager
from pymem.engine.procedural_manager import ProceduralMemoryManager
from pymem.engine.semantic_manager import SemanticMemoryManager
from pymem.engine.working_manager import WorkingMemoryManager
from pymem.intelligence.consolidator import MemoryConsolidator
from pymem.intelligence.extractor import MemoryExtractor
from pymem.intelligence.graph_builder import GraphBuilder
from pymem.models import (
    AddMemoryResult,
    AgentContext,
    ForgetResult,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    SearchResult,
)
from pymem.storage.base import EmbeddingPlugin, GraphStorePlugin, VectorStorePlugin

logger = logging.getLogger(__name__)


class MemoryEngine:
    """THE entry point for all memory operations.

    Composes the four type managers, intelligence layer, and storage plugins.
    Every external operation goes through this class.
    """

    def __init__(
        self,
        vector_store: VectorStorePlugin,
        graph_store: GraphStorePlugin,
        relational_store: Any,
        embedding_provider: EmbeddingPlugin,
        extractor: MemoryExtractor | None = None,
        consolidator: MemoryConsolidator | None = None,
        graph_builder: GraphBuilder | None = None,
        cache: Any = None,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._relational = relational_store
        self._embedder = embedding_provider
        self._extractor = extractor
        self._consolidator = consolidator
        self._graph_builder = graph_builder
        self._decay = DecayEngine()

        # Type managers
        self.working = WorkingMemoryManager(cache) if cache else None
        self.episodic = EpisodicMemoryManager(graph_store, relational_store, embedding_provider)
        self.semantic = SemanticMemoryManager(
            vector_store, graph_store, relational_store, embedding_provider
        )
        self.procedural = ProceduralMemoryManager(relational_store, embedding_provider)

        # Context assembler
        self._context_assembler: ContextAssembler | None = None
        if self.working:
            self._context_assembler = ContextAssembler(
                working=self.working,
                semantic=self.semantic,
                episodic=self.episodic,
                procedural=self.procedural,
                embedding_provider=embedding_provider,
            )

    async def add(
        self,
        messages: list[dict[str, str]],
        scope: MemoryScope,
        memory_types: list[MemoryType] | None = None,
    ) -> AddMemoryResult:
        """Primary write path. Extract and store memories from a conversation.

        Steps:
        1. Append to working memory (fast, no LLM)
        2. Run extraction pipeline (LLM via PyGate)
        3. Compute embeddings, consolidate, store
        """
        # Step 1: Always update working memory
        if self.working and scope.session_id:
            await self.working.append(messages, scope)

        # Step 2: Run extraction
        return await self._run_extraction_pipeline(messages, scope, memory_types)

    async def _run_extraction_pipeline(
        self,
        messages: list[dict[str, str]],
        scope: MemoryScope,
        memory_types: list[MemoryType] | None = None,
    ) -> AddMemoryResult:
        """Full extraction pipeline — extract, embed, consolidate, store."""
        # Extract memories
        if self._extractor:
            try:
                extracted = await self._extractor.extract(messages, scope)
            except Exception:
                logger.warning("LLM extraction failed, using simple extraction")
                extracted = await self._extractor.extract_simple(messages, scope)
        else:
            # No extractor — create simple memories from user messages
            extracted = self._simple_extract(messages, scope)

        if not extracted:
            return AddMemoryResult(status="stored", memories_count=0)

        # Filter by requested types
        if memory_types:
            extracted = [m for m in extracted if m.memory_type in memory_types]

        # Process each extracted memory
        stored: list[MemoryRecord] = []
        for memory in extracted:
            try:
                result = await self._process_single_memory(memory, scope)
                if result:
                    stored.append(result)
            except Exception:
                logger.exception("Failed to process memory: %s", memory.id)

        # Build graph from new semantic memories
        semantic_memories = [m for m in stored if m.memory_type == MemoryType.SEMANTIC]
        if semantic_memories and self._graph_builder:
            try:
                nodes, edges = await self._graph_builder.build_from_memories(
                    semantic_memories, scope.to_dict()
                )
                for node in nodes:
                    await self._graph.upsert_node(node)
                for edge in edges:
                    await self._graph.upsert_edge(edge)
            except Exception:
                logger.warning("Graph building failed, continuing without graph")

        # Update user stats
        if scope.user_id:
            for memory in stored:
                try:
                    await self._relational.update_user_stats(
                        scope.user_id, memory.memory_type.value
                    )
                except Exception:
                    pass

        return AddMemoryResult(
            status="stored",
            memories_count=len(stored),
            memories=stored,
        )

    async def _process_single_memory(
        self,
        memory: MemoryRecord,
        scope: MemoryScope,
    ) -> MemoryRecord | None:
        """Process a single extracted memory: embed, consolidate, store."""
        if memory.memory_type == MemoryType.SEMANTIC:
            # Consolidate with existing similar memories
            if self._consolidator:
                similar = await self.semantic.get_similar(
                    memory.content, scope, limit=5
                )
                if similar:
                    try:
                        result = await self._consolidator.consolidate(memory, similar)
                    except Exception:
                        result = await self._consolidator.consolidate_simple(
                            memory, similar
                        )

                    if result.action == "discard":
                        return None
                    if result.action == "merge" and result.memory:
                        memory = result.memory
                    elif result.action == "supersede" and result.superseded_id:
                        await self._relational.soft_delete_memory(result.superseded_id)

            return await self.semantic.add(
                content=memory.content,
                scope=scope,
                score=memory.score,
                metadata=memory.metadata,
                memory_id=memory.id,
            )

        elif memory.memory_type == MemoryType.EPISODIC:
            return await self.episodic.add_event(
                content=memory.content,
                scope=scope,
                metadata=memory.metadata,
            )

        elif memory.memory_type == MemoryType.PROCEDURAL:
            tags = memory.metadata.get("tags", [])
            return await self.procedural.add(
                content=memory.content,
                scope=scope,
                tags=tags,
            )

        return None

    async def search(
        self,
        query: str,
        scope: MemoryScope,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
    ) -> SearchResult:
        """Multi-type parallel search. Returns merged, ranked results."""
        query_embedding = await self._embedder.embed(query)
        types = memory_types or [
            MemoryType.SEMANTIC,
            MemoryType.EPISODIC,
            MemoryType.PROCEDURAL,
        ]

        tasks = []
        if MemoryType.SEMANTIC in types:
            tasks.append(self.semantic.search(query_embedding, scope, limit))
        else:
            tasks.append(self._empty_list())

        if MemoryType.EPISODIC in types:
            tasks.append(self.episodic.search(query_embedding, scope, limit))
        else:
            tasks.append(self._empty_list())

        if MemoryType.PROCEDURAL in types:
            tasks.append(self.procedural.search(query, scope, limit))
        else:
            tasks.append(self._empty_list())

        semantic_results, episodic_results, procedural_results = await asyncio.gather(
            *tasks
        )

        # Merge and sort by score
        all_results = semantic_results + episodic_results + procedural_results
        all_results.sort(key=lambda m: m.score, reverse=True)
        merged = all_results[:limit]

        # Update access metadata
        for memory in merged:
            try:
                await self._relational.update_access(memory.id)
            except Exception:
                pass

        return SearchResult(
            memories=merged,
            total=len(merged),
        )

    async def get_context(
        self,
        scope: MemoryScope,
        query: str | None = None,
        max_tokens: int = 4000,
    ) -> AgentContext:
        """Assemble complete context for injection into an agent's prompt."""
        if self._context_assembler:
            return await self._context_assembler.assemble(scope, query, max_tokens)

        # Fallback without working memory
        query_embedding = None
        if query:
            query_embedding = await self._embedder.embed(query)

        semantic, episodic, procedural = await asyncio.gather(
            self.semantic.search(query_embedding, scope, limit=8),
            self.episodic.get_recent(scope, limit=5),
            self.procedural.get_all(scope),
        )

        formatted = ContextAssembler._format_prompt_block(
            [], semantic, episodic, procedural
        )
        return AgentContext(
            relevant_facts=semantic,
            recent_events=episodic,
            agent_instructions=procedural,
            formatted=formatted,
        )

    async def get(self, memory_id: str) -> MemoryRecord | None:
        """Get a specific memory by ID."""
        row = await self._relational.get_memory(memory_id)
        if not row:
            return None
        return MemoryRecord(
            id=row["id"],
            memory_type=MemoryType(row["memory_type"]),
            content=row["content"],
            score=row.get("score", 0.5),
            access_count=row.get("access_count", 0),
            user_id=row.get("user_id"),
            agent_id=row.get("agent_id"),
            session_id=row.get("session_id"),
            org_id=row.get("org_id"),
            version=row.get("version", 1),
            is_deleted=bool(row.get("is_deleted", False)),
        )

    async def update(
        self,
        memory_id: str,
        content: str,
        scope: MemoryScope,
    ) -> MemoryRecord | None:
        """Update memory content. Creates a new version."""
        existing = await self._relational.get_memory(memory_id)
        if not existing:
            return None

        new_id = str(uuid4())
        now = datetime.utcnow()

        # Create new version
        new_record = {
            "id": new_id,
            "memory_type": existing["memory_type"],
            "content": content,
            "score": existing.get("score", 0.5),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "version": existing.get("version", 1) + 1,
            "parent_id": memory_id,
            "user_id": existing.get("user_id"),
            "agent_id": existing.get("agent_id"),
            "session_id": existing.get("session_id"),
            "org_id": existing.get("org_id"),
        }
        await self._relational.save_memory(new_record)

        # Update vector store if semantic
        if existing["memory_type"] == "semantic":
            embedding = await self._embedder.embed(content)
            scope_dict = scope.to_dict()
            await self._vector.upsert(
                memory_id=new_id,
                embedding=embedding,
                content=content,
                metadata=scope_dict,
            )
            # Remove old vector
            await self._vector.delete(memory_id)

        # Soft-delete old version
        await self._relational.soft_delete_memory(memory_id)

        return MemoryRecord(
            id=new_id,
            memory_type=MemoryType(existing["memory_type"]),
            content=content,
            version=new_record["version"],
            parent_id=memory_id,
        )

    async def delete(
        self,
        memory_id: str,
        scope: MemoryScope,
        hard_delete: bool = False,
    ) -> None:
        """Delete a memory. Soft delete by default."""
        if hard_delete:
            await self._relational.hard_delete_memory(memory_id)
            await self._vector.delete(memory_id)
            await self._graph.delete_node(memory_id)
        else:
            await self._relational.soft_delete_memory(memory_id)

    async def forget(
        self,
        scope: MemoryScope,
        memory_types: list[MemoryType] | None = None,
    ) -> ForgetResult:
        """GDPR-compliant forget. Removes ALL data for this scope."""
        types = memory_types or list(MemoryType)
        deleted_per_type: dict[str, int] = {}

        for mtype in types:
            count = 0
            if mtype == MemoryType.SEMANTIC:
                count = await self.semantic.delete_by_scope(scope)
            elif mtype == MemoryType.EPISODIC:
                count = await self.episodic.delete_by_scope(scope)
            elif mtype == MemoryType.PROCEDURAL:
                count = await self.procedural.delete_by_scope(scope)
            elif mtype == MemoryType.WORKING and self.working and scope.session_id:
                await self.working.expire_session(scope.session_id)
                count = 1
            deleted_per_type[mtype.value] = count

        # Also clear graph
        await self._graph.delete_by_scope(scope.to_dict())

        # Log audit
        try:
            await self._relational.log_audit(
                audit_id=str(uuid4()),
                memory_id="*",
                operation="forget",
                actor_id=scope.user_id or scope.agent_id or "system",
                scope=scope.to_dict(),
            )
        except Exception:
            pass

        return ForgetResult(
            deleted_per_type=deleted_per_type,
            confirmation_id=str(uuid4()),
        )

    async def share(
        self,
        memory_id: str,
        from_scope: MemoryScope,
        to_scope: MemoryScope,
    ) -> MemoryRecord | None:
        """Clone a memory into a different scope."""
        existing = await self._relational.get_memory(memory_id)
        if not existing:
            return None

        new_id = str(uuid4())
        to_dict = to_scope.to_dict()
        new_record = {**existing, "id": new_id, **to_dict}
        await self._relational.save_memory(new_record)

        return MemoryRecord(
            id=new_id,
            memory_type=MemoryType(existing["memory_type"]),
            content=existing["content"],
            **to_dict,
        )

    @staticmethod
    def _simple_extract(
        messages: list[dict[str, str]],
        scope: MemoryScope,
    ) -> list[MemoryRecord]:
        """Simple extraction without LLM."""
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

    @staticmethod
    async def _empty_list() -> list[MemoryRecord]:
        return []
