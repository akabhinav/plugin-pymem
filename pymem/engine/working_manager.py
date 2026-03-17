"""Working memory manager — session-scoped, Redis-backed sliding window."""

from __future__ import annotations

import json
from typing import Any

from pymem.models import MemoryScope


class WorkingMemoryManager:
    """Manages working memory — in-context sliding window per session.

    - Stored in Redis with session TTL
    - No LLM needed (fast, synchronous append)
    - Expires when session ends
    """

    def __init__(self, cache: Any, ttl: int = 3600) -> None:
        self._cache = cache
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"pymem:working:{session_id}"

    async def append(self, messages: list[dict[str, str]], scope: MemoryScope) -> None:
        """Append messages to the working memory window."""
        if not scope.session_id:
            return
        key = self._key(scope.session_id)
        for msg in messages:
            await self._cache.rpush(key, json.dumps(msg))
        await self._cache.expire(key, self._ttl)

    async def get_window(
        self,
        scope: MemoryScope,
        max_messages: int = 50,
    ) -> list[dict[str, str]]:
        """Get the current working memory window for a session."""
        if not scope.session_id:
            return []
        key = self._key(scope.session_id)
        raw_messages = await self._cache.lrange(key, -max_messages, -1)
        return [json.loads(m) for m in raw_messages]

    async def expire_session(self, session_id: str) -> None:
        """Clear working memory when session ends."""
        key = self._key(session_id)
        await self._cache.delete(key)

    async def get_length(self, session_id: str) -> int:
        """Get number of messages in working memory."""
        key = self._key(session_id)
        return await self._cache.llen(key)
