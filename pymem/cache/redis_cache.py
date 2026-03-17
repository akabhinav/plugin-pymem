"""Cache layer — Redis for production, in-memory for dev/test."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class RedisCache:
    """Async Redis cache wrapper for working memory and session state."""

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._url = url
        self._client: Any = None

    async def initialize(self) -> None:
        import redis.asyncio as redis

        self._client = redis.from_url(self._url, decode_responses=True)

    async def rpush(self, key: str, value: str) -> None:
        await self._client.rpush(key, value)

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        return await self._client.lrange(key, start, stop)

    async def llen(self, key: str) -> int:
        return await self._client.llen(key)

    async def expire(self, key: str, seconds: int) -> None:
        await self._client.expire(key, seconds)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def get(self, key: str) -> str | None:
        return await self._client.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        await self._client.set(key, value, ex=ex)

    async def health_check(self) -> bool:
        try:
            return await self._client.ping()
        except Exception:
            return False

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()


class InMemoryCache:
    """Simple in-memory cache for development and testing. No Redis needed."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lists: dict[str, list[str]] = defaultdict(list)

    async def initialize(self) -> None:
        pass

    async def rpush(self, key: str, value: str) -> None:
        self._lists[key].append(value)

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        lst = self._lists.get(key, [])
        if stop == -1:
            return lst[start:]
        return lst[start : stop + 1]

    async def llen(self, key: str) -> int:
        return len(self._lists.get(key, []))

    async def expire(self, key: str, seconds: int) -> None:
        pass  # No TTL in memory cache

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._lists.pop(key, None)

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._data[key] = value

    async def health_check(self) -> bool:
        return True

    async def shutdown(self) -> None:
        self._data.clear()
        self._lists.clear()
