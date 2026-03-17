"""Tests for embedding providers."""

import pytest

from pymem.embeddings.local_embeddings import LocalEmbeddingProvider


@pytest.fixture
def embedder():
    return LocalEmbeddingProvider()


@pytest.mark.asyncio
class TestLocalEmbeddings:
    async def test_embed_returns_correct_dimension(self, embedder):
        result = await embedder.embed("hello world")
        assert len(result) == 384

    async def test_embed_is_deterministic(self, embedder):
        r1 = await embedder.embed("same text")
        r2 = await embedder.embed("same text")
        assert r1 == r2

    async def test_embed_different_texts(self, embedder):
        r1 = await embedder.embed("hello")
        r2 = await embedder.embed("goodbye")
        assert r1 != r2

    async def test_embed_batch(self, embedder):
        results = await embedder.embed_batch(["hello", "world", "test"])
        assert len(results) == 3
        assert all(len(r) == 384 for r in results)

    async def test_embed_is_normalized(self, embedder):
        result = await embedder.embed("test")
        magnitude = sum(v * v for v in result) ** 0.5
        assert abs(magnitude - 1.0) < 0.01

    def test_get_dimension(self, embedder):
        assert embedder.get_dimension() == 384
