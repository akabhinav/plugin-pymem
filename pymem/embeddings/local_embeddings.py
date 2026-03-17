"""Local embedding provider using sentence-transformers or a simple fallback."""

from __future__ import annotations

import asyncio
import hashlib
import struct
from typing import Any

from pymem.storage.base import EmbeddingPlugin
from pymem.storage.registry import register_embedding_provider


@register_embedding_provider("sentence-transformers")
class SentenceTransformerEmbedder(EmbeddingPlugin):
    """Local, no API cost, no PyGate needed.
    Default model: all-MiniLM-L6-v2 (384 dimensions).
    """

    dimension: int = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs: Any) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, text: str) -> list[float]:
        model = self._load_model()
        return await asyncio.to_thread(
            lambda: model.encode(text, normalize_embeddings=True).tolist()
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        return await asyncio.to_thread(
            lambda: model.encode(
                texts, normalize_embeddings=True, batch_size=32
            ).tolist()
        )


@register_embedding_provider("local")
class LocalEmbeddingProvider(EmbeddingPlugin):
    """Simple hash-based embedding for development. No ML dependencies needed.

    Produces deterministic 384-dimensional pseudo-embeddings from text.
    NOT suitable for production semantic search — use sentence-transformers instead.
    """

    dimension: int = 384

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def embed(self, text: str) -> list[float]:
        return self._hash_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic pseudo-embedding from text hash.

        Uses hash bytes mapped to [-1, 1] range to produce valid float vectors.
        """
        result: list[float] = []
        chunk = 0
        while len(result) < self.dimension:
            h = hashlib.sha256(f"{text}:{chunk}".encode()).digest()
            for byte in h:
                if len(result) >= self.dimension:
                    break
                # Map byte [0, 255] to [-1.0, 1.0]
                result.append((byte / 127.5) - 1.0)
            chunk += 1
        result = result[:self.dimension]
        # Normalize to unit vector
        magnitude = sum(v * v for v in result) ** 0.5
        if magnitude > 0:
            result = [v / magnitude for v in result]
        return result
