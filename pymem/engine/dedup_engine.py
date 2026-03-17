"""Dedup engine — cross-memory deduplication using embedding similarity."""

from __future__ import annotations

from pymem.config.constants import CONSOLIDATION_SIMILARITY_THRESHOLD
from pymem.storage.base import EmbeddingPlugin


class DedupEngine:
    """Cross-memory deduplication using cosine similarity of embeddings."""

    def __init__(
        self,
        embedding_provider: EmbeddingPlugin,
        threshold: float = CONSOLIDATION_SIMILARITY_THRESHOLD,
    ) -> None:
        self._embedder = embedding_provider
        self._threshold = threshold

    async def find_duplicates(
        self,
        new_embedding: list[float],
        existing_embeddings: list[tuple[str, list[float]]],
    ) -> list[tuple[str, float]]:
        """Find existing memories that are near-duplicates of the new one.

        Returns list of (memory_id, similarity_score) above threshold.
        """
        duplicates: list[tuple[str, float]] = []
        for memory_id, embedding in existing_embeddings:
            sim = self._cosine_similarity(new_embedding, embedding)
            if sim >= self._threshold:
                duplicates.append((memory_id, sim))
        return sorted(duplicates, key=lambda x: x[1], reverse=True)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
