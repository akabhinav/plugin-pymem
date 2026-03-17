"""Decay engine — implements forgetting curve for memory score decay."""

from __future__ import annotations

import math

from pymem.config.constants import (
    DECAY_LAMBDA_EPISODIC,
    DECAY_LAMBDA_PROCEDURAL,
    DECAY_LAMBDA_SEMANTIC,
    DECAY_LAMBDA_WORKING,
)
from pymem.models import MemoryType


class DecayEngine:
    """Implements forgetting curve: score_t = score_0 * exp(-lambda * days_since_access).

    Lambda values per type:
    - Semantic: 0.01 (~100 days)
    - Episodic: 0.005 (~200 days)
    - Working: None (TTL-based)
    - Procedural: None (never decays)
    """

    LAMBDAS: dict[MemoryType, float | None] = {
        MemoryType.SEMANTIC: DECAY_LAMBDA_SEMANTIC,
        MemoryType.EPISODIC: DECAY_LAMBDA_EPISODIC,
        MemoryType.WORKING: DECAY_LAMBDA_WORKING,
        MemoryType.PROCEDURAL: DECAY_LAMBDA_PROCEDURAL,
    }

    def calculate_score(
        self,
        original_score: float,
        days_since_access: float,
        memory_type: MemoryType,
    ) -> float:
        """Calculate decayed score using forgetting curve."""
        lam = self.LAMBDAS.get(memory_type)
        if lam is None:
            return original_score
        return original_score * math.exp(-lam * days_since_access)

    def should_soft_delete(self, decayed_score: float) -> bool:
        """Memory should be soft-deleted when score drops below threshold."""
        return decayed_score < 0.1

    def should_downrank(self, decayed_score: float) -> bool:
        """Memory should be deprioritised in search results."""
        return decayed_score < 0.3
