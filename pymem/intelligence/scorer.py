"""Memory importance scorer — assigns scores to memories via LLM or heuristics."""

from __future__ import annotations

from pymem.intelligence.pygate_client import PyGateClient
from pymem.models import MemoryRecord

SCORING_PROMPT = """You are a memory importance scorer.

Given a memory, rate its importance on a scale of 0.0 to 1.0:
- 0.9-1.0: Critical facts, key preferences, important events
- 0.7-0.8: Useful knowledge, notable events
- 0.5-0.6: Moderately useful context
- 0.3-0.4: Borderline useful
- 0.0-0.2: Trivial, redundant, or conversational filler

Return JSON: {"score": 0.0-1.0, "reason": "..."}"""


class MemoryScorer:
    """Assigns importance scores to memories."""

    def __init__(self, pygate: PyGateClient | None = None) -> None:
        self._pygate = pygate

    async def score(self, memory: MemoryRecord) -> float:
        """Score a memory using LLM if available, else heuristics."""
        if self._pygate:
            return await self._score_with_llm(memory)
        return self._score_heuristic(memory)

    async def _score_with_llm(self, memory: MemoryRecord) -> float:
        assert self._pygate
        result = await self._pygate.complete_json(
            system=SCORING_PROMPT,
            user=f"Memory: {memory.content}",
        )
        return float(result.get("score", 0.5))

    @staticmethod
    def _score_heuristic(memory: MemoryRecord) -> float:
        """Simple heuristic scoring based on content characteristics."""
        content = memory.content.lower()
        score = 0.5

        # Boost for preference/fact indicators
        preference_words = ["prefer", "like", "always", "never", "important", "hate", "love"]
        if any(w in content for w in preference_words):
            score += 0.15

        # Boost for longer, more substantive content
        word_count = len(content.split())
        if word_count > 15:
            score += 0.1
        elif word_count < 5:
            score -= 0.1

        # Boost for entity presence
        if memory.metadata.get("entities"):
            score += 0.1

        return min(1.0, max(0.0, score))
