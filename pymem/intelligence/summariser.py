"""Conversation summariser — creates episodic summaries via PyGate."""

from __future__ import annotations

from pymem.intelligence.pygate_client import PyGateClient

SUMMARISATION_PROMPT = """You are a conversation summariser for an AI agent memory system.

Given a conversation, produce a concise summary covering:
1. Key topics discussed
2. Decisions made or actions taken
3. Important facts mentioned
4. Unresolved questions or next steps

Return JSON: {"summary": "...", "key_topics": ["..."], "actions": ["..."]}"""


class ConversationSummariser:
    """Summarises conversations for episodic memory storage."""

    def __init__(self, pygate: PyGateClient) -> None:
        self._pygate = pygate

    async def summarise(self, messages: list[dict[str, str]]) -> dict:
        """Summarise a conversation via PyGate LLM."""
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages[-30:]
        )

        return await self._pygate.complete_json(
            system=SUMMARISATION_PROMPT,
            user=f"Conversation:\n{conversation_text}",
        )

    @staticmethod
    def summarise_simple(messages: list[dict[str, str]]) -> dict:
        """Simple summarisation without LLM — extracts key messages."""
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        combined = "; ".join(user_messages[-5:])
        return {
            "summary": combined[:500] if combined else "No conversation content",
            "key_topics": [],
            "actions": [],
        }
