"""PyGate client — THE ONLY PLACE in PyMem that calls an LLM.

All extraction, consolidation, scoring, and summarisation routes through here.
NEVER import openai, anthropic, or any LLM SDK directly anywhere else.
"""

from __future__ import annotations

import json
import logging

import httpx

logger = logging.getLogger(__name__)


class PyGateClient:
    """Thin async HTTP client wrapping PyGate's OpenAI-compatible chat completions API."""

    def __init__(self, url: str, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        self.model = model

    async def complete(
        self,
        system: str,
        user: str,
        response_format: dict | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Send a chat completion request through PyGate."""
        body: dict = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format:
            body["response_format"] = response_format

        resp = await self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def complete_json(self, system: str, user: str) -> dict:
        """Send a chat completion and parse the response as JSON."""
        raw = await self.complete(
            system,
            user,
            response_format={"type": "json_object"},
        )
        # Strip markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
        return json.loads(clean)

    async def close(self) -> None:
        await self._client.aclose()
