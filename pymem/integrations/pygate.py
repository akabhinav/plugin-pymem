"""PyGate integration — LLM client wrapper for platform-wide use."""

from __future__ import annotations

from pymem.config.settings import get_settings
from pymem.intelligence.pygate_client import PyGateClient


def create_pygate_client(
    model: str | None = None,
) -> PyGateClient:
    """Create a PyGate client from settings."""
    settings = get_settings()
    return PyGateClient(
        url=settings.PYGATE_URL,
        api_key=settings.PYGATE_API_KEY.get_secret_value(),
        model=model or settings.EXTRACTION_MODEL,
    )
