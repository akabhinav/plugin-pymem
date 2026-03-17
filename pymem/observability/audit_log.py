"""Audit logging — every memory write/read/delete logged."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger("pymem.audit")


class AuditLogger:
    """Logs all memory operations for compliance and debugging."""

    def __init__(self, relational_store: Any = None) -> None:
        self._store = relational_store

    async def log(
        self,
        operation: str,
        memory_id: str,
        actor_id: str,
        scope: dict[str, str],
        before_state: str | None = None,
        after_state: str | None = None,
    ) -> None:
        """Log an audit entry."""
        logger.info(
            "memory_operation",
            operation=operation,
            memory_id=memory_id,
            actor_id=actor_id,
            scope=scope,
        )

        if self._store:
            try:
                await self._store.log_audit(
                    audit_id=str(uuid4()),
                    memory_id=memory_id,
                    operation=operation,
                    actor_id=actor_id,
                    scope=scope,
                    before_state=before_state,
                    after_state=after_state,
                )
            except Exception:
                logging.exception("Failed to write audit log to store")
