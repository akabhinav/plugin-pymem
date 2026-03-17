"""OpenTelemetry tracing for memory operations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

try:
    from opentelemetry import trace

    tracer = trace.get_tracer("pymem")

    @contextmanager
    def trace_operation(
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """Create a traced span for a memory operation."""
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v))
            yield span

except ImportError:
    @contextmanager
    def trace_operation(
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[None, None, None]:
        yield None
