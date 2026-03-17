"""Prometheus metrics for memory operations."""

from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram

    MEMORY_ADD_TOTAL = Counter(
        "pymem_add_total", "Memories added", ["type", "source"]
    )
    MEMORY_SEARCH_LATENCY = Histogram(
        "pymem_search_latency", "Search latency seconds", ["type"]
    )
    EXTRACTION_DURATION = Histogram(
        "pymem_extraction_seconds", "Extraction pipeline duration"
    )
    CONSOLIDATION_MERGES = Counter(
        "pymem_consolidation_merges", "Memories merged during consolidation"
    )
    DECAY_DELETIONS = Counter(
        "pymem_decay_deletions", "Memories expired by decay engine"
    )
    CACHE_HIT_RATE = Gauge(
        "pymem_cache_hit_rate", "Working memory cache hit rate"
    )
    VECTOR_STORE_COUNT = Gauge(
        "pymem_vector_count", "Total vectors indexed"
    )
    API_REQUEST_TOTAL = Counter(
        "pymem_api_requests_total", "Total API requests", ["method", "endpoint"]
    )

except ImportError:
    # Prometheus not available — use no-op metrics
    class _NoOp:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass

    MEMORY_ADD_TOTAL = _NoOp()
    MEMORY_SEARCH_LATENCY = _NoOp()
    EXTRACTION_DURATION = _NoOp()
    CONSOLIDATION_MERGES = _NoOp()
    DECAY_DELETIONS = _NoOp()
    CACHE_HIT_RATE = _NoOp()
    VECTOR_STORE_COUNT = _NoOp()
    API_REQUEST_TOTAL = _NoOp()
