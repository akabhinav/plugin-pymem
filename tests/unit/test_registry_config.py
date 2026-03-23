"""Unit tests for plugin registry and configuration."""

from __future__ import annotations

import pytest

from pymem.storage.registry import (
    get_embedding_provider,
    get_graph_store,
    get_vector_store,
    list_embedding_providers,
    list_graph_stores,
    list_vector_stores,
)


# Ensure plugins are registered
import pymem.embeddings.local_embeddings  # noqa: F401
import pymem.storage.vector.in_memory_vector  # noqa: F401
import pymem.storage.graph.in_memory_graph  # noqa: F401


class TestPluginRegistry:
    def test_list_vector_stores_includes_memory(self):
        stores = list_vector_stores()
        assert "memory" in stores

    def test_list_graph_stores_includes_memory(self):
        stores = list_graph_stores()
        assert "memory" in stores

    def test_list_embedding_providers_includes_local(self):
        providers = list_embedding_providers()
        assert "local" in providers

    def test_get_vector_store_memory(self):
        store = get_vector_store("memory")
        assert store is not None

    def test_get_graph_store_memory(self):
        store = get_graph_store("memory")
        assert store is not None

    def test_get_embedding_provider_local(self):
        provider = get_embedding_provider("local")
        assert provider is not None

    def test_get_unknown_vector_store_raises(self):
        with pytest.raises(ValueError, match="Unknown vector store"):
            get_vector_store("nonexistent_store")

    def test_get_unknown_graph_store_raises(self):
        with pytest.raises(ValueError, match="Unknown graph store"):
            get_graph_store("nonexistent_store")

    def test_get_unknown_embedding_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider("nonexistent_provider")


class TestSettings:
    def test_settings_loads(self):
        from pymem.config.settings import get_settings

        settings = get_settings()
        assert settings.PYMEM_VERSION == "1.0.0"
        assert settings.EMBEDDING_DIMENSION == 384
        assert settings.DEFAULT_SEARCH_LIMIT == 10

    def test_settings_defaults(self):
        from pymem.config.settings import get_settings

        settings = get_settings()
        assert settings.WORKING_MEMORY_TTL_SECONDS == 3600
        assert settings.MAX_SEARCH_LIMIT == 100
        assert settings.MIN_MEMORY_SCORE == 0.3

    def test_settings_embedding_config(self):
        from pymem.config.settings import get_settings

        settings = get_settings()
        assert settings.EMBEDDING_PROVIDER in ("sentence-transformers", "local")
        assert settings.EMBEDDING_BATCH_SIZE > 0


class TestConstants:
    def test_score_thresholds(self):
        from pymem.config.constants import (
            SCORE_MIN_KEEP,
            SCORE_MIN_ACTIVE,
            SCORE_MIN_SEARCH,
        )

        assert SCORE_MIN_KEEP > SCORE_MIN_ACTIVE
        assert SCORE_MIN_ACTIVE > SCORE_MIN_SEARCH

    def test_context_budgets(self):
        from pymem.config.constants import (
            CONTEXT_MAX_TOKENS_DEFAULT,
            CONTEXT_WORKING_BUDGET,
            CONTEXT_SEMANTIC_BUDGET,
            CONTEXT_EPISODIC_BUDGET,
            CONTEXT_PROCEDURAL_BUDGET,
        )

        sum_parts = (
            CONTEXT_WORKING_BUDGET
            + CONTEXT_SEMANTIC_BUDGET
            + CONTEXT_EPISODIC_BUDGET
            + CONTEXT_PROCEDURAL_BUDGET
        )
        assert sum_parts == CONTEXT_MAX_TOKENS_DEFAULT
