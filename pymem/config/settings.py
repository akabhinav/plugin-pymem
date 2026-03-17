"""All environment variables via pydantic-settings. Single source of truth."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """PyMem configuration — fully env-driven, type-safe."""

    # Platform
    PYMEM_MASTER_KEY: SecretStr = SecretStr("sk-pymem-dev")
    PYMEM_ENV: str = "dev"
    PYMEM_VERSION: str = "1.0.0"

    # Primary relational store
    DATABASE_URL: SecretStr = SecretStr(
        "postgresql+asyncpg://pymem:pymem@localhost:5432/pymem"
    )

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_WORKING_MEMORY_DB: int = 1
    CELERY_BROKER_URL: str = "redis://localhost:6379/2"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/3"

    # Vector store plugin selection
    VECTOR_STORE_BACKEND: str = "pgvector"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "pymem"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Graph store plugin selection
    GRAPH_STORE_BACKEND: str = "kuzu"
    KUZU_DB_PATH: str = "./data/kuzu"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: SecretStr | None = None
    MEMGRAPH_URI: str = "bolt://localhost:7688"

    # Embedding provider
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # PyGate integration
    PYGATE_URL: str = "http://localhost:4000"
    PYGATE_API_KEY: SecretStr = SecretStr("sk-pygate-dev")
    EXTRACTION_MODEL: str = "gpt-4o-mini"
    CONSOLIDATION_MODEL: str = "gpt-4o-mini"
    SCORING_MODEL: str = "gpt-4o-mini"

    # Memory defaults
    WORKING_MEMORY_TTL_SECONDS: int = 3600
    WORKING_MEMORY_MAX_TOKENS: int = 8000
    EPISODIC_MEMORY_TTL_DAYS: int = 90
    SEMANTIC_MEMORY_DECAY_DAYS: int = 365
    DEFAULT_SEARCH_LIMIT: int = 10
    MAX_SEARCH_LIMIT: int = 100

    # Extraction pipeline
    EXTRACTION_ASYNC: bool = True
    EXTRACTION_BATCH_SIZE: int = 10
    MIN_MEMORY_SCORE: float = 0.3

    # Multi-tenancy
    DEFAULT_ORG_MEMORY_LIMIT: int = 100_000
    DEFAULT_USER_MEMORY_LIMIT: int = 10_000

    # Observability
    LOG_LEVEL: str = "INFO"
    PROMETHEUS_ENABLED: bool = True
    OTEL_ENDPOINT: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
