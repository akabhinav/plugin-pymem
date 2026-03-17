"""Defaults, limits, TTLs — non-configurable constants."""

# Memory type identifiers
MEMORY_TYPE_WORKING = "working"
MEMORY_TYPE_EPISODIC = "episodic"
MEMORY_TYPE_SEMANTIC = "semantic"
MEMORY_TYPE_PROCEDURAL = "procedural"

# Decay constants (lambda values for forgetting curve)
DECAY_LAMBDA_SEMANTIC = 0.01  # ~100 days half-life
DECAY_LAMBDA_EPISODIC = 0.005  # ~200 days half-life
DECAY_LAMBDA_WORKING = None  # TTL-based, not curve
DECAY_LAMBDA_PROCEDURAL = None  # Never decays

# Score thresholds
SCORE_MIN_KEEP = 0.3  # Below this: discard at extraction
SCORE_MIN_ACTIVE = 0.1  # Below this: soft-delete at decay
SCORE_MIN_SEARCH = 0.05  # Below this: exclude from search

# Consolidation
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.92  # Above this: merge/discard

# API limits
API_MAX_MESSAGES_PER_ADD = 100
API_MAX_BULK_IMPORT = 1000
API_MAX_SEARCH_LIMIT = 100
API_DEFAULT_SEARCH_LIMIT = 10

# Redis key prefixes
REDIS_PREFIX_WORKING = "pymem:working"
REDIS_PREFIX_SESSION = "pymem:session"
REDIS_PREFIX_CACHE = "pymem:cache"

# Source types
SOURCE_CONVERSATION = "conversation"
SOURCE_TASK = "task"
SOURCE_MANUAL = "manual"
SOURCE_IMPORT = "import"

# Procedural memory priority range
PRIORITY_HIGHEST = 1
PRIORITY_DEFAULT = 5
PRIORITY_LOWEST = 10

# Context assembly token budgets
CONTEXT_MAX_TOKENS_DEFAULT = 4000
CONTEXT_WORKING_BUDGET = 800
CONTEXT_SEMANTIC_BUDGET = 1500
CONTEXT_EPISODIC_BUDGET = 1000
CONTEXT_PROCEDURAL_BUDGET = 700
