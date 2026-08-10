"""Shared configuration for the extraction pipeline."""

# Concept/literal text store + embedding model.
DEFAULT_MAP_PATH = "vocab.sqlite3"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Per-endpoint context retained for each concept.
ENDPOINT_CONTEXT_LIMIT = 24

# Extraction limits.
EXTRACTION_SENTENCE_LIMIT = 96
EXTRACTION_CLAUSE_LIMIT = 180
MAX_CONNECTIONS_PER_QUERY = 512

# Closed-class function words. Used only to keep pronoun/function-word tokens
# out of subject/predicate slots and concept keys.
STOPWORD_TOKENS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "had",
    "has", "have", "i", "in", "is", "it", "its", "of", "on", "or", "our",
    "that", "the", "their", "them", "these", "they", "this", "those", "to",
    "was", "we", "were", "with", "you",
})
