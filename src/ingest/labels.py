# Shared text cleanup for raw language boundaries.
from functools import lru_cache
import re


_SPACE_RE = re.compile(r"\s+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_CONTENT_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_QUERY_PREFIX_RE = re.compile(r"^(?:query|search)\s*:\s*", re.IGNORECASE)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")
_DISALLOWED_NATURAL_RE = re.compile(r"[{}[\]<>\\/@#|=_+*^~`]")
_LONG_HEX_RE = re.compile(r"\b[a-f0-9]{16,}\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TRIM_CHARS = " \t\r\n\"'.,:;?!"
DEFAULT_CONTENT_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "to", "for", "in", "on", "with", "and", "or", "is", "are", "be", "can",
})


# Return the canonical lowercase label used for durable comparisons.
@lru_cache(maxsize=50000)
def normalize_label(value: str) -> str:
    text = _SPACE_RE.sub(" ", str(value).strip().lower())
    return text.strip(_TRIM_CHARS)


# Collapse arbitrary whitespace runs without changing case or punctuation.
def collapse_whitespace(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


# Make imported labels closer to readable text before field conversion.
def readable_label(value: object) -> str:
    text = str(value or "").strip().replace("_", " ")
    text = _CAMEL_BOUNDARY_RE.sub(" ", text)
    return collapse_whitespace(text)


# Normalize and validate a natural-language phrase for memory ingestion.
def clean_label_phrase(value: object, *, max_length: int) -> str:
    text = readable_label(value)
    if not text:
        return ""
    text = normalize_label(text)
    if not text or len(text) > max_length:
        return ""

    if looks_like_non_natural_label(text):
        return ""
    return text


# Detect labels that look like URLs, code tokens, hashes, or markup.
def looks_like_non_natural_label(text: str) -> bool:
    value = str(text or "").strip()
    if not value or value.startswith(("http://", "https://", "www.", "/")):
        return True
    if not any(character.isalpha() for character in value):
        return True
    if _DISALLOWED_NATURAL_RE.search(value):
        return True
    if _LONG_HEX_RE.search(value):
        return True
    for character in value:
        if character.isalnum() or character.isspace() or character in "'-":
            continue
        return True
    tokens = value.split()
    return any(len(token) > 36 for token in tokens)


# Check whether a short phrase is suitable for behavior sampling.
def natural_label_phrase(value: object, *, max_length: int = 48, max_words: int = 5) -> bool:
    text = collapse_whitespace(value)
    if not text or len(text) > max_length:
        return False
    if _DISALLOWED_NATURAL_RE.search(text):
        return False
    words = text.split()
    return bool(words) and len(words) <= max_words and any(character.isalpha() for character in text)


# Extract answer-matching content words while dropping connector terms.
def content_label_words(value: object, *, stop_words: frozenset[str] = DEFAULT_CONTENT_STOP_WORDS) -> tuple[str, ...]:
    words = (match.group(0).lower() for match in _CONTENT_WORD_RE.finditer(str(value or "")))
    return tuple(word for word in words if word and word not in stop_words)


# Clean one search query without interpreting its meaning.
def clean_search_query(value: object, *, max_length: int = 180) -> str:
    query = _LIST_PREFIX_RE.sub("", collapse_whitespace(value)).strip("\"'")
    query = _QUERY_PREFIX_RE.sub("", query)
    return query[:max_length].strip()


# Trim a URL-like boundary value without applying semantic cleanup.
def clean_url(value: object) -> str:
    return str(value or "").strip()


def _word_boundary_truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    last_space = cut.rfind(" ")
    return cut[:last_space] if last_space > 0 else cut


# Split scraped text into bounded sentence chunks for extraction.
def sentence_chunks(text: object, *, limit: int, min_length: int = 8, max_chars: int = 1200) -> tuple[str, ...]:
    normalized = collapse_whitespace(text)
    if not normalized:
        return ()
    chunks: list[str] = []
    for piece in _SENTENCE_SPLIT_RE.split(normalized):
        sentence = piece.strip()
        if len(sentence) < min_length:
            continue
        chunks.append(_word_boundary_truncate(sentence, max_chars))
        if len(chunks) >= limit:
            break
    if not chunks and normalized:
        chunks.append(_word_boundary_truncate(normalized, max_chars))
    return tuple(chunks)
