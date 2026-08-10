"""Concept/literal vocabulary, sentence embeddings, and the ConnectionEndpoint.

`vocab.sqlite3` maps surface text to stable integer ids (and back); concepts
(subjects/predicates) and literals (relations, tenses, quantifiers, modifiers)
are deduplicated by case-folded, whitespace-normalised text. `embed_texts`
returns sentence-transformer vectors used by the pipeline's embed step.
`ConnectionEndpoint` is the value type for one side of an extracted relation.
"""
import sqlite3
import threading
from functools import lru_cache

from config import DEFAULT_MAP_PATH, EMBEDDING_MODEL_NAME, ENDPOINT_CONTEXT_LIMIT

_DB_CONNECTIONS = {}
_WRITE_LOCK = threading.RLock()
_EMBEDDING_MODEL = None


# --------------------------------------------------------------------------- #
# Text -> id store
# --------------------------------------------------------------------------- #
def normalize_text(text):
    return " ".join(str(text or "").strip().lower().split())


def _clean_text(text):
    return " ".join(str(text or "").strip().split())


def _connect(path=DEFAULT_MAP_PATH):
    db_path = str(path or DEFAULT_MAP_PATH)
    key = (threading.get_ident(), db_path)
    conn = _DB_CONNECTIONS.get(key)
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY,
                canonical TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS concept_aliases (
                normalized TEXT PRIMARY KEY,
                concept_id INTEGER NOT NULL REFERENCES concepts(id)
            );
            CREATE TABLE IF NOT EXISTS literals (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS literal_aliases (
                normalized TEXT PRIMARY KEY,
                literal_id INTEGER NOT NULL REFERENCES literals(id)
            );
            """
        )
        _DB_CONNECTIONS[key] = conn
    return conn


def _clear_read_caches():
    concept_from_index.cache_clear()
    literal_from_index.cache_clear()


def _get_or_create(table, alias_table, id_col, text_col, text, path):
    cleaned = _clean_text(text)
    normalized = normalize_text(cleaned)
    if not normalized:
        return -1
    conn = _connect(path)
    row = conn.execute(
        f"SELECT {id_col} FROM {alias_table} WHERE normalized=?", (normalized,)
    ).fetchone()
    if row is not None:
        return int(row[0])
    with _WRITE_LOCK:
        row = conn.execute(
            f"SELECT {id_col} FROM {alias_table} WHERE normalized=?", (normalized,)
        ).fetchone()
        if row is not None:
            return int(row[0])
        with conn:
            cursor = conn.execute(
                f"INSERT INTO {table}({text_col}) VALUES(?)", (cleaned,)
            )
            new_id = int(cursor.lastrowid)
            conn.execute(
                f"INSERT OR IGNORE INTO {alias_table}(normalized, {id_col}) VALUES(?, ?)",
                (normalized, new_id),
            )
    _clear_read_caches()
    return new_id


def get_or_create_index(text, vect=None, path=DEFAULT_MAP_PATH):
    return _get_or_create(
        "concepts", "concept_aliases", "concept_id", "canonical", text, path
    )


get_index = get_or_create_index


def get_literal_index(text, vect=None, path=DEFAULT_MAP_PATH):
    return _get_or_create(
        "literals", "literal_aliases", "literal_id", "text", text, path
    )


@lru_cache(maxsize=65536)
def concept_from_index(index, path=DEFAULT_MAP_PATH):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return ""
    row = _connect(path).execute(
        "SELECT canonical FROM concepts WHERE id=?", (index,)
    ).fetchone()
    return str(row[0]) if row else ""


@lru_cache(maxsize=65536)
def literal_from_index(index, path=DEFAULT_MAP_PATH):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return ""
    row = _connect(path).execute(
        "SELECT text FROM literals WHERE id=?", (index,)
    ).fetchone()
    return str(row[0]) if row else ""


def precache_text_vectors(texts, normalize=True):
    """No-op; retained for the extractor's warmup call."""
    return {}


# --------------------------------------------------------------------------- #
# Embeddings (the pipeline's embed step)
# --------------------------------------------------------------------------- #
def _model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def embed_texts(texts):
    """Return one float32 numpy vector per input string (normalised)."""
    import numpy as np

    texts = [str(t or "") for t in texts]
    if not texts:
        return []
    vectors = _model().encode(texts, normalize_embeddings=True)
    return [np.asarray(v, dtype=np.float32) for v in vectors]


# --------------------------------------------------------------------------- #
# Connection endpoint
# --------------------------------------------------------------------------- #
class ConnectionEndpoint:
    __slots__ = ("quantifier", "tense", "truth", "modifier_idx", "subject_predicate_id")
    _context_registry = {}

    def __init__(self, quantifier: int, tense: int, truth: int, ASU_idx, modifier_idx=None):
        self.quantifier = self._normalize_optional_id(quantifier, default=-1)
        self.tense = self._normalize_optional_id(tense, default=-1)
        self.truth = self._normalize_optional_id(truth, default=-1)
        self.subject_predicate_id = self._normalize_asu_id(ASU_idx)
        self.modifier_idx = self._normalize_modifier_ids(modifier_idx)

    @staticmethod
    def _normalize_id(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _normalize_optional_id(cls, value, default=-1):
        normalized = cls._normalize_id(value, default=default)
        return normalized if normalized >= -1 else default

    @classmethod
    def _normalize_asu_id(cls, value):
        if isinstance(value, int):
            return cls._normalize_optional_id(value, default=-1)
        text = " ".join(str(value or "").strip().split())
        if not text:
            return -1
        try:
            return int(get_index(text))
        except Exception:
            return -1

    @classmethod
    def _normalize_modifier_ids(cls, modifier_idx):
        if modifier_idx in (None, "", []):
            return ()
        if isinstance(modifier_idx, int):
            modifier_idx = [modifier_idx]
        values = []
        seen = set()
        for value in list(modifier_idx or []):
            normalized = cls._normalize_optional_id(value, default=-1)
            if normalized < 0 or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
        return tuple(values)

    @property
    def ASU_idx(self):
        return self.subject_predicate_id

    @ASU_idx.setter
    def ASU_idx(self, value):
        self.subject_predicate_id = self._normalize_asu_id(value)

    @classmethod
    def asu_from_idx(cls, idx):
        return concept_from_index(cls._normalize_optional_id(idx, default=-1))

    def asu_value(self):
        return self.asu_from_idx(self.subject_predicate_id)

    @classmethod
    def register_modifier(cls, modifier):
        if isinstance(modifier, int):
            return cls._normalize_optional_id(modifier, default=-1)
        text = " ".join(str(modifier or "").strip().split())
        return get_literal_index(text) if text else -1

    def modifier_value(self):
        return [literal_from_index(idx) for idx in self.modifier_idx if idx >= 0]

    @classmethod
    def register_context(cls, asu_idx, text="", source="unknown"):
        normalized_idx = cls._normalize_optional_id(asu_idx, default=-1)
        clean_text = " ".join(str(text or "").strip().split())
        clean_source = " ".join(str(source or "").strip().split()) or "unknown"
        if normalized_idx < 0 or not clean_text:
            return
        records = cls._context_registry.setdefault(normalized_idx, [])
        record = {"text": clean_text, "source": clean_source}
        if record in records:
            return
        records.append(record)
        if len(records) > ENDPOINT_CONTEXT_LIMIT:
            del records[:-ENDPOINT_CONTEXT_LIMIT]

    @classmethod
    def contexts_from_idx(cls, idx):
        normalized_idx = cls._normalize_optional_id(idx, default=-1)
        if normalized_idx < 0:
            return []
        return list(cls._context_registry.get(normalized_idx, ()))

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.quantifier,
                self.tense,
                self.truth,
                self.subject_predicate_id,
                list(self.modifier_idx),
            ),
        )

    def __str__(self):
        return (
            f"<ConnectionEndpoint: Q[{self.quantifier}] T[{self.tense}] "
            f"TR[{self.truth}] MOD[{list(self.modifier_idx)}] ID[{self.subject_predicate_id}]>"
        )
