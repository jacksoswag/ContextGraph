import re
import sqlite3
import threading
from functools import lru_cache

import numpy as np

from constants import (
    DEFAULT_MAP_PATH,
    EMBEDDING_MODEL_NAME,
    MAX_MERGE_SIMILARITY,
    MIN_MERGE_SIMILARITY,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
EMBEDDING_MODEL = None
_DB_CONNECTIONS = {}


def normalize_text(text):
    return " ".join(str(text or "").strip().lower().split())


def _clean_text(text):
    return " ".join(str(text or "").strip().split())


def vector_to_numpy(vect):
    array = np.asarray(vect, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("vector cannot be empty")
    return array


def vector_to_list(vect):
    return [float(value) for value in vector_to_numpy(vect).tolist()]


def _normalize_vector(vect):
    array = vector_to_numpy(vect)
    norm = float(np.linalg.norm(array))
    if norm > 1e-12:
        array = array / norm
    return array


def _vector_blob(vect):
    if not vect:
        return None
    return vector_to_numpy(vect).astype(np.float32).tobytes()


def _vector_from_blob(blob):
    if not blob:
        return []
    return vector_to_list(np.frombuffer(blob, dtype=np.float32))


def _db_path(path=DEFAULT_MAP_PATH):
    return str(path or DEFAULT_MAP_PATH)


def _connect(path=DEFAULT_MAP_PATH):
    db_path = _db_path(path)
    cache_key = (threading.get_ident(), db_path)
    conn = _DB_CONNECTIONS.get(cache_key)
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        _DB_CONNECTIONS[cache_key] = conn
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS concepts (
            id INTEGER PRIMARY KEY,
            canonical TEXT NOT NULL,
            vector BLOB,
            specificity REAL NOT NULL DEFAULT 0,
            merge_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS concept_aliases (
            normalized TEXT PRIMARY KEY,
            concept_id INTEGER NOT NULL REFERENCES concepts(id)
        );
        CREATE TABLE IF NOT EXISTS concept_terms (
            term TEXT NOT NULL,
            concept_id INTEGER NOT NULL REFERENCES concepts(id),
            PRIMARY KEY(term, concept_id)
        );
        CREATE INDEX IF NOT EXISTS idx_concept_terms_concept
            ON concept_terms(concept_id);
        CREATE TABLE IF NOT EXISTS literals (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            vector BLOB,
            merge_count INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS literal_aliases (
            normalized TEXT PRIMARY KEY,
            literal_id INTEGER NOT NULL REFERENCES literals(id)
        );
        """
    )
    _ensure_concept_terms(conn)


def _concept_terms(text):
    return tuple(sorted({token.lower() for token in TOKEN_RE.findall(normalize_text(text))}))


def _store_concept_terms(conn, concept_id, *texts):
    terms = set()
    for text in texts:
        terms.update(_concept_terms(text))
    if not terms:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO concept_terms(term, concept_id) VALUES(?, ?)",
        [(term, int(concept_id)) for term in terms],
    )


def _ensure_concept_terms(conn):
    row = conn.execute("SELECT value FROM metadata WHERE key='concept_terms_version'").fetchone()
    if row and str(row[0]) == "1":
        return
    with conn:
        conn.execute("DELETE FROM concept_terms")
        for concept_id, canonical in conn.execute("SELECT id, canonical FROM concepts"):
            _store_concept_terms(conn, concept_id, canonical)
        for normalized, concept_id in conn.execute("SELECT normalized, concept_id FROM concept_aliases"):
            _store_concept_terms(conn, concept_id, normalized)
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('concept_terms_version', '1')"
        )


def _clear_read_caches():
    concept_from_index.cache_clear()
    concept_vector_from_index.cache_clear()
    literal_from_index.cache_clear()
    literal_vector_from_index.cache_clear()
    concept_vector_from_text.cache_clear()


def _context_without_component(component_text, clause_text):
    component = _clean_text(component_text)
    clause = _clean_text(clause_text)
    if not component or not clause:
        return clause
    trimmed = re.compile(re.escape(component), re.IGNORECASE).sub(" ", clause, count=1)
    trimmed = _clean_text(trimmed)
    return trimmed or clause


def contextual_component_vector(text, clause_text="", focus_weight=0.8, context_weight=0.2):
    component = _clean_text(text)
    clause = _clean_text(clause_text)
    if not component:
        return []
    component_vec = vector_to_numpy(_encode_text_vector(component, normalize=False))
    if not clause or normalize_text(component) == normalize_text(clause):
        return vector_to_list(_normalize_vector(component_vec))
    rest_clause = _context_without_component(component, clause)
    if not rest_clause:
        return vector_to_list(_normalize_vector(component_vec))
    context_vec = vector_to_numpy(_encode_text_vector(rest_clause, normalize=False))
    blended = (component_vec * float(focus_weight)) + (context_vec * float(context_weight))
    return vector_to_list(_normalize_vector(blended))


def contextual_component_vectors(items, focus_weight=0.8, context_weight=0.2):
    normalized_items = []
    encode_inputs = []
    for text, clause_text in list(items or []):
        component = _clean_text(text)
        clause = _clean_text(clause_text)
        if not component:
            normalized_items.append((component, clause, ""))
            continue
        rest_clause = ""
        if clause and normalize_text(component) != normalize_text(clause):
            rest_clause = _context_without_component(component, clause)
        normalized_items.append((component, clause, rest_clause))
        encode_inputs.append(component)
        if rest_clause:
            encode_inputs.append(rest_clause)

    encoded = _encode_text_vectors(encode_inputs, normalize=False)
    vectors = []
    for component, _clause, rest_clause in normalized_items:
        if not component:
            vectors.append([])
            continue
        component_vec = vector_to_numpy(encoded.get(component, []))
        if not rest_clause:
            vectors.append(vector_to_list(_normalize_vector(component_vec)))
            continue
        context_vec = vector_to_numpy(encoded.get(rest_clause, []))
        blended = (component_vec * float(focus_weight)) + (context_vec * float(context_weight))
        vectors.append(vector_to_list(_normalize_vector(blended)))
    return vectors


def vector_magnitude_signal(vect):
    magnitude = float(np.linalg.norm(vector_to_numpy(vect)))
    return float(max(0.0, min(1.0, magnitude / (magnitude + 1.0))))


def cosine_similarity(left, right):
    left_vec = vector_to_numpy(left)
    right_vec = vector_to_numpy(right)
    if left_vec.shape != right_vec.shape:
        raise ValueError("vector dimensions do not match")
    left_norm = float(np.linalg.norm(left_vec))
    right_norm = float(np.linalg.norm(right_vec))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


def text_similarity(left_text, right_text):
    left = _clean_text(left_text)
    right = _clean_text(right_text)
    if not left or not right:
        return 0.0
    if normalize_text(left) == normalize_text(right):
        return 1.0
    if right < left:
        left, right = right, left
    return _text_similarity_cached(left, right)


@lru_cache(maxsize=32768)
def _text_similarity_cached(left, right):
    return cosine_similarity(
        _encode_text_vector(left, normalize=True),
        _encode_text_vector(right, normalize=True),
    )


def vector_specificity(vect):
    vec = vector_to_numpy(vect)
    dimension = max(1, int(vec.size))
    magnitude = float(np.linalg.norm(vec))
    nonzero_ratio = float(np.count_nonzero(np.abs(vec) > 1e-8)) / float(dimension)
    magnitude_signal = magnitude / (magnitude + 1.0)
    sparsity_signal = 1.0 - nonzero_ratio
    specificity = (0.8 * magnitude_signal) + (0.2 * sparsity_signal)
    return float(max(0.0, min(1.0, specificity)))


def merge_similarity_threshold(left_vect, right_vect):
    specificity = max(vector_specificity(left_vect), vector_specificity(right_vect))
    return MIN_MERGE_SIMILARITY + ((MAX_MERGE_SIMILARITY - MIN_MERGE_SIMILARITY) * specificity)


def merge_vectors(existing_vect, incoming_vect, merge_count):
    existing = vector_to_numpy(existing_vect)
    incoming = vector_to_numpy(incoming_vect)
    if existing.shape != incoming.shape:
        raise ValueError("vector dimensions do not match")
    weight = max(1, int(merge_count))
    merged = ((existing * weight) + incoming) / float(weight + 1)
    return vector_to_list(merged)


def _ensure_vector_dimension(conn, dimension):
    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError("vector dimension must be positive")
    row = conn.execute("SELECT value FROM metadata WHERE key='vector_dim'").fetchone()
    current = int(row[0]) if row else 0
    if current and current != dimension:
        raise ValueError(f"vector dimension mismatch: store={current}, incoming={dimension}")
    if not current:
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('vector_dim', ?)",
            (str(dimension),),
        )


def _concept_match_rows(conn, text, candidate_limit=96):
    terms = _concept_terms(text)
    if not terms:
        return []
    placeholders = ",".join("?" for _ in terms)
    return conn.execute(
        f"""
        SELECT c.id, c.vector
        FROM concept_terms AS t
        JOIN concepts AS c ON c.id = t.concept_id
        WHERE t.term IN ({placeholders})
        GROUP BY c.id
        ORDER BY COUNT(*) DESC, c.merge_count DESC
        LIMIT ?
        """,
        (*terms, int(candidate_limit)),
    ).fetchall()


def _best_concept_match(conn, vect, text=""):
    incoming = vector_to_numpy(vect)
    best = None
    rows = _concept_match_rows(conn, text)
    for idx, vector_blob in rows:
        vector = _vector_from_blob(vector_blob)
        if not vector:
            continue
        try:
            similarity = cosine_similarity(incoming, vector)
            threshold = merge_similarity_threshold(incoming, vector)
        except Exception:
            continue
        if best is None or similarity > best["similarity"]:
            best = {"index": int(idx), "similarity": similarity, "threshold": threshold, "vector": vector}
    return best


def get_literal_index(text, vect=None, path=DEFAULT_MAP_PATH):
    cleaned = _clean_text(text)
    normalized = normalize_text(cleaned)
    if not normalized:
        return -1
    conn = _connect(path)
    row = conn.execute(
        "SELECT literal_id FROM literal_aliases WHERE normalized=?",
        (normalized,),
    ).fetchone()
    if row is not None:
        return int(row[0])
    vector = vector_to_list(vect) if vect else []
    with conn:
        next_id = conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM literals").fetchone()[0]
        conn.execute(
            "INSERT INTO literals(id, text, vector, merge_count) VALUES(?, ?, ?, 1)",
            (int(next_id), cleaned, _vector_blob(vector)),
        )
        conn.execute(
            "INSERT INTO literal_aliases(normalized, literal_id) VALUES(?, ?)",
            (normalized, int(next_id)),
        )
    _clear_read_caches()
    return int(next_id)


@lru_cache(maxsize=65536)
def literal_from_index(index, path=DEFAULT_MAP_PATH):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return ""
    row = _connect(path).execute("SELECT text FROM literals WHERE id=?", (index,)).fetchone()
    return str(row[0]) if row else ""


@lru_cache(maxsize=65536)
def literal_vector_from_index(index, path=DEFAULT_MAP_PATH):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return []
    row = _connect(path).execute("SELECT vector FROM literals WHERE id=?", (index,)).fetchone()
    return _vector_from_blob(row[0]) if row else []


@lru_cache(maxsize=65536)
def concept_from_index(index, path=DEFAULT_MAP_PATH):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return ""
    row = _connect(path).execute("SELECT canonical FROM concepts WHERE id=?", (index,)).fetchone()
    return str(row[0]) if row else ""


@lru_cache(maxsize=65536)
def concept_vector_from_index(index, path=DEFAULT_MAP_PATH):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return []
    row = _connect(path).execute("SELECT vector FROM concepts WHERE id=?", (index,)).fetchone()
    return _vector_from_blob(row[0]) if row else []


@lru_cache(maxsize=65536)
def concept_vector_from_text(text, path=DEFAULT_MAP_PATH):
    normalized = normalize_text(text)
    if not normalized:
        return []
    row = _connect(path).execute(
        "SELECT concept_id FROM concept_aliases WHERE normalized=?",
        (normalized,),
    ).fetchone()
    return concept_vector_from_index(int(row[0]), path=path) if row else []


def existing_concept_index(text, path=DEFAULT_MAP_PATH):
    normalized = normalize_text(text)
    if not normalized:
        return -1
    row = _connect(path).execute(
        "SELECT concept_id FROM concept_aliases WHERE normalized=?",
        (normalized,),
    ).fetchone()
    return int(row[0]) if row is not None else -1


def _encode_text_vector(text, normalize=True):
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    return list(_encode_text_vector_cached(cleaned, bool(normalize)))


def _encode_text_vectors(texts, normalize=True):
    cleaned_values = []
    seen = set()
    for text in list(texts or []):
        cleaned = _clean_text(text)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        cleaned_values.append(cleaned)
    if not cleaned_values:
        return {}

    model = _load_embedding_model()
    if not model:
        return {cleaned: [] for cleaned in cleaned_values}

    try:
        encoded_values = model.encode(cleaned_values, normalize_embeddings=bool(normalize))
    except Exception:
        return {cleaned: _encode_text_vector(cleaned, normalize=normalize) for cleaned in cleaned_values}

    return {
        cleaned: vector_to_list(encoded)
        for cleaned, encoded in zip(cleaned_values, encoded_values)
    }


@lru_cache(maxsize=16384)
def _encode_text_vector_cached(cleaned, normalize):
    cleaned = _clean_text(cleaned)
    if not cleaned:
        return tuple()
    model = _load_embedding_model()
    if model:
        try:
            encoded = model.encode([cleaned], normalize_embeddings=normalize)
            return tuple(vector_to_list(encoded[0]))
        except Exception:
            pass
    return []


def str_to_magnitude_vector(text):
    return _encode_text_vector(text, normalize=False)


def text_magnitude_signal(text):
    vector = str_to_magnitude_vector(text)
    if not vector:
        return 0.0
    return vector_magnitude_signal(vector)


def concept_magnitude_signal(index, path=DEFAULT_MAP_PATH):
    vector = concept_vector_from_index(index, path=path)
    if vector:
        return vector_magnitude_signal(vector)
    return text_magnitude_signal(concept_from_index(index, path=path))


def literal_magnitude_signal(index, path=DEFAULT_MAP_PATH):
    vector = literal_vector_from_index(index, path=path)
    if vector:
        return vector_magnitude_signal(vector)
    return text_magnitude_signal(literal_from_index(index, path=path))


def _specifics_surface_text(specifics):
    parts = []
    seen = set()
    for item in list(specifics or []):
        if isinstance(item, dict):
            for key in ("context", "surface", "text", "normalized", "cue", "kind", "scope"):
                text = _clean_text(item.get(key, ""))
                if text:
                    break
            else:
                text = ""
        else:
            text = _clean_text(item)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        parts.append(text)
    return " ; ".join(parts)


def _concept_with_specifics(index, specifics=None, path=DEFAULT_MAP_PATH):
    base = _clean_text(concept_from_index(index, path=path))
    details = _specifics_surface_text(specifics)
    return f"{base} ; {details}" if base and details else base or details


def _literal_with_specifics(index, specifics=None, path=DEFAULT_MAP_PATH):
    base = _clean_text(literal_from_index(index, path=path))
    details = _specifics_surface_text(specifics)
    return f"{base} ; {details}" if base and details else base or details


def _token_specificity_signal(text, cap=8):
    tokens = _concept_terms(text)
    if not tokens:
        return 0.0
    return min(1.0, len(tokens) / float(max(1, cap)))


def _specifics_signal(*groups):
    count = 0
    for group in groups:
        count += len([item for item in list(group or []) if _clean_text(item)])
    return min(1.0, count / 4.0)


def _evidence_signal(text):
    text = _clean_text(text)
    if not text:
        return 0.0
    token_signal = _token_specificity_signal(text, cap=18)
    figure_signal = 1.0 if re.search(r"(\d|%|\b[A-Z]{2,}\b)", text) else 0.0
    return max(token_signal, figure_signal)


def connector_utility(
    subject_index,
    relation_index,
    predicate_index,
    subject_specifics=None,
    predicate_specifics=None,
    connection_specifics=None,
    subject_modifiers=None,
    predicate_modifiers=None,
    evidence_text="",
    path=DEFAULT_MAP_PATH,
):
    subject_text = _concept_with_specifics(subject_index, subject_specifics, path=path)
    predicate_text = _concept_with_specifics(predicate_index, predicate_specifics, path=path)
    relation_text = _literal_with_specifics(relation_index, connection_specifics, path=path)

    subject_score = max(
        _token_specificity_signal(subject_text),
        _specifics_signal(subject_specifics, subject_modifiers),
    )
    predicate_score = max(
        _token_specificity_signal(predicate_text),
        _specifics_signal(predicate_specifics, predicate_modifiers),
    )
    relation_score = max(
        _token_specificity_signal(relation_text, cap=3),
        _specifics_signal(connection_specifics),
    )
    endpoint_score = (subject_score + predicate_score) / 2.0
    detail_score = max(
        _specifics_signal(
            subject_specifics,
            predicate_specifics,
            connection_specifics,
            subject_modifiers,
            predicate_modifiers,
        ),
        _evidence_signal(evidence_text),
    )
    utility = 0.12 + (0.38 * endpoint_score) + (0.25 * relation_score) + (0.25 * detail_score)
    return float(max(0.0, min(1.0, utility)))


def concept_similarity(left_index, right_index, path=DEFAULT_MAP_PATH):
    return text_similarity(
        concept_from_index(left_index, path=path),
        concept_from_index(right_index, path=path),
    )


def literal_similarity(left_index, right_index, path=DEFAULT_MAP_PATH):
    left_vector = literal_vector_from_index(left_index, path=path)
    right_vector = literal_vector_from_index(right_index, path=path)
    if left_vector and right_vector:
        return cosine_similarity(left_vector, right_vector)
    return text_similarity(
        literal_from_index(left_index, path=path),
        literal_from_index(right_index, path=path),
    )


def get_or_create_index(text, vect, path=DEFAULT_MAP_PATH):
    cleaned = _clean_text(text)
    normalized = normalize_text(cleaned)
    if not normalized:
        return -1
    vector = vector_to_list(vect)
    conn = _connect(path)
    row = conn.execute(
        "SELECT concept_id FROM concept_aliases WHERE normalized=?",
        (normalized,),
    ).fetchone()
    if row is not None:
        return int(row[0])

    with conn:
        _ensure_vector_dimension(conn, len(vector))
        match = _best_concept_match(conn, vector, cleaned)
        if match and match["similarity"] >= match["threshold"]:
            idx = int(match["index"])
            row = conn.execute(
                "SELECT canonical, vector, merge_count FROM concepts WHERE id=?",
                (idx,),
            ).fetchone()
            canonical, existing_blob, merge_count = row
            merged_vector = merge_vectors(_vector_from_blob(existing_blob), vector, merge_count)
            canonical = cleaned if len(cleaned) > len(str(canonical or "")) else canonical
            conn.execute(
                "UPDATE concepts SET canonical=?, vector=?, specificity=?, merge_count=? WHERE id=?",
                (
                    canonical,
                    _vector_blob(merged_vector),
                    vector_specificity(merged_vector),
                    int(merge_count) + 1,
                    idx,
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO concept_aliases(normalized, concept_id) VALUES(?, ?)",
                (normalized, idx),
            )
            _store_concept_terms(conn, idx, canonical, cleaned)
            _clear_read_caches()
            return idx

        next_id = conn.execute("SELECT COALESCE(MAX(id), -1) + 1 FROM concepts").fetchone()[0]
        conn.execute(
            "INSERT INTO concepts(id, canonical, vector, specificity, merge_count) VALUES(?, ?, ?, ?, 1)",
            (int(next_id), cleaned, _vector_blob(vector), vector_specificity(vector)),
        )
        conn.execute(
            "INSERT INTO concept_aliases(normalized, concept_id) VALUES(?, ?)",
            (normalized, int(next_id)),
        )
        _store_concept_terms(conn, next_id, cleaned)
    _clear_read_caches()
    return int(next_id)


get_index = get_or_create_index


def _load_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is False:
        return None
    if EMBEDDING_MODEL is not None:
        return EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        try:
            EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except TypeError:
            EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception:
            EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        EMBEDDING_MODEL = False
    return EMBEDDING_MODEL


def str_to_vector(text):
    return _encode_text_vector(text, normalize=True)
