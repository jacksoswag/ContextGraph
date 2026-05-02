import re, sqlite3, threading; from collections import OrderedDict; from functools import lru_cache; from pathlib import Path; import numpy as np; from u_constants import (DEFAULT_MAP_PATH, EMBEDDING_MODEL_NAME, MAX_MERGE_SIMILARITY, MIN_MERGE_SIMILARITY); from u_language_constants import (CANONICAL_MODIFIER_CUES, STOPWORD_TOKENS)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?"); EMBEDDING_MODEL = None; _DB_CONNECTIONS = {}; _WRITE_LOCK = threading.RLock(); TEXT_VECTOR_CACHE_LIMIT = 32768; _TEXT_VECTOR_CACHE = OrderedDict()
# Normalizes text for semantic word storage.
def normalize_text(text): return " ".join(str(text or "").strip().lower().split())
# Cleans text for semantic word storage.
def _clean_text(text): return " ".join(str(text or "").strip().split())
# Computes canonical quality for semantic word storage.
def _canonical_quality(text):
    clean = _clean_text(text); tokens = [token.lower() for token in TOKEN_RE.findall(clean)]
    if not tokens: return (-1,)
    cue_count = sum(1 for token in tokens if token in CANONICAL_MODIFIER_CUES); dangling_cue = int(tokens[-1] in CANONICAL_MODIFIER_CUES); meaningful = [token for token in tokens if token not in CANONICAL_MODIFIER_CUES and token not in STOPWORD_TOKENS]; overlong = max(0, len(meaningful) - 5)
    return (-dangling_cue, -cue_count, min(len(meaningful), 5), -overlong, -len(clean))
# Chooses the cleaner canonical label between existing and incoming text.
def _prefer_canonical(existing, incoming):
    existing = _clean_text(existing); incoming = _clean_text(incoming)
    if not existing: return incoming
    if not incoming: return existing
    incoming_quality = _canonical_quality(incoming); existing_quality = _canonical_quality(existing)
    if incoming_quality > existing_quality: return incoming
    if incoming_quality < existing_quality and incoming_quality[:-1] != existing_quality[:-1]: return existing
    return incoming
# Converts a vector-like value into a non-empty float32 numpy array.
def vector_to_numpy(vect):
    array = np.asarray(vect, dtype=np.float32).reshape(-1)
    if array.size == 0: raise ValueError("vector cannot be empty")
    return array
# Returns vector to list for semantic word storage.
def vector_to_list(vect):
    return [float(value) for value in vector_to_numpy(vect).tolist()]
# Returns vector blob for semantic word storage.
def _vector_blob(vect):
    if not vect: return None
    return vector_to_numpy(vect).astype(np.float32).tobytes()
# Decodes a stored float32 vector blob into a Python list.
def _vector_from_blob(blob):
    if not blob: return []
    return vector_to_list(np.frombuffer(blob, dtype=np.float32))
# Resolves the SQLite word-map path under the configured sql directory.
def _db_path(path=DEFAULT_MAP_PATH):
    db_path = Path(path or DEFAULT_MAP_PATH).expanduser()
    if not db_path.is_absolute(): db_path = Path(__file__).resolve().parent / db_path
    return db_path
# Opens a thread-local SQLite word-map connection and ensures schema readiness.
def _connect(path=DEFAULT_MAP_PATH):
    db_path = _db_path(path); db_path.parent.mkdir(parents=True, exist_ok=True); cache_key = (threading.get_ident(), str(db_path)); conn = _DB_CONNECTIONS.get(cache_key)
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None); conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=NORMAL"); conn.execute("PRAGMA temp_store=MEMORY"); _DB_CONNECTIONS[cache_key] = conn
    _ensure_schema(conn)
    return conn
# Creates the concept, literal, alias, and metadata tables used by semantic storage.
def _ensure_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);; CREATE TABLE IF NOT EXISTS concepts (id INTEGER PRIMARY KEY, canonical TEXT NOT NULL, vector BLOB, specificity REAL NOT NULL DEFAULT 0, merge_count INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS concept_aliases (normalized TEXT PRIMARY KEY, concept_id INTEGER NOT NULL REFERENCES concepts(id));; CREATE TABLE IF NOT EXISTS concept_terms (term TEXT NOT NULL, concept_id INTEGER NOT NULL REFERENCES concepts(id), PRIMARY KEY(term, concept_id));
        CREATE INDEX IF NOT EXISTS idx_concept_terms_concept ON concept_terms(concept_id);; CREATE TABLE IF NOT EXISTS literals (id INTEGER PRIMARY KEY, text TEXT NOT NULL, vector BLOB, merge_count INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS literal_aliases (normalized TEXT PRIMARY KEY, literal_id INTEGER NOT NULL REFERENCES literals(id));""")
    _ensure_concept_terms(conn)
# Returns concept terms for semantic word storage.
def _concept_terms(text):
    return tuple(sorted({token.lower() for token in TOKEN_RE.findall(normalize_text(text))}))
# Stores searchable token terms for a concept id.
def _store_concept_terms(conn, concept_id, *texts):
    terms = set()
    for text in texts: terms.update(_concept_terms(text))
    if not terms: return
    conn.executemany("INSERT OR IGNORE INTO concept_terms(term, concept_id) VALUES(?, ?)", [(term, int(concept_id)) for term in terms])
# Ensures concept term rows exist for all stored concepts.
def _ensure_concept_terms(conn):
    row = conn.execute("SELECT value FROM metadata WHERE key='concept_terms_version'").fetchone()
    if row and str(row[0]) == "1": return
    with conn:
        conn.execute("DELETE FROM concept_terms")
        for concept_id, canonical in conn.execute("SELECT id, canonical FROM concepts"): _store_concept_terms(conn, concept_id, canonical)
        for normalized, concept_id in conn.execute("SELECT normalized, concept_id FROM concept_aliases"): _store_concept_terms(conn, concept_id, normalized)
        conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('concept_terms_version', '1')")
# Clears read caches caches or state.
def _clear_read_caches():
    concept_from_index.cache_clear(); concept_vector_from_index.cache_clear(); literal_from_index.cache_clear(); concept_vector_from_text.cache_clear()
# Stores one encoded text vector in the bounded in-memory cache.
def _remember_text_vector(cleaned, normalize, vector):
    key = (cleaned, bool(normalize)); _TEXT_VECTOR_CACHE[key] = tuple(vector); _TEXT_VECTOR_CACHE.move_to_end(key)
    while len(_TEXT_VECTOR_CACHE) > TEXT_VECTOR_CACHE_LIMIT:
        _TEXT_VECTOR_CACHE.popitem(last=False)
# Computes cosine similarity for semantic word storage.
def cosine_similarity(left, right):
    left_vec = vector_to_numpy(left); right_vec = vector_to_numpy(right)
    if left_vec.shape != right_vec.shape: raise ValueError("vector dimensions do not match")
    left_norm = float(np.linalg.norm(left_vec)); right_norm = float(np.linalg.norm(right_vec))
    if left_norm <= 1e-12 or right_norm <= 1e-12: return 0.0
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))
# Computes text similarity for semantic word storage.
def text_similarity(left_text, right_text):
    left = _clean_text(left_text); right = _clean_text(right_text)
    if not left or not right: return 0.0
    if normalize_text(left) == normalize_text(right): return 1.0
    if right < left: left, right = right, left
    return _text_similarity_cached(left, right)
# Computes cached cosine similarity for two cleaned text strings.
@lru_cache(maxsize=32768)
def _text_similarity_cached(left, right):
    return cosine_similarity(_encode_text_vector(left, normalize=True), _encode_text_vector(right, normalize=True))
# Computes vector specificity for semantic word storage.
def vector_specificity(vect):
    vec = vector_to_numpy(vect); dimension = max(1, int(vec.size)); magnitude = float(np.linalg.norm(vec)); nonzero_ratio = float(np.count_nonzero(np.abs(vec) > 1e-8)) / float(dimension); magnitude_signal = magnitude / (magnitude + 1.0); sparsity_signal = 1.0 - nonzero_ratio; specificity = (0.8 * magnitude_signal) + (0.2 * sparsity_signal)
    return float(max(0.0, min(1.0, specificity)))
# Merges similarity threshold while preserving stronger data.
def merge_similarity_threshold(left_vect, right_vect):
    specificity = max(vector_specificity(left_vect), vector_specificity(right_vect))
    return MIN_MERGE_SIMILARITY + ((MAX_MERGE_SIMILARITY - MIN_MERGE_SIMILARITY) * specificity)
# Merges vectors while preserving stronger data.
def merge_vectors(existing_vect, incoming_vect, merge_count):
    existing = vector_to_numpy(existing_vect); incoming = vector_to_numpy(incoming_vect)
    if existing.shape != incoming.shape: raise ValueError("vector dimensions do not match")
    weight = max(1, int(merge_count)); merged = ((existing * weight) + incoming) / float(weight + 1)
    return vector_to_list(merged)
# Ensures the stored embedding dimension matches the active model.
def _ensure_vector_dimension(conn, dimension):
    dimension = int(dimension)
    if dimension <= 0: raise ValueError("vector dimension must be positive")
    row = conn.execute("SELECT value FROM metadata WHERE key='vector_dim'").fetchone(); current = int(row[0]) if row else 0
    if current and current != dimension: raise ValueError(f"vector dimension mismatch: store={current}, incoming={dimension}")
    if not current: conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('vector_dim', ?)", (str(dimension),))
# Returns concept match rows used by semantic word storage.
def _concept_match_rows(conn, text, candidate_limit=96):
    terms = _concept_terms(text)
    if not terms: return []
    placeholders = ",".join("?" for _ in terms)
    return conn.execute(
        f"""
        SELECT c.id, c.vector; FROM concept_terms AS t; JOIN concepts AS c ON c.id = t.concept_id; WHERE t.term IN ({placeholders}); GROUP BY c.id; ORDER BY COUNT(*) DESC, c.merge_count DESC; LIMIT ?
        """, (*terms, int(candidate_limit))).fetchall()
# Returns the closest merge-compatible concept row for text and vector.
def _best_concept_match(conn, vect, text=""):
    incoming = vector_to_numpy(vect); best = None; rows = _concept_match_rows(conn, text)
    for idx, vector_blob in rows:
        vector = _vector_from_blob(vector_blob)
        if not vector: continue
        try:
            similarity = cosine_similarity(incoming, vector); threshold = merge_similarity_threshold(incoming, vector)
        except Exception: continue
        if best is None or similarity > best["similarity"]:
            best = {"index": int(idx), "similarity": similarity, "threshold": threshold, "vector": vector}
    return best
# Returns the literal id for exact text, creating a literal row when needed.
def get_literal_index(text, vect=None, path=DEFAULT_MAP_PATH):
    cleaned = _clean_text(text); normalized = normalize_text(cleaned)
    if not normalized: return -1
    conn = _connect(path); row = conn.execute("SELECT literal_id FROM literal_aliases WHERE normalized=?", (normalized,)).fetchone()
    if row is not None: return int(row[0])
    vector = vector_to_list(vect) if vect else []
    with _WRITE_LOCK:
        row = conn.execute("SELECT literal_id FROM literal_aliases WHERE normalized=?", (normalized,)).fetchone()
        if row is not None: return int(row[0])
        with conn:
            cursor = conn.execute("INSERT INTO literals(text, vector, merge_count) VALUES(?, ?, 1)", (cleaned, _vector_blob(vector))); literal_id = int(cursor.lastrowid); conn.execute("INSERT OR IGNORE INTO literal_aliases(normalized, literal_id) VALUES(?, ?)", (normalized, literal_id))
            row = conn.execute("SELECT literal_id FROM literal_aliases WHERE normalized=?", (normalized,)).fetchone()
    _clear_read_caches()
    return int(row[0]) if row is not None else literal_id
# Returns literal text for a stored literal id.
@lru_cache(maxsize=65536)
def literal_from_index(index, path=DEFAULT_MAP_PATH):
    try: index = int(index)
    except (TypeError, ValueError): return ""
    row = _connect(path).execute("SELECT text FROM literals WHERE id=?", (index,)).fetchone()
    return str(row[0]) if row else ""
# Returns canonical concept text for a stored concept id.
@lru_cache(maxsize=65536)
def concept_from_index(index, path=DEFAULT_MAP_PATH):
    try: index = int(index)
    except (TypeError, ValueError): return ""
    row = _connect(path).execute("SELECT canonical FROM concepts WHERE id=?", (index,)).fetchone()
    return str(row[0]) if row else ""
# Returns concept vector from index for semantic word storage.
@lru_cache(maxsize=65536)
def concept_vector_from_index(index, path=DEFAULT_MAP_PATH):
    try: index = int(index)
    except (TypeError, ValueError): return []
    row = _connect(path).execute("SELECT vector FROM concepts WHERE id=?", (index,)).fetchone()
    return _vector_from_blob(row[0]) if row else []
# Returns concept vector from text for semantic word storage.
@lru_cache(maxsize=65536)
def concept_vector_from_text(text, path=DEFAULT_MAP_PATH):
    normalized = normalize_text(text)
    if not normalized: return []
    row = _connect(path).execute("SELECT concept_id FROM concept_aliases WHERE normalized=?", (normalized,)).fetchone()
    return concept_vector_from_index(int(row[0]), path=path) if row else []
# Returns an existing existing concept index without creating new data.
def existing_concept_index(text, path=DEFAULT_MAP_PATH):
    normalized = normalize_text(text)
    if not normalized: return -1
    row = _connect(path).execute("SELECT concept_id FROM concept_aliases WHERE normalized=?", (normalized,)).fetchone()
    return int(row[0]) if row is not None else -1
# Encodes text into a normalized semantic vector.
def _encode_text_vector(text, normalize=True):
    cleaned = _clean_text(text)
    if not cleaned: return []
    cached = _TEXT_VECTOR_CACHE.get((cleaned, bool(normalize)))
    if cached is not None:
        _TEXT_VECTOR_CACHE.move_to_end((cleaned, bool(normalize)))
        return list(cached)
    return list(_encode_text_vector_cached(cleaned, bool(normalize)))
# Encodes unique text strings into cached semantic vectors.
def _encode_text_vectors(texts, normalize=True):
    cleaned_values = []; seen = set()
    for text in list(texts or []):
        cleaned = _clean_text(text)
        if not cleaned or cleaned in seen: continue
        seen.add(cleaned); cleaned_values.append(cleaned)
    if not cleaned_values: return {}
    model = _load_embedding_model()
    if not model: return {cleaned: [] for cleaned in cleaned_values}
    try:
        encoded_values = model.encode(cleaned_values, normalize_embeddings=bool(normalize))
    except Exception: return {cleaned: _encode_text_vector(cleaned, normalize=normalize) for cleaned in cleaned_values}
    vectors = {}
    for cleaned, encoded in zip(cleaned_values, encoded_values):
        vector = vector_to_list(encoded); _remember_text_vector(cleaned, normalize, vector); vectors[cleaned] = vector
    return vectors
# Warms the text-vector cache for upcoming extraction or merge work.
def precache_text_vectors(texts, normalize=True):
    return _encode_text_vectors(texts, normalize=normalize)
# Encodes one cleaned text string through the cached embedding model.
@lru_cache(maxsize=16384)
def _encode_text_vector_cached(cleaned, normalize):
    cleaned = _clean_text(cleaned)
    if not cleaned: return tuple()
    model = _load_embedding_model()
    if model:
        try:
            encoded = model.encode([cleaned], normalize_embeddings=normalize)
            return tuple(vector_to_list(encoded[0]))
        except Exception: pass
    return []
# Returns specifics surface text for semantic word storage.
def _specifics_surface_text(specifics):
    parts = []; seen = set()
    for item in list(specifics or []):
        if isinstance(item, dict):
            for key in ("context", "surface", "text", "normalized", "cue", "kind", "scope"):
                text = _clean_text(item.get(key, ""))
                if text: break
            else: text = ""
        else: text = _clean_text(item)
        key = text.lower()
        if not text or key in seen: continue
        seen.add(key); parts.append(text)
    return " ; ".join(parts)
# Builds concept with specifics for semantic word storage.
def _concept_with_specifics(index, specifics=None, path=DEFAULT_MAP_PATH):
    base = _clean_text(concept_from_index(index, path=path)); details = _specifics_surface_text(specifics)
    return f"{base} ; {details}" if base and details else base or details
# Builds literal with specifics for semantic word storage.
def _literal_with_specifics(index, specifics=None, path=DEFAULT_MAP_PATH):
    base = _clean_text(literal_from_index(index, path=path)); details = _specifics_surface_text(specifics)
    return f"{base} ; {details}" if base and details else base or details
# Computes token specificity signal for semantic word storage.
def _token_specificity_signal(text, cap=8):
    tokens = _concept_terms(text)
    if not tokens: return 0.0
    return min(1.0, len(tokens) / float(max(1, cap)))
# Builds specifics signal for semantic word storage.
def _specifics_signal(*groups):
    count = 0
    for group in groups: count += len([item for item in list(group or []) if _clean_text(item)])
    return min(1.0, count / 4.0)
# Computes connector utility for semantic word storage.
def connector_utility(subject_index, relation_index, predicate_index, subject_specifics=None, predicate_specifics=None, connection_specifics=None, subject_modifiers=None, predicate_modifiers=None, evidence_text="", path=DEFAULT_MAP_PATH):
    subject_text = _concept_with_specifics(subject_index, subject_specifics, path=path); predicate_text = _concept_with_specifics(predicate_index, predicate_specifics, path=path); relation_text = _literal_with_specifics(relation_index, connection_specifics, path=path)
    subject_score = max(_token_specificity_signal(subject_text), _specifics_signal(subject_specifics, subject_modifiers)); predicate_score = max(_token_specificity_signal(predicate_text), _specifics_signal(predicate_specifics, predicate_modifiers))
    connective_specificity = max(_token_specificity_signal(relation_text, cap=3), _specifics_signal(connection_specifics)); actant_specificity = (subject_score + predicate_score) / 2.0; inverse_connective_specificity = 1.0 - connective_specificity; utility = 0.05 + (0.75 * actant_specificity) + (0.20 * inverse_connective_specificity)
    return float(max(0.0, min(1.0, utility)))
# Returns the semantic concept id for text, creating or merging a vector-backed row as needed.
def get_or_create_index(text, vect, path=DEFAULT_MAP_PATH):
    cleaned = _clean_text(text); normalized = normalize_text(cleaned)
    if not normalized: return -1
    vector = vector_to_list(vect); conn = _connect(path); row = conn.execute("SELECT concept_id FROM concept_aliases WHERE normalized=?", (normalized,)).fetchone()
    if row is not None:
        idx = int(row[0]); existing = conn.execute("SELECT canonical FROM concepts WHERE id=?", (idx,)).fetchone()
        if existing is not None:
            preferred = _prefer_canonical(existing[0], cleaned)
            if preferred != existing[0]:
                with conn:
                    conn.execute("UPDATE concepts SET canonical=? WHERE id=?", (preferred, idx)); _store_concept_terms(conn, idx, preferred, cleaned)
                _clear_read_caches()
        return idx
    with _WRITE_LOCK:
        row = conn.execute("SELECT concept_id FROM concept_aliases WHERE normalized=?", (normalized,)).fetchone()
        if row is not None:
            idx = int(row[0]); existing = conn.execute("SELECT canonical FROM concepts WHERE id=?", (idx,)).fetchone()
            if existing is not None:
                preferred = _prefer_canonical(existing[0], cleaned)
                if preferred != existing[0]:
                    with conn:
                        conn.execute("UPDATE concepts SET canonical=? WHERE id=?", (preferred, idx)); _store_concept_terms(conn, idx, preferred, cleaned)
                    _clear_read_caches()
            return idx
        with conn:
            _ensure_vector_dimension(conn, len(vector))
            match = _best_concept_match(conn, vector, cleaned)
            if match and match["similarity"] >= match["threshold"]:
                idx = int(match["index"]); row = conn.execute("SELECT canonical, vector, merge_count FROM concepts WHERE id=?", (idx,)).fetchone(); canonical, existing_blob, merge_count = row; merged_vector = merge_vectors(_vector_from_blob(existing_blob), vector, merge_count); canonical = _prefer_canonical(canonical, cleaned)
                conn.execute("UPDATE concepts SET canonical=?, vector=?, specificity=?, merge_count=? WHERE id=?", (canonical, _vector_blob(merged_vector), vector_specificity(merged_vector), int(merge_count) + 1, idx)); conn.execute("INSERT OR REPLACE INTO concept_aliases(normalized, concept_id) VALUES(?, ?)", (normalized, idx))
                _store_concept_terms(conn, idx, canonical, cleaned); _clear_read_caches()
                return idx
            cursor = conn.execute("INSERT INTO concepts(canonical, vector, specificity, merge_count) VALUES(?, ?, ?, 1)", (cleaned, _vector_blob(vector), vector_specificity(vector))); concept_id = int(cursor.lastrowid); conn.execute("INSERT OR IGNORE INTO concept_aliases(normalized, concept_id) VALUES(?, ?)", (normalized, concept_id))
            row = conn.execute("SELECT concept_id FROM concept_aliases WHERE normalized=?", (normalized,)).fetchone(); idx = int(row[0]) if row is not None else concept_id; _store_concept_terms(conn, idx, cleaned)
    _clear_read_caches()
    return int(idx)
# Loads the sentence-transformer embedding model on first use.
def _load_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is False: return None
    if EMBEDDING_MODEL is not None: return EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        try: EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except TypeError: EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception: EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception: EMBEDDING_MODEL = False
    return EMBEDDING_MODEL
# Embeds text into the shared semantic vector space used by agents and merge logic.
def str_to_vector(text):
    return _encode_text_vector(text, normalize=True)
