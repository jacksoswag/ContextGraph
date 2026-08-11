import hashlib, json, re, sqlite3, threading, time; from urllib.parse import urlparse; from engine.common.constants import EXTRACTION_CACHE_PATH, EXTRACTION_CLAUSE_LIMIT, EXTRACTION_SENTENCE_LIMIT
from engine.extract.noise_cleanup import (MAX_AGENT_TOKENS, clean_agent_name, clean_clause_text, clean_source_text, clean_text, has_contraction_pronoun, is_usable_clause_text, is_usable_agent_text, is_usable_relation_text); from engine.extract.word_info_map import get_literal_index, literal_from_index, precache_text_vectors, text_similarity
from engine.common.language import (AMOUNT_UNIT_PATTERNS, CLAUSE_SPLIT_COORDINATORS, CORE_QUANTIFIER_TOKENS, DATE_MONTH_PATTERNS, LINKING_RELATION_PHRASES, LINKING_NEGATION_RELATIONS, MODIFIER_CUE_PHRASES, NEGATION_CUES, NEGATIVE_RELATION_FORMS, NON_NEGATING_PHRASES, PARTICIPIAL_MODIFIER_CUES, PERCENT_QUANTIFIER_QUALIFIERS, PERCENT_TOKENS, QUANTIFIER_EXCLUSION_PHRASES, QUANTIFIER_RULES, RELATION_STOPWORDS, SCOPE_BREAK_WORDS, SEMANTIC_NEGATION_EXACT_TERMS, SEMANTIC_NEGATION_PREFIXES, SEMANTIC_NEGATION_PROTOTYPES, SEMANTIC_NEGATION_SUFFIXES, STOPWORD_TOKENS, SUBJECT_PREFIX_PHRASES, TRAILING_NOISE_PHRASES, TRAILING_MODAL_TOKENS, YEAR_PREFIX_PATTERN)
from engine.extract.linguistic_roles import (dependency_relation, embedded_statement_text, looks_like_relation_word, parsed_tense, preparse_texts); from engine.agents.connection import ConnectionEndpoint
# Builds a regex-safe phrase alternation from language constants.
def _phrase_pattern(values):
    return "|".join(re.escape(value).replace(r"\ ", r"\s+") for value in sorted(values, key=len, reverse=True))
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+"); CLAUSE_SPLIT_RE = re.compile(rf"\s*(?:;|,\s+(?:{_phrase_pattern(CLAUSE_SPLIT_COORDINATORS)})\s+|\s+\b(?:{_phrase_pattern(CLAUSE_SPLIT_COORDINATORS)})\b\s+)\s*", re.IGNORECASE); TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
NUMBER_RE = re.compile(rf"(?:\$|€|£)?\b\d[\d,]*(?:\.\d+)?(?:\s?(?:{_phrase_pattern(AMOUNT_UNIT_PATTERNS)}))?(?=\W|$)", re.IGNORECASE); DATE_RE = re.compile(rf"\bQ[1-4]\s+{YEAR_PREFIX_PATTERN}\d{{2}}\b|\b(?:{'|'.join(DATE_MONTH_PATTERNS)})(?:\s+\d{{1,2}})?(?:,\s*)?\s+{YEAR_PREFIX_PATTERN}\d{{2}}\b|\b{YEAR_PREFIX_PATTERN}\d{{2}}\b", re.IGNORECASE)
LINKING_PATTERN = re.compile(rf"^(?P<subject>.+?)\s+(?P<relation>{_phrase_pattern(LINKING_RELATION_PHRASES)})\s+(?P<predicate>.+)$",re.IGNORECASE); SUBJECT_PREFIXES = re.compile(rf"^(?:{_phrase_pattern(SUBJECT_PREFIX_PHRASES)})\b\s+", re.IGNORECASE); TRAILING_NOISE_RE = re.compile(rf"[\s.·|]*(?:{_phrase_pattern(TRAILING_NOISE_PHRASES)})\b.*$", re.IGNORECASE)
MODIFIER_CUE_RE = re.compile(rf"\b(?:{_phrase_pattern(MODIFIER_CUE_PHRASES)})\b", re.IGNORECASE); NEGATION_RE = re.compile(rf"\b(?:{_phrase_pattern(NEGATION_CUES)})\b", re.IGNORECASE); WITHOUT_RE = re.compile(r"\bwithout\b", re.IGNORECASE); NON_NEGATING_RE = re.compile(rf"\b(?:{_phrase_pattern(NON_NEGATING_PHRASES)})\b", re.IGNORECASE)
SCOPE_BREAK_RE = re.compile(rf"[,;:]|\b(?:{_phrase_pattern(SCOPE_BREAK_WORDS)})\b", re.IGNORECASE); NEGATIVE_RELATION_RE = re.compile(rf"^(?:{_phrase_pattern(NEGATIVE_RELATION_FORMS)})$", re.IGNORECASE); SEMANTIC_NEGATION_THRESHOLD = 0.74
SEMANTIC_NEGATION_CANDIDATE_RE = re.compile(rf"^(?:{_phrase_pattern(SEMANTIC_NEGATION_PREFIXES)})[a-z]{{3,}}$|^[a-z]+(?:{_phrase_pattern(SEMANTIC_NEGATION_SUFFIXES)})$|^(?:{_phrase_pattern(SEMANTIC_NEGATION_EXACT_TERMS)})$", re.IGNORECASE); NEGATION_PRE_SCOPE_CHARS = 72; NEGATION_POST_SCOPE_CHARS = 72
QUANTIFIER_EXCLUSION_RE = re.compile(rf"^(?:{_phrase_pattern(QUANTIFIER_EXCLUSION_PHRASES)})\b", re.IGNORECASE); QUANTIFIER_PATTERNS = tuple(
    (quantifier, re.compile(rf"^(?:{_phrase_pattern(phrases)})\b", re.IGNORECASE))
    for quantifier, phrases in QUANTIFIER_RULES
); PERCENT_QUANTIFIER_RE = re.compile(rf"^(?P<qualifier>{_phrase_pattern(PERCENT_QUANTIFIER_QUALIFIERS)})?\s*(?P<value>\d+(?:\.\d+)?)\s*(?:{_phrase_pattern(PERCENT_TOKENS)})\s+(?:of\b)?", re.IGNORECASE)
LEADING_MODIFIER_RE = re.compile(rf"^(?P<modifier>{MODIFIER_CUE_RE.pattern}[^,;]{{0,120}})[,;]\s*(?P<core>.+)$", re.IGNORECASE)
TRAILING_MODIFIER_RE = re.compile(rf"^(?P<core>.+?)\s+(?P<modifier>{MODIFIER_CUE_RE.pattern}\s+.+)$", re.IGNORECASE); LEADING_PARTICIPIAL_MODIFIER_RE = re.compile(rf"^(?P<modifier>[A-Za-z]+ing\s+(?:{_phrase_pattern(PARTICIPIAL_MODIFIER_CUES)}))\s+(?P<core>.+)$", re.IGNORECASE); EXTRACTION_CACHE_VERSION = "logic-extractor-v15"
_EXTRACTION_CACHE_LOCK = threading.RLock(); _EXTRACTION_CACHE_CONNECTIONS = {}
# Opens the SQLite extraction cache for block-level reuse.
def _cache_connection():
    cache_key = (threading.get_ident(), str(EXTRACTION_CACHE_PATH)); conn = _EXTRACTION_CACHE_CONNECTIONS.get(cache_key)
    if conn is None:
        EXTRACTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True); conn = sqlite3.connect(EXTRACTION_CACHE_PATH, timeout=30.0, isolation_level=None); conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS extraction_cache (
                cache_key TEXT PRIMARY KEY, value TEXT NOT NULL,
                created REAL NOT NULL)""")
        _EXTRACTION_CACHE_CONNECTIONS[cache_key] = conn
    return conn
# Builds a versioned cache key from source and cleaned block text.
def _block_cache_key(content, source):
    digest = hashlib.sha256(); digest.update(EXTRACTION_CACHE_VERSION.encode("utf-8")); digest.update(b"\0"); digest.update(clean_text(source).encode("utf-8")); digest.update(b"\0"); digest.update(clean_text(content).encode("utf-8"))
    return digest.hexdigest()
# Reads a serialized extraction result from the SQLite cache.
def _cache_get(key):
    try: row = _cache_connection().execute("SELECT value FROM extraction_cache WHERE cache_key=?", (key,)).fetchone()
    except sqlite3.Error: return None
    if row is None: return None
    try: return json.loads(row[0])
    except json.JSONDecodeError: return None
# Writes a serialized extraction result to the SQLite cache.
def _cache_set(key, value):
    try: encoded = json.dumps(value)
    except TypeError: return
    try:
        with _EXTRACTION_CACHE_LOCK:
            _cache_connection().execute("""INSERT OR REPLACE INTO extraction_cache(cache_key, value, created) VALUES(?, ?, ?)""", (key, encoded, time.time()))
    except sqlite3.Error: return
# Cleans part for logic extraction.
def _clean_part(text):
    text = clean_text(text); text = TRAILING_NOISE_RE.sub("", text)
    text = re.sub(r"^[,.:;()\[\]\s]+|[,.:;()\[\]\s]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
# Returns literal text for logic extraction.
def _literal_text(index):
    try: index = int(index)
    except (TypeError, ValueError): return ""
    return literal_from_index(index) if index >= 0 else ""
# Serializes endpoint metadata for extraction-cache storage.
def _serialize_endpoint(endpoint):
    return {"quantifier": _literal_text(getattr(endpoint, "quantifier", -1)), "tense": _literal_text(getattr(endpoint, "tense", -1)), "truth": int(getattr(endpoint, "truth", -1)), "asu": endpoint.asu_value() if isinstance(endpoint, ConnectionEndpoint) else "", "modifiers": endpoint.modifier_value() if isinstance(endpoint, ConnectionEndpoint) else [],}
# Rebuilds a ConnectionEndpoint from cached endpoint payload data.
def _endpoint_from_payload(payload):
    payload = dict(payload or {}); quantifier = get_literal_index(payload.get("quantifier", "")) if payload.get("quantifier") else -1; tense = get_literal_index(payload.get("tense", "")) if payload.get("tense") else -1; modifiers = [ConnectionEndpoint.register_modifier(value) for value in list(payload.get("modifiers", []) or [])]
    return ConnectionEndpoint(quantifier=quantifier, tense=tense, truth=int(payload.get("truth", -1)), ASU_idx=payload.get("asu", ""), modifier_idx=[idx for idx in modifiers if idx >= 0],)
# Serializes one connection record for extraction-cache storage.
def _serialize_connection(connection):
    return {"subject": _serialize_endpoint(connection.get("subject")), "predicate": _serialize_endpoint(connection.get("predicate")), "connection": _literal_text(connection.get("connection")), "source": connection.get("source", "unknown"), "text": connection.get("text", ""), "subject_specifics": connection.get("subject_specifics", []), "predicate_specifics": connection.get("predicate_specifics", []), "connection_specifics": connection.get("connection_specifics", [])}
# Rebuilds one connection record from cached extraction payload data.
def _connection_from_payload(payload):
    payload = dict(payload or {}); relation = _clean_part(payload.get("connection", ""))
    if not relation or not is_usable_relation_text(relation): return None
    subject = _endpoint_from_payload(payload.get("subject", {})); predicate = _endpoint_from_payload(payload.get("predicate", {}))
    if not subject.asu_value() or not predicate.asu_value(): return None
    return {"subject": subject, "predicate": predicate, "connection": get_literal_index(relation), "source": payload.get("source", "unknown"), "text": payload.get("text", ""), "subject_specifics": payload.get("subject_specifics", []), "predicate_specifics": payload.get("predicate_specifics", []), "connection_specifics": payload.get("connection_specifics", [])}
# Returns the display source for a scraped block.
def _source_for_block(block):
    raw_tag = clean_source_text((block or {}).get("tag", "")); url = _clean_url((block or {}).get("url", ""))
    if "|" in raw_tag:
        title, embedded_url = raw_tag.rsplit("|", 1); tag = _clean_part(title); url = url or _clean_url(embedded_url)
    else: tag = _clean_part(raw_tag)
    if tag and url: return f"{tag}|{url}"
    return url or tag or "unknown"
# Cleans url for logic extraction.
def _clean_url(url):
    url = str(url or "").strip().strip("\"'<>[]()"); parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc: return ""
    return url
# Splits sentences into usable parts.
def _split_sentences(text):
    pieces = []
    for sentence in SENTENCE_RE.split(str(text or "")):
        clean = clean_clause_text(sentence)
        if clean: pieces.append(clean.rstrip(".!?"))
        if len(pieces) >= EXTRACTION_SENTENCE_LIMIT: break
    return pieces
# Splits clauses into usable parts.
def _split_clauses(sentence):
    clauses = []
    for clause in CLAUSE_SPLIT_RE.split(sentence):
        clean = clean_clause_text(clause)
        if not clean: continue
        clauses.append(clean.rstrip(".!?"))
        if len(clauses) >= EXTRACTION_CLAUSE_LIMIT: break
    return clauses
# Builds relation span for logic extraction.
def _relation_span(text, relation):
    relation = _clean_part(relation).lower()
    if not relation: return None
    match = re.search(rf"\b{re.escape(relation)}\b", text, flags=re.IGNORECASE)
    if match: return match.span()
    relation_tokens = TOKEN_RE.findall(relation)
    if not relation_tokens: return None
    for token in relation_tokens:
        match = re.search(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE)
        if match: return match.span()
    return None
# Returns whether the input has scope break between.
def _has_scope_break_between(text, left, right):
    start, end = sorted((max(0, int(left)), max(0, int(right))))
    return bool(SCOPE_BREAK_RE.search(text[start:end]))
# Returns negation in relation scope for logic extraction.
def _negation_in_relation_scope(text, relation):
    span = _relation_span(text, relation)
    if span is None: return bool(NEGATION_RE.search(text))
    relation_start, relation_end = span; relation_clean = _clean_part(relation).lower()
    if NEGATIVE_RELATION_RE.match(relation_clean): return True
    for match in NEGATION_RE.finditer(text):
        before_relation = match.end() <= relation_start; after_relation = match.start() >= relation_end; near_before = before_relation and relation_start - match.end() <= NEGATION_PRE_SCOPE_CHARS; near_after = after_relation and match.start() - relation_end <= NEGATION_POST_SCOPE_CHARS; overlaps_relation = not before_relation and not after_relation
        if not (near_before or near_after or overlaps_relation): continue
        if before_relation and _has_scope_break_between(text, match.end(), relation_start): continue
        if after_relation and _has_scope_break_between(text, relation_end, match.start()): continue
        return True
    for match in WITHOUT_RE.finditer(text):
        after_relation = match.start() >= relation_end
        if after_relation and relation_clean not in LINKING_NEGATION_RELATIONS: continue
        near_before = match.end() <= relation_start and relation_start - match.end() <= NEGATION_PRE_SCOPE_CHARS; near_after = after_relation and match.start() - relation_end <= NEGATION_POST_SCOPE_CHARS
        if not (near_before or near_after): continue
        if match.end() <= relation_start and _has_scope_break_between(text, match.end(), relation_start): continue
        if after_relation and _has_scope_break_between(text, relation_end, match.start()): continue
        return True
    return False
# Returns semantic negation score for logic extraction.
def _semantic_negation_score(text):
    clean = _clean_part(text).lower()
    if len(clean) < 4 or clean in STOPWORD_TOKENS: return 0.0
    if not SEMANTIC_NEGATION_CANDIDATE_RE.match(clean): return 0.0
    try: return max(text_similarity(clean, cue) for cue in SEMANTIC_NEGATION_PROTOTYPES)
    except Exception: return 0.0
# Returns semantic negation in scope for logic extraction.
def _semantic_negation_in_scope(text, relation):
    span = _relation_span(text, relation)
    if span is None: return False
    start, end = span; scoped = text[max(0, start - NEGATION_PRE_SCOPE_CHARS): min(len(text), end + NEGATION_POST_SCOPE_CHARS)]
    for token in TOKEN_RE.findall(scoped):
        if _semantic_negation_score(token) >= SEMANTIC_NEGATION_THRESHOLD: return True
    return False
# Returns the clause truth flag after scoped lexical and semantic negation checks.
def _truth_for_clause(clause, relation=""):
    clean = NON_NEGATING_RE.sub(" ", clean_text(clause).lower()); clean = re.sub(r"\s+", " ", clean).strip()
    if not clean: return 1
    if _negation_in_relation_scope(clean, relation): return 0
    if relation and _semantic_negation_in_scope(clean, relation): return 0
    return 1
# Returns the literal id for relation tense parsed from the clause.
def _tense_for_relation(relation, clause):
    return get_literal_index(parsed_tense(clause, relation))
# Converts leading percentage language into a normalized quantifier phrase.
def _quantifier_from_percent(text):
    match = PERCENT_QUANTIFIER_RE.search(text)
    if not match: return ""
    try: value = float(match.group("value"))
    except (TypeError, ValueError): return ""
    qualifier = (match.group("qualifier") or "").lower()
    if "less" in qualifier or "under" in qualifier or "below" in qualifier: value = max(0.0, value - 0.01)
    elif "more" in qualifier or "over" in qualifier or "above" in qualifier: value = min(100.0, value + 0.01)
    if value >= 99.5: return "all"
    if value > 50.0: return "most"
    if value <= 0.5: return "none"
    if value < 25.0: return "few"
    return "some"
# Returns a quantifier literal id inferred from the subject text.
def _quantifier_for_subject(subject):
    lower = _clean_part(subject).lower()
    if not lower: return -1
    if QUANTIFIER_EXCLUSION_RE.match(lower): return -1
    quantifier = _quantifier_from_percent(lower)
    if quantifier: return get_literal_index(quantifier)
    for quantifier, pattern in QUANTIFIER_PATTERNS:
        if pattern.search(lower): return get_literal_index(quantifier)
    return -1
# Returns modifier ids for logic extraction.
def _modifier_ids(text):
    modifiers = []
    for value in DATE_RE.findall(text):
        idx = ConnectionEndpoint.register_modifier(value)
        if idx >= 0 and idx not in modifiers: modifiers.append(idx)
    for value in NUMBER_RE.findall(text):
        if re.fullmatch(r"(?:18|19|20|21)\d{2}", _clean_part(value)): continue
        idx = ConnectionEndpoint.register_modifier(value)
        if idx >= 0 and idx not in modifiers: modifiers.append(idx)
    return modifiers[:4]
# Adds a modifier literal id to an endpoint payload without duplicates.
def _add_modifier(modifiers, value):
    clean = _clean_part(value); clean = re.sub(rf"^{MODIFIER_CUE_RE.pattern}\s+(?={MODIFIER_CUE_RE.pattern}\b)", "", clean, flags=re.IGNORECASE); clean = re.sub(rf"\s+{MODIFIER_CUE_RE.pattern}$", "", clean, flags=re.IGNORECASE)
    if not clean: return
    if has_contraction_pronoun(clean): return
    if not [token for token in TOKEN_RE.findall(clean) if token.lower() not in STOPWORD_TOKENS]: return
    idx = ConnectionEndpoint.register_modifier(clean)
    if idx >= 0 and idx not in modifiers: modifiers.append(idx)
# Builds drop specific literals for logic extraction.
def _drop_specific_literals(text):
    text = DATE_RE.sub(" ", text); text = NUMBER_RE.sub(" ", text)
    return clean_agent_name(text)
# Builds strip embedded statement clause for logic extraction.
def _strip_embedded_statement_clause(text, modifiers):
    statement = embedded_statement_text(text)
    if not statement: return text
    prefix = text[: max(0, str(text).find(statement))].strip()
    if prefix: _add_modifier(modifiers, prefix)
    return statement
# Cleans shorten core for logic extraction.
def _shorten_core(text):
    text = _clean_part(text); text = SUBJECT_PREFIXES.sub("", text); text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE); text = re.sub(rf"^(?:{_phrase_pattern(CORE_QUANTIFIER_TOKENS)})\s+", "", text, flags=re.IGNORECASE); text = re.sub(rf"\s+\b(?:{_phrase_pattern(TRAILING_MODAL_TOKENS)})\s+(?:not\s*)?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:that|which|who)\s*$", "", text, flags=re.IGNORECASE); text = re.sub(rf"^{MODIFIER_CUE_RE.pattern}\s+", "", text, flags=re.IGNORECASE); text = re.sub(rf"\s+{MODIFIER_CUE_RE.pattern}$", "", text, flags=re.IGNORECASE); tokens = TOKEN_RE.findall(text)
    if len(tokens) > MAX_AGENT_TOKENS: text = " ".join(tokens[-MAX_AGENT_TOKENS:])
    return clean_agent_name(text)
# Returns endpoint payload for logic extraction.
def _endpoint_payload(text, source, preferred_core=""):
    original = _clean_part(text); modifiers = _modifier_ids(original); specifics = _specifics(original, source); core = _drop_specific_literals(original)
    match = LEADING_MODIFIER_RE.match(core)
    if match:
        _add_modifier(modifiers, match.group("modifier")); core = _clean_part(match.group("core"))
    core = _strip_embedded_statement_clause(core, modifiers)
    match = LEADING_PARTICIPIAL_MODIFIER_RE.match(core)
    if match:
        _add_modifier(modifiers, match.group("modifier")); core = _clean_part(match.group("core"))
    match = TRAILING_MODIFIER_RE.match(core)
    if match:
        _add_modifier(modifiers, match.group("modifier")); core = _clean_part(match.group("core"))
    core = _shorten_core(core) or _shorten_core(preferred_core) or _shorten_core(original)
    return {"core": core, "modifier_ids": modifiers[:4], "specifics": specifics}
# Extracts concrete dates, numbers, and amounts from endpoint or clause text.
def _specifics(text, source):
    specifics = []; seen = set()
    for match in DATE_RE.findall(text):
        clean = _clean_part(match); key = ("date", clean.lower())
        if clean and key not in seen:
            seen.add(key); specifics.append({"kind": "date", "text": clean, "source": source})
    for match in NUMBER_RE.findall(text):
        clean = _clean_part(match)
        if re.fullmatch(r"(?:18|19|20|21)\d{2}", clean): continue
        key = ("number", clean.lower())
        if clean and key not in seen:
            seen.add(key); specifics.append({"kind": "amount", "text": clean, "source": source})
    return specifics
# Cleans trim endpoint for logic extraction.
def _trim_endpoint(text):
    return _endpoint_payload(text, "", preferred_core=text)["core"]
# Returns whether the input is amount tail.
def _is_amount_tail(predicate_raw, relation):
    clean = _clean_part(predicate_raw)
    if not clean: return False
    amount_match = NUMBER_RE.search(clean)
    if amount_match and amount_match.start() <= 16: return True
    leading_modifier_amount = re.match(rf"^{MODIFIER_CUE_RE.pattern}\s+(?:{NUMBER_RE.pattern}|{DATE_RE.pattern})", clean, flags=re.IGNORECASE)
    if leading_modifier_amount: return True
    if relation in {"is", "are", "was", "were", "become", "becomes", "became", "has", "have", "had"}:
        return bool(re.match(rf"^(?:{NUMBER_RE.pattern}|{DATE_RE.pattern})", clean, flags=re.IGNORECASE))
    return False
# Returns whether text looks like action verb.
def _looks_like_action_verb(token):
    raw = str(token or "").strip(); token = raw.lower()
    if token in RELATION_STOPWORDS: return False
    return looks_like_relation_word(raw)
# Packages a candidate subject, relation, predicate, and modifiers before validation.
def _candidate(subject_raw, relation, predicate_raw, clause):
    relation = _clean_part(relation).lower()
    if not is_usable_relation_text(relation): return None
    subject = _trim_endpoint(subject_raw)
    if _is_amount_tail(predicate_raw, relation): predicate = f"{relation} amount"
    else: predicate = _trim_endpoint(predicate_raw)
    if NUMBER_RE.match(predicate): predicate = f"{relation} amount"
    if not is_usable_agent_text(subject) or not is_usable_agent_text(predicate): return None
    if subject.lower() == predicate.lower(): return None
    return { "subject": subject, "subject_raw": _clean_part(subject_raw), "relation": relation, "predicate": predicate, "predicate_raw": _clean_part(predicate_raw), "clause": clean_clause_text(clause), }
# Extracts a candidate connection from action-verb clause structure.
def _candidate_from_action_clause(clean):
    matches = list(TOKEN_RE.finditer(clean))
    if len(matches) < 3: return None
    action_matches = [(idx, match) for idx, match in enumerate(matches[1:-1], 1) if _looks_like_action_verb(match.group(0))]; last_break = max(clean.rfind(","), clean.rfind(";")); action_matches.sort(key=lambda item: (item[1].start() <= last_break, item[1].start()))
    for idx, match in action_matches:
        relation = match.group(0).lower()
        if not _looks_like_action_verb(relation): continue
        predicate_raw = clean[matches[idx].end():]; candidate = _candidate(clean[:matches[idx].start()], relation, predicate_raw, clean)
        if candidate is None: continue
        return candidate
    return None
# Extracts a candidate connection from dependency-parser subject and predicate spans.
def _candidate_from_dependency(clean):
    dependency_candidate = dependency_relation(clean)
    if dependency_candidate is None: return None
    return _candidate(dependency_candidate["subject_raw"], dependency_candidate["relation"], dependency_candidate["predicate_raw"], clean)
# Chooses the best extraction strategy for one cleaned clause.
def _candidate_from_clause(clause):
    clean = clean_clause_text(clause).rstrip(".!?")
    if len(TOKEN_RE.findall(clean)) < 3: return None
    statement = embedded_statement_text(clean)
    if statement and statement != clean:
        nested = _candidate_from_clause(statement)
        if nested is not None: return nested
    dependency_candidate = _candidate_from_dependency(clean)
    if dependency_candidate is not None: return dependency_candidate
    match = LINKING_PATTERN.match(clean)
    if not match: return _candidate_from_action_clause(clean)
    relation = _clean_part(match.group("relation")).lower()
    if len(relation) < 2: return None
    return _candidate(match.group("subject"), relation, match.group("predicate"), clean)
# Converts a candidate into validated endpoint objects and connection metadata.
def _connection_from_candidate(candidate, source):
    relation = candidate["relation"]; clause = candidate["clause"]; truth = _truth_for_clause(clause, relation=relation); tense = _tense_for_relation(relation, clause); subject_payload = _endpoint_payload(candidate.get("subject_raw") or candidate["subject"], source, preferred_core=candidate["subject"])
    predicate_payload = _endpoint_payload(candidate.get("predicate_raw") or candidate["predicate"], source, preferred_core=candidate["predicate"])
    if _is_amount_tail(candidate.get("predicate_raw") or "", relation): predicate_payload["core"] = candidate["predicate"]
    subject = subject_payload["core"]; predicate = predicate_payload["core"]; subject_specifics = subject_payload["specifics"]; predicate_specifics = predicate_payload["specifics"]; connection_specifics = _specifics(clause, source)
    return {"subject": ConnectionEndpoint(quantifier=_quantifier_for_subject(candidate.get("subject_raw") or subject), tense=tense, truth=truth, ASU_idx=subject, modifier_idx=subject_payload["modifier_ids"]), "predicate": ConnectionEndpoint(quantifier=_quantifier_for_subject(candidate.get("predicate_raw") or predicate), tense=tense, truth=truth, ASU_idx=predicate, modifier_idx=predicate_payload["modifier_ids"]), "connection": get_literal_index(relation), "source": source, "text": clause, "subject_specifics": subject_specifics, "predicate_specifics": predicate_specifics, "connection_specifics": connection_specifics,}
# Builds block fingerprint for logic extraction.
def _block_fingerprint(block):
    content = clean_text((block or {}).get("content", "")); url = _clean_url((block or {}).get("url", "")); source = _source_for_block(block)
    if url: return ("url", url, hashlib.sha1(content.encode("utf-8")).hexdigest())
    return ("content", hashlib.sha1(f"{source}\0{content}".encode("utf-8")).hexdigest())
# Returns unique blocks for logic extraction.
def _unique_blocks(blocks):
    unique = []; seen = set()
    for block in list(blocks or []):
        content = clean_text((block or {}).get("content", ""))
        if not content: continue
        fingerprint = _block_fingerprint(block)
        if fingerprint in seen: continue
        seen.add(fingerprint); unique.append(block)
    return unique
# Extracts and caches connection records for one scraped source block.
def _connections_from_block(block, query=""):
    content = clean_text((block or {}).get("content", ""))
    if not content: return []
    source = _source_for_block(block) or query or "unknown"; cache_key = _block_cache_key(content, source); cached = _cache_get(cache_key)
    if isinstance(cached, list):
        hydrated = []
        for item in cached:
            connection = _connection_from_payload(item)
            if connection is not None: hydrated.append(connection)
        return hydrated
    candidates, clauses, seen = [], [], set()
    for sentence in _split_sentences(content):
        clauses.extend(_split_clauses(sentence))
        if len(clauses) >= EXTRACTION_CLAUSE_LIMIT:
            clauses = clauses[:EXTRACTION_CLAUSE_LIMIT]; break
    clauses = [clause for clause in clauses if is_usable_clause_text(clause)]; preparse_texts(clauses)
    for clause in clauses:
        candidate = _candidate_from_clause(clause)
        if candidate is None: continue
        key = (candidate["subject"].lower(), candidate["relation"].lower(), candidate["predicate"].lower(), source)
        if key in seen: continue
        seen.add(key); candidates.append(candidate)
    vector_warmup = []
    for candidate in candidates:
        vector_warmup.append(candidate.get("subject", "")); vector_warmup.append(candidate.get("predicate", ""))
    if vector_warmup: precache_text_vectors(vector_warmup)
    block_results = [_connection_from_candidate(candidate, source) for candidate in candidates]; _cache_set(cache_key, [_serialize_connection(connection) for connection in block_results])
    return block_results
# Extracts cleaned subject-relation-predicate connection records from scraped text blocks.
def find_connections(blocks, query="", connection_limit=None):
    limit = int(connection_limit or EXTRACTION_CLAUSE_LIMIT); results = []; seen = set()
    for block in _unique_blocks(blocks):
        for connection in _connections_from_block(block, query=query):
            subject_sp = connection.get("subject"); predicate_sp = connection.get("predicate")
            key = (subject_sp.asu_value().lower() if isinstance(subject_sp, ConnectionEndpoint) else "", str(connection.get("connection", "")).lower(), predicate_sp.asu_value().lower() if isinstance(predicate_sp, ConnectionEndpoint) else "", connection.get("source", ""), connection.get("text", ""),)
            if key in seen: continue
            seen.add(key); results.append(connection)
            if len(results) >= limit: return results
    return results
