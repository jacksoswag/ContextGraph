import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from constants import EXTRACTION_CLAUSE_LIMIT, EXTRACTION_SENTENCE_LIMIT
from d_noise_cleanup import (
    MAX_AGENT_TOKENS,
    clean_clause_text,
    clean_source_text,
    clean_text,
    is_usable_clause_text,
    is_usable_agent_text,
)
from d_word_info_map import get_literal_index, literal_from_index, precache_text_vectors
from linguistic_roles import (
    dependency_relation,
    embedded_statement_text,
    looks_like_relation_word,
    parsed_tense,
    preparse_texts,
)
from o_connection import ConnectionEndpoint


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:;|,\s+(?:and|or|but|so|while|whereas|although|though)\s+|\s+\b(?:and|or|but|so|while|whereas|although|though)\b\s+)\s*",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
NUMBER_RE = re.compile(
    r"(?:\$|€|£)?\b\d[\d,]*(?:\.\d+)?(?:\s?(?:%|percent|dollars?|euros?|pounds?|million|billion|thousand))?(?=\W|$)",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\bQ[1-4]\s+(?:18|19|20|21)\d{2}\b|"
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)(?:\s+\d{1,2})?(?:,\s*)?\s+(?:18|19|20|21)\d{2}\b|"
    r"\b(?:18|19|20|21)\d{2}\b",
    re.IGNORECASE,
)

LINKING_PATTERN = re.compile(
    r"^(?P<subject>.+?)\s+"
    r"(?P<relation>is|are|was|were|becomes?|became|has|have|had|will have|will be)"
    r"\s+(?P<predicate>.+)$",
    re.IGNORECASE,
)
RELATION_STOPWORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "that",
    "the",
    "this",
    "was",
    "were",
    "with",
}
SUBJECT_PREFIXES = re.compile(
    r"^(?:according to|in|on|at|during|after|before|because|that|when|where|while|although|though|whereas)\b\s+",
    re.IGNORECASE,
)
TRAILING_NOISE_RE = re.compile(
    r"\s+(?:according to|via|source:|read more|learn more)\b.*$",
    re.IGNORECASE,
)
MODIFIER_CUE_RE = re.compile(
    r"\b(?:about|above|according to|across|after|against|along|although|among|around|as of|at|"
    r"based on|because|because of|before|below|between|by|despite|during|for|from|"
    r"in|including|inside|into|near|of|on|over|per|through|to|under|using|"
    r"via|where|whereas|when|while|with|within|without)\b",
    re.IGNORECASE,
)
LEADING_MODIFIER_RE = re.compile(
    rf"^(?P<modifier>{MODIFIER_CUE_RE.pattern}[^,;]{{0,120}})[,;]\s*(?P<core>.+)$",
    re.IGNORECASE,
)
TRAILING_MODIFIER_RE = re.compile(
    rf"^(?P<core>.+?)\s+(?P<modifier>{MODIFIER_CUE_RE.pattern}\s+.+)$",
    re.IGNORECASE,
)
EXTRACTION_CACHE_VERSION = "logic-extractor-v5"
EXTRACTION_CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "extraction_cache.sqlite3"
_EXTRACTION_CACHE_LOCK = threading.RLock()
_EXTRACTION_CACHE_CONNECTIONS = {}


def _cache_connection():
    cache_key = (threading.get_ident(), str(EXTRACTION_CACHE_PATH))
    conn = _EXTRACTION_CACHE_CONNECTIONS.get(cache_key)
    if conn is None:
        EXTRACTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(EXTRACTION_CACHE_PATH, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_cache (
                cache_key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created REAL NOT NULL
            )
            """
        )
        _EXTRACTION_CACHE_CONNECTIONS[cache_key] = conn
    return conn


def _block_cache_key(content, source):
    digest = hashlib.sha256()
    digest.update(EXTRACTION_CACHE_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(clean_text(source).encode("utf-8"))
    digest.update(b"\0")
    digest.update(clean_text(content).encode("utf-8"))
    return digest.hexdigest()


def _cache_get(key):
    try:
        row = _cache_connection().execute(
            "SELECT value FROM extraction_cache WHERE cache_key=?",
            (key,),
        ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def _cache_set(key, value):
    try:
        encoded = json.dumps(value)
    except TypeError:
        return
    try:
        with _EXTRACTION_CACHE_LOCK:
            _cache_connection().execute(
                """
                INSERT OR REPLACE INTO extraction_cache(cache_key, value, created)
                VALUES(?, ?, ?)
                """,
                (key, encoded, time.time()),
            )
    except sqlite3.Error:
        return


def _clean_part(text):
    text = clean_text(text)
    text = TRAILING_NOISE_RE.sub("", text)
    text = re.sub(r"^[,.:;()\[\]\s]+|[,.:;()\[\]\s]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _literal_text(index):
    try:
        index = int(index)
    except (TypeError, ValueError):
        return ""
    return literal_from_index(index) if index >= 0 else ""


def _serialize_endpoint(endpoint):
    return {
        "quantifier": _literal_text(getattr(endpoint, "quantifier", -1)),
        "tense": _literal_text(getattr(endpoint, "tense", -1)),
        "truth": int(getattr(endpoint, "truth", -1)),
        "asu": endpoint.asu_value() if isinstance(endpoint, ConnectionEndpoint) else "",
        "modifiers": endpoint.modifier_value() if isinstance(endpoint, ConnectionEndpoint) else [],
    }


def _endpoint_from_payload(payload):
    payload = dict(payload or {})
    quantifier = get_literal_index(payload.get("quantifier", "")) if payload.get("quantifier") else -1
    tense = get_literal_index(payload.get("tense", "")) if payload.get("tense") else -1
    modifiers = [
        ConnectionEndpoint.register_modifier(value)
        for value in list(payload.get("modifiers", []) or [])
    ]
    return ConnectionEndpoint(
        quantifier=quantifier,
        tense=tense,
        truth=int(payload.get("truth", -1)),
        ASU_idx=payload.get("asu", ""),
        modifier_idx=[idx for idx in modifiers if idx >= 0],
    )


def _serialize_connection(connection):
    return {
        "subject": _serialize_endpoint(connection.get("subject")),
        "predicate": _serialize_endpoint(connection.get("predicate")),
        "connection": _literal_text(connection.get("connection")),
        "source": connection.get("source", "unknown"),
        "text": connection.get("text", ""),
        "subject_specifics": connection.get("subject_specifics", []),
        "predicate_specifics": connection.get("predicate_specifics", []),
        "connection_specifics": connection.get("connection_specifics", []),
    }


def _connection_from_payload(payload):
    payload = dict(payload or {})
    relation = _clean_part(payload.get("connection", ""))
    if not relation:
        return None
    subject = _endpoint_from_payload(payload.get("subject", {}))
    predicate = _endpoint_from_payload(payload.get("predicate", {}))
    if not subject.asu_value() or not predicate.asu_value():
        return None
    return {
        "subject": subject,
        "predicate": predicate,
        "connection": get_literal_index(relation),
        "source": payload.get("source", "unknown"),
        "text": payload.get("text", ""),
        "subject_specifics": payload.get("subject_specifics", []),
        "predicate_specifics": payload.get("predicate_specifics", []),
        "connection_specifics": payload.get("connection_specifics", []),
    }


def _source_for_block(block):
    raw_tag = clean_source_text((block or {}).get("tag", ""))
    url = _clean_url((block or {}).get("url", ""))
    if "|" in raw_tag:
        title, embedded_url = raw_tag.rsplit("|", 1)
        tag = _clean_part(title)
        url = url or _clean_url(embedded_url)
    else:
        tag = _clean_part(raw_tag)
    if tag and url:
        return f"{tag}|{url}"
    return url or tag or "unknown"


def _clean_url(url):
    url = str(url or "").strip().strip("\"'<>[]()")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def _split_sentences(text):
    pieces = []
    for sentence in SENTENCE_RE.split(str(text or "")):
        clean = clean_clause_text(sentence)
        if clean:
            pieces.append(clean.rstrip(".!?"))
        if len(pieces) >= EXTRACTION_SENTENCE_LIMIT:
            break
    return pieces


def _split_clauses(sentence):
    clauses = []
    for clause in CLAUSE_SPLIT_RE.split(sentence):
        clean = clean_clause_text(clause)
        if not clean:
            continue
        clauses.append(clean.rstrip(".!?"))
        if len(clauses) >= EXTRACTION_CLAUSE_LIMIT:
            break
    return clauses


def _truth_for_clause(clause):
    lower = f" {clause.lower()} "
    return 0 if re.search(r"\b(no|not|never|without|failed to|fails to)\b", lower) else 1


def _tense_for_relation(relation, clause):
    return get_literal_index(parsed_tense(clause, relation))


def _quantifier_for_subject(subject):
    lower = str(subject or "").lower()
    for quantifier in ("all", "some", "many", "most", "several", "few", "one"):
        if re.search(rf"\b{quantifier}\b", lower):
            return get_literal_index(quantifier)
    return -1


def _modifier_ids(text):
    modifiers = []
    for value in DATE_RE.findall(text):
        idx = ConnectionEndpoint.register_modifier(value)
        if idx >= 0 and idx not in modifiers:
            modifiers.append(idx)
    for value in NUMBER_RE.findall(text):
        if re.fullmatch(r"(?:18|19|20|21)\d{2}", _clean_part(value)):
            continue
        idx = ConnectionEndpoint.register_modifier(value)
        if idx >= 0 and idx not in modifiers:
            modifiers.append(idx)
    return modifiers[:4]


def _add_modifier(modifiers, value):
    clean = _clean_part(value)
    clean = re.sub(
        rf"^{MODIFIER_CUE_RE.pattern}\s+(?={MODIFIER_CUE_RE.pattern}\b)",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(rf"\s+{MODIFIER_CUE_RE.pattern}$", "", clean, flags=re.IGNORECASE)
    if not clean:
        return
    idx = ConnectionEndpoint.register_modifier(clean)
    if idx >= 0 and idx not in modifiers:
        modifiers.append(idx)


def _drop_specific_literals(text):
    text = DATE_RE.sub(" ", text)
    text = NUMBER_RE.sub(" ", text)
    return _clean_part(text)


def _strip_embedded_statement_clause(text, modifiers):
    statement = embedded_statement_text(text)
    if not statement:
        return text
    prefix = text[: max(0, str(text).find(statement))].strip()
    if prefix:
        _add_modifier(modifiers, prefix)
    return statement


def _shorten_core(text):
    text = _clean_part(text)
    text = SUBJECT_PREFIXES.sub("", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:all|some|many|most|several|few|one)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\b(?:do|does|did|can|could|may|might|must|should|would|will)\s+(?:not\s*)?$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:that|which|who)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"^{MODIFIER_CUE_RE.pattern}\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"\s+{MODIFIER_CUE_RE.pattern}$", "", text, flags=re.IGNORECASE)
    tokens = TOKEN_RE.findall(text)
    if len(tokens) > MAX_AGENT_TOKENS:
        text = " ".join(tokens[-MAX_AGENT_TOKENS:])
    return _clean_part(text)


def _endpoint_payload(text, source, fallback_core=""):
    original = _clean_part(text)
    modifiers = _modifier_ids(original)
    specifics = _specifics(original, source)
    core = _drop_specific_literals(original)

    match = LEADING_MODIFIER_RE.match(core)
    if match:
        _add_modifier(modifiers, match.group("modifier"))
        core = _clean_part(match.group("core"))

    core = _strip_embedded_statement_clause(core, modifiers)

    match = TRAILING_MODIFIER_RE.match(core)
    if match:
        _add_modifier(modifiers, match.group("modifier"))
        core = _clean_part(match.group("core"))

    core = _shorten_core(core) or _shorten_core(fallback_core) or _shorten_core(original)
    return {
        "core": core,
        "modifier_ids": modifiers[:4],
        "specifics": specifics,
    }


def _specifics(text, source):
    specifics = []
    seen = set()
    for match in DATE_RE.findall(text):
        clean = _clean_part(match)
        key = ("date", clean.lower())
        if clean and key not in seen:
            seen.add(key)
            specifics.append({"kind": "date", "text": clean, "source": source})
    for match in NUMBER_RE.findall(text):
        clean = _clean_part(match)
        if re.fullmatch(r"(?:18|19|20|21)\d{2}", clean):
            continue
        key = ("number", clean.lower())
        if clean and key not in seen:
            seen.add(key)
            specifics.append({"kind": "amount", "text": clean, "source": source})
    return specifics


def _trim_endpoint(text):
    return _endpoint_payload(text, "", fallback_core=text)["core"]


def _is_amount_tail(predicate_raw, relation):
    clean = _clean_part(predicate_raw)
    if not clean:
        return False
    amount_match = NUMBER_RE.search(clean)
    if amount_match and amount_match.start() <= 16:
        return True
    leading_modifier_amount = re.match(
        rf"^{MODIFIER_CUE_RE.pattern}\s+(?:{NUMBER_RE.pattern}|{DATE_RE.pattern})",
        clean,
        flags=re.IGNORECASE,
    )
    if leading_modifier_amount:
        return True
    if relation in {"is", "are", "was", "were", "become", "becomes", "became", "has", "have", "had"}:
        return bool(re.match(rf"^(?:{NUMBER_RE.pattern}|{DATE_RE.pattern})", clean, flags=re.IGNORECASE))
    return False


def _looks_like_action_verb(token):
    token = token.lower()
    if token in RELATION_STOPWORDS:
        return False
    return looks_like_relation_word(token)


def _candidate(subject_raw, relation, predicate_raw, clause):
    subject = _trim_endpoint(subject_raw)
    if _is_amount_tail(predicate_raw, relation):
        predicate = f"{relation} amount"
    else:
        predicate = _trim_endpoint(predicate_raw)
    if NUMBER_RE.match(predicate):
        predicate = f"{relation} amount"
    if not is_usable_agent_text(subject) or not is_usable_agent_text(predicate):
        return None
    if subject.lower() == predicate.lower():
        return None
    return {
        "subject": subject,
        "subject_raw": _clean_part(subject_raw),
        "relation": _clean_part(relation).lower(),
        "predicate": predicate,
        "predicate_raw": _clean_part(predicate_raw),
        "clause": clean_clause_text(clause),
    }


def _candidate_from_action_clause(clean):
    matches = list(TOKEN_RE.finditer(clean))
    if len(matches) < 3:
        return None
    statement = embedded_statement_text(clean)
    if statement and statement != clean:
        nested = _candidate_from_clause(statement)
        if nested is not None:
            return nested
    dependency_candidate = dependency_relation(clean)
    if dependency_candidate is not None:
        candidate = _candidate(
            dependency_candidate["subject_raw"],
            dependency_candidate["relation"],
            dependency_candidate["predicate_raw"],
            clean,
        )
        if candidate is not None:
            return candidate
    action_matches = [
        (idx, match)
        for idx, match in enumerate(matches[1:-1], 1)
        if _looks_like_action_verb(match.group(0))
    ]
    last_break = max(clean.rfind(","), clean.rfind(";"))
    action_matches.sort(key=lambda item: (item[1].start() <= last_break, item[1].start()))
    for idx, match in action_matches:
        relation = match.group(0).lower()
        if not _looks_like_action_verb(relation):
            continue
        predicate_raw = clean[matches[idx].end():]
        candidate = _candidate(clean[:matches[idx].start()], relation, predicate_raw, clean)
        if candidate is None:
            continue
        return candidate
    return None


def _candidate_from_clause(clause):
    clean = clean_clause_text(clause).rstrip(".!?")
    if len(TOKEN_RE.findall(clean)) < 3:
        return None

    statement = embedded_statement_text(clean)
    if statement and statement != clean:
        nested = _candidate_from_clause(statement)
        if nested is not None:
            return nested

    dependency_candidate = dependency_relation(clean)
    if dependency_candidate is not None:
        candidate = _candidate(
            dependency_candidate["subject_raw"],
            dependency_candidate["relation"],
            dependency_candidate["predicate_raw"],
            clean,
        )
        if candidate is not None:
            return candidate

    match = LINKING_PATTERN.match(clean)
    if not match:
        return _candidate_from_action_clause(clean)

    relation = _clean_part(match.group("relation")).lower()
    if len(relation) < 2:
        return None
    return _candidate(match.group("subject"), relation, match.group("predicate"), clean)


def _connection_from_candidate(candidate, source):
    relation = candidate["relation"]
    clause = candidate["clause"]
    truth = _truth_for_clause(clause)
    tense = _tense_for_relation(relation, clause)
    subject_payload = _endpoint_payload(
        candidate.get("subject_raw") or candidate["subject"],
        source,
        fallback_core=candidate["subject"],
    )
    predicate_payload = _endpoint_payload(
        candidate.get("predicate_raw") or candidate["predicate"],
        source,
        fallback_core=candidate["predicate"],
    )
    if _is_amount_tail(candidate.get("predicate_raw") or "", relation):
        predicate_payload["core"] = candidate["predicate"]
    subject = subject_payload["core"]
    predicate = predicate_payload["core"]
    subject_specifics = subject_payload["specifics"]
    predicate_specifics = predicate_payload["specifics"]
    connection_specifics = _specifics(clause, source)
    return {
        "subject": ConnectionEndpoint(
            quantifier=_quantifier_for_subject(candidate.get("subject_raw") or subject),
            tense=tense,
            truth=truth,
            ASU_idx=subject,
            modifier_idx=subject_payload["modifier_ids"],
        ),
        "predicate": ConnectionEndpoint(
            quantifier=_quantifier_for_subject(candidate.get("predicate_raw") or predicate),
            tense=tense,
            truth=truth,
            ASU_idx=predicate,
            modifier_idx=predicate_payload["modifier_ids"],
        ),
        "connection": get_literal_index(relation),
        "source": source,
        "text": clause,
        "subject_specifics": subject_specifics,
        "predicate_specifics": predicate_specifics,
        "connection_specifics": connection_specifics,
    }


def _block_fingerprint(block):
    content = clean_text((block or {}).get("content", ""))
    url = _clean_url((block or {}).get("url", ""))
    source = _source_for_block(block)
    if url:
        return ("url", url, hashlib.sha1(content.encode("utf-8")).hexdigest())
    return ("content", hashlib.sha1(f"{source}\0{content}".encode("utf-8")).hexdigest())


def _unique_blocks(blocks):
    unique = []
    seen = set()
    for block in list(blocks or []):
        content = clean_text((block or {}).get("content", ""))
        if not content:
            continue
        fingerprint = _block_fingerprint(block)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(block)
    return unique


def _connections_from_block(block, query=""):
    content = clean_text((block or {}).get("content", ""))
    if not content:
        return []
    source = _source_for_block(block) or query or "unknown"
    cache_key = _block_cache_key(content, source)
    cached = _cache_get(cache_key)
    if isinstance(cached, list):
        hydrated = []
        for item in cached:
            connection = _connection_from_payload(item)
            if connection is not None:
                hydrated.append(connection)
        return hydrated

    candidates = []
    seen = set()
    clauses = []
    for sentence in _split_sentences(content):
        clauses.extend(_split_clauses(sentence))
        if len(clauses) >= EXTRACTION_CLAUSE_LIMIT:
            clauses = clauses[:EXTRACTION_CLAUSE_LIMIT]
            break
    clauses = [clause for clause in clauses if is_usable_clause_text(clause)]
    preparse_texts(clauses)

    for clause in clauses:
        candidate = _candidate_from_clause(clause)
        if candidate is None:
            continue
        key = (
            candidate["subject"].lower(),
            candidate["relation"].lower(),
            candidate["predicate"].lower(),
            source,
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    vector_warmup = []
    for candidate in candidates:
        vector_warmup.append(candidate.get("subject", ""))
        vector_warmup.append(candidate.get("predicate", ""))
    if vector_warmup:
        precache_text_vectors(vector_warmup)

    block_results = [
        _connection_from_candidate(candidate, source)
        for candidate in candidates
    ]

    _cache_set(cache_key, [_serialize_connection(connection) for connection in block_results])
    return block_results


def find_connections(blocks, query="", connection_limit=None):
    limit = int(connection_limit or EXTRACTION_CLAUSE_LIMIT)
    results = []
    seen = set()
    for block in _unique_blocks(blocks):
        for connection in _connections_from_block(block, query=query):
            subject_sp = connection.get("subject")
            predicate_sp = connection.get("predicate")
            key = (
                subject_sp.asu_value().lower() if isinstance(subject_sp, ConnectionEndpoint) else "",
                str(connection.get("connection", "")).lower(),
                predicate_sp.asu_value().lower() if isinstance(predicate_sp, ConnectionEndpoint) else "",
                connection.get("source", ""),
                connection.get("text", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(connection)
            if len(results) >= limit:
                return results
    return results
