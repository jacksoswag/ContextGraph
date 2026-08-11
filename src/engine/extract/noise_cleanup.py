import copy, re
from engine.common.language import (AMOUNT_ENDPOINT_PHRASES,AMOUNT_UNIT_PATTERNS,AUXILIARY_TOKENS,CONTRACTION_FRAGMENT_TOKENS,CONTRACTION_PRONOUN_BASES,CONTRACTION_SUFFIXES,COORDINATOR_TOKENS,DISCOURSE_MARKER_TOKENS,GENERIC_REFERENTS,LEADING_DISCOURSE_MARKERS,LEADING_FILLER_TOKENS,LINKING_RELATION_TOKENS,NAVIGATION_TRAILING_PHRASES,PAREN_SOURCE_LABEL_PATTERNS,STOPWORD_TOKENS,SUBORDINATE_CLAUSE_TOKENS,TEMPORAL_ONLY_TOKENS,TITLE_SECTION_NUMBER_WORDS,TITLE_SECTION_WORDS,TRAILING_DISCOURSE_MARKERS,WEAK_AGENT_TOKENS,WHOLE_PHRASE_REJECT_STARTS)
from engine.extract.linguistic_roles import (looks_like_clause, looks_like_imperative); from engine.agents.connection import ConnectionEndpoint

# Builds phrase pattern for text cleanup.
def _phrase_pattern(values): return "|".join(re.escape(value).replace(r"\ ", r"\s+") for value in sorted(values, key=len, reverse=True))
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE); HTML_RE = re.compile(r"<[^>]+>"); BRACKET_CITATION_RE = re.compile(r"\[(?:\d+|citation needed|edit|note \d+)\]", re.IGNORECASE); PAREN_SOURCE_RE = re.compile(rf"\((?:{'|'.join(PAREN_SOURCE_LABEL_PATTERNS)})\)", re.IGNORECASE)
NAVIGATION_TRAILING_RE = re.compile(rf"[\s.·|]*(?:{_phrase_pattern(NAVIGATION_TRAILING_PHRASES)})\b.*$",re.IGNORECASE); TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?"); CONTRACTION_PRONOUN_RE = re.compile(rf"\b(?:{_phrase_pattern(CONTRACTION_PRONOUN_BASES)})'(?:{_phrase_pattern(CONTRACTION_SUFFIXES)})\b",re.IGNORECASE)
COORDINATOR_RE = re.compile(rf"\s*(?:;|,\s*(?:{_phrase_pattern(COORDINATOR_TOKENS)})\s+|,\s*|\s+\b(?:{_phrase_pattern(COORDINATOR_TOKENS)})\b\s+)\s*", re.IGNORECASE); SPECIFIC_NUMBER_RE = re.compile(rf"(?:\$|€|£)?\b\d[\d,]*(?:\.\d+)?(?:\s?(?:{_phrase_pattern(AMOUNT_UNIT_PATTERNS)}))?(?=\W|$)",re.IGNORECASE,)
LEADING_FILLER_RE = re.compile(rf"^(?:{_phrase_pattern(LEADING_FILLER_TOKENS)})\s+", re.IGNORECASE); LEADING_DISCOURSE_RE = re.compile(rf"^(?:{_phrase_pattern(LEADING_DISCOURSE_MARKERS)})\b[\s,:.-]*", re.IGNORECASE,); TRAILING_DISCOURSE_RE = re.compile(rf"\s+(?:{_phrase_pattern(TRAILING_DISCOURSE_MARKERS)})\b[\s,:.-]*$",re.IGNORECASE)
TITLE_SECTION_RE = re.compile(rf"^(?:{_phrase_pattern(TITLE_SECTION_WORDS)})\s+(?:\d+|{_phrase_pattern(TITLE_SECTION_NUMBER_WORDS)})\b[\s:.-]*",re.IGNORECASE,); MAX_AGENT_CHARS = 72; MAX_AGENT_TOKENS = 8; MAX_CLAUSE_CHARS = 360; MAX_COORDINATED_PARTS = 4; MAX_CONNECTION_EXPANSIONS = 8

# Normalizes text for text cleanup.
def _normalize_text(text, strip_urls=True):
    text = str(text or ""); text = text.replace("\u2018", "'").replace("\u2019", "'"); text = text.replace("\u201c", '"').replace("\u201d", '"'); text = text.replace("\u2013", "-").replace("\u2014", "-"); text = HTML_RE.sub(" ", text)
    if strip_urls: text = URL_RE.sub(" ", text)
    text = BRACKET_CITATION_RE.sub(" ", text); text = PAREN_SOURCE_RE.sub(" ", text); text = NAVIGATION_TRAILING_RE.sub(" ", text); text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text); text = re.sub(r"[_*`~]+", " ", text); text = re.sub(r"\s+", " ", text)
    return text

# Cleans text for text cleanup.
def clean_text(text):
    text = _normalize_text(text, strip_urls=True); text = re.sub(r"\s*-\s*", " ", text)
    return text.strip(" \t\r\n\"'.,:|")

# Cleans source text for text cleanup.
def clean_source_text(text):
    text = _normalize_text(text, strip_urls=False)
    return text.strip(" \t\r\n\"'.,:")

# Returns agent tokens for text cleanup.
def _agent_tokens(text):
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]

# Returns whether the input has contraction pronoun.
def has_contraction_pronoun(text):
    return bool(CONTRACTION_PRONOUN_RE.search(str(text or "")))

# Returns content tokens for text cleanup.
def _content_tokens(text):
    return [token for token in _agent_tokens(text) if token not in STOPWORD_TOKENS and token not in DISCOURSE_MARKER_TOKENS and token not in WEAK_AGENT_TOKENS]

# Cleans strip leading noise for text cleanup.
def _strip_leading_noise(text):
    clean = text
    while clean:
        previous = clean; clean = LEADING_FILLER_RE.sub("", clean).strip(); clean = LEADING_DISCOURSE_RE.sub("", clean).strip(); clean = TITLE_SECTION_RE.sub("", clean).strip()
        if clean == previous: break
    return clean

# Cleans collapse repeated title for text cleanup.
def _collapse_repeated_title(text):
    clean = text.strip()
    if ":" not in clean: return clean
    parts = [part.strip(" \t\r\n\"'.,:") for part in clean.split(":") if part.strip(" \t\r\n\"'.,:")]
    if len(parts) < 2: return clean
    best = clean; best_tokens = _content_tokens(best)
    for part in parts:
        candidate = _strip_leading_noise(part); candidate_tokens = _content_tokens(candidate)
        if not candidate_tokens: continue
        if best_tokens and set(candidate_tokens) <= set(best_tokens) and len(candidate) < len(best):
            best = candidate; best_tokens = candidate_tokens
    return best

# Cleans collapse repeated word runs for text cleanup.
def _collapse_repeated_word_runs(text):
    words = str(text or "").split()
    if len(words) < 2: return str(text or "").strip()
    
# Builds a sortable key for repeated title collapse candidates.
    def key(word): return re.sub(r"[^a-z0-9]+", "", word.lower())
    changed = True
    while changed:
        changed = False; max_run = min(4, len(words) // 2)
        for size in range(max_run, 0, -1):
            idx = 0; next_words = []
            while idx < len(words):
                left = [key(word) for word in words[idx : idx + size]]; right = [key(word) for word in words[idx + size : idx + (2 * size)]]
                if size <= len(left) == len(right) and left and left == right:
                    next_words.extend(words[idx : idx + size]); idx += 2 * size; changed = True
                else:
                    next_words.append(words[idx]); idx += 1
            words = next_words
            if changed: break
    return " ".join(words).strip()

# Cleans agent name for text cleanup.
def clean_agent_name(text):
    clean = clean_text(text); clean = _collapse_repeated_title(clean); clean = _strip_leading_noise(clean); clean = LEADING_FILLER_RE.sub("", clean).strip(); clean = TRAILING_DISCOURSE_RE.sub("", clean).strip(); clean = re.sub(r"\s*[\[(][^\])]*$", "", clean).strip(); clean = _collapse_repeated_word_runs(clean); clean = re.sub(r"\s+", " ", clean)
    return clean[:MAX_AGENT_CHARS].strip()

# Returns meaningful tokens for text cleanup.
def _meaningful_tokens(text):
    return [token.lower() for token in TOKEN_RE.findall(str(text or "")) if token and token.lower() not in STOPWORD_TOKENS]

# Returns whether the input is usable agent text.
def is_usable_agent_text(text):
    original_tokens = _agent_tokens(text)
    if original_tokens and original_tokens[0] in WHOLE_PHRASE_REJECT_STARTS: return False
    clean = clean_agent_name(text)
    if not clean: return False
    if has_contraction_pronoun(clean): return False
    if is_clause_like_agent_text(clean): return False
    lowered = clean.lower()
    if URL_RE.search(clean) or "{" in clean or "}" in clean or "|" in clean: return False
    if "#" in clean: return False
    if lowered in GENERIC_REFERENTS: return False
    raw_tokens = _agent_tokens(clean)
    if raw_tokens and raw_tokens[0] in GENERIC_REFERENTS: return False
    if raw_tokens and raw_tokens[0] in AUXILIARY_TOKENS and lowered not in AMOUNT_ENDPOINT_PHRASES: return False
    if raw_tokens and set(raw_tokens) <= TEMPORAL_ONLY_TOKENS: return False
    if looks_like_imperative(clean): return False
    tokens = _meaningful_tokens(clean)
    if not tokens: return False
    content_tokens = _content_tokens(clean)
    if not content_tokens and clean.lower() not in AMOUNT_ENDPOINT_PHRASES: return False
    if len(tokens) == 1 and tokens[0] in DISCOURSE_MARKER_TOKENS | WEAK_AGENT_TOKENS: return False
    if tokens[0] in DISCOURSE_MARKER_TOKENS: return False
    if len(tokens) > MAX_AGENT_TOKENS: return False
    if len(tokens) == 1 and (len(tokens[0]) < 3 or tokens[0].isdigit()): return False
    alpha_chars = sum(1 for char in clean if char.isalpha())
    if alpha_chars < 2: return False
    non_word_chars = sum(1 for char in clean if not (char.isalnum() or char.isspace() or char in "'&./"))
    return non_word_chars <= max(2, len(clean) // 8)

# Returns whether the input is usable relation text.
def is_usable_relation_text(text):
    clean = clean_text(text).lower()
    if not clean: return False
    if has_contraction_pronoun(clean): return False
    tokens = _agent_tokens(clean)
    if not tokens: return False
    if len(tokens) == 1 and tokens[0] in CONTRACTION_FRAGMENT_TOKENS: return False
    if tokens[0] in DISCOURSE_MARKER_TOKENS or tokens[-1] in DISCOURSE_MARKER_TOKENS: return False
    if tokens[0] in GENERIC_REFERENTS or tokens[-1] in GENERIC_REFERENTS: return False
    if any(token in CONTRACTION_FRAGMENT_TOKENS for token in tokens) and not set(tokens) & LINKING_RELATION_TOKENS: return False
    return True

# Returns whether the input is clause like agent text.
def is_clause_like_agent_text(text):
    clean = clean_agent_name(text); tokens = _agent_tokens(clean)
    if len(tokens) < 3: return False
    if looks_like_clause(clean): return True
    token_set = set(tokens)
    if len(tokens) >= 4 and token_set & AUXILIARY_TOKENS: return True
    if token_set & SUBORDINATE_CLAUSE_TOKENS and token_set & AUXILIARY_TOKENS: return True
    if tokens[0] in WEAK_AGENT_TOKENS and token_set & AUXILIARY_TOKENS: return True
    return False

# Returns whether text looks like title or list noise.
def _looks_like_title_or_list_noise(text):
    clean = clean_text(text); tokens = TOKEN_RE.findall(clean)
    if len(tokens) < 12: return False
    number_count = sum(1 for token in tokens if any(char.isdigit() for char in token)); alpha_tokens = [token for token in tokens if any(char.isalpha() for char in token)]; titleish_count = sum(1 for token in alpha_tokens if token[:1].isupper() and token.lower() not in STOPWORD_TOKENS); lower_tokens = {token.lower() for token in tokens}
    has_sentence_verb = bool(lower_tokens & AUXILIARY_TOKENS); has_clause_marker = bool(lower_tokens & SUBORDINATE_CLAUSE_TOKENS); punctuation_count = sum(1 for char in str(text or "") if char in ".!?;:,")
    if number_count >= 4 and titleish_count >= max(5, len(alpha_tokens) // 3) and not has_sentence_verb: return True
    if number_count >= 6 and punctuation_count <= 2 and not has_clause_marker: return True
    return False

# Returns whether the input is usable clause text.
def is_usable_clause_text(text):
    clean = clean_clause_text(text)
    if not clean: return False
    if has_contraction_pronoun(clean): return False
    tokens = TOKEN_RE.findall(clean)
    if len(tokens) < 2: return False
    if _looks_like_title_or_list_noise(clean): return False
    content_tokens = _content_tokens(clean)
    if not content_tokens: return False
    return True

# Cleans clause text for text cleanup.
def clean_clause_text(text, subject="", predicate=""):
    clean = clean_text(text)
    if not clean and subject and predicate: clean = f"{subject} {predicate}"
    if not clean: return ""
    if len(clean) > MAX_CLAUSE_CHARS:
        sentence_end = re.search(r"^(.{80,%d}?[.!?])\s" % MAX_CLAUSE_CHARS, clean); clean = sentence_end.group(1) if sentence_end else clean[:MAX_CLAUSE_CHARS]
    clean = clean.strip(" ,:")
    if not clean: return ""
    if clean[-1] not in ".!?": clean += "."
    return clean

# Splits coordinated phrase into usable parts.
def split_coordinated_phrase(text):
    clean = clean_agent_name(text)
    if not clean: return []
    parts = [clean_agent_name(part) for part in COORDINATOR_RE.split(clean) if clean_agent_name(part)]; usable = []; seen = set()
    for part in parts:
        key = part.lower()
        if key in seen or not is_usable_agent_text(part): continue
        seen.add(key); usable.append(part)
        if len(usable) >= MAX_COORDINATED_PARTS: break
    return usable or ([clean] if is_usable_agent_text(clean) else [])

# Cleans specific value for text cleanup.
def _clean_specific_value(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if isinstance(item, str):
                text = clean_text(item)
                if text: cleaned[key] = text
            else:
                cleaned[key] = item
        text_fields = [str(cleaned.get(key, "")).strip() for key in ("text", "surface", "normalized", "context", "cue")]
        if any(text_fields) or any(SPECIFIC_NUMBER_RE.search(text) for text in text_fields): return cleaned
        return None
    clean = clean_text(value)
    if not clean: return None
    if has_contraction_pronoun(clean): return None
    if len(clean) > MAX_CLAUSE_CHARS: clean = clean[:MAX_CLAUSE_CHARS].strip()
    return clean

# Cleans specifics for text cleanup.
def clean_specifics(values):
    cleaned = []; seen = set()
    for value in list(values or []):
        item = _clean_specific_value(value)
        if item is None: continue
        key = repr(sorted(item.items())) if isinstance(item, dict) else item.lower()
        if key in seen: continue
        seen.add(key); cleaned.append(item)
    return cleaned

# Cleans endpoint for text cleanup.
def _clean_endpoint(endpoint, text):
    return ConnectionEndpoint(quantifier=getattr(endpoint, "quantifier", -1), tense=getattr(endpoint, "tense", -1), truth=getattr(endpoint, "truth", -1), ASU_idx=text, modifier_idx=getattr(endpoint, "modifier_idx", ()))

# Builds expand clean connection record for text cleanup.
def expand_clean_connection_record(record):
    subject_sp = (record or {}).get("subject"); predicate_sp = (record or {}).get("predicate"); relation_id = (record or {}).get("connection")
    if not isinstance(subject_sp, ConnectionEndpoint) or not isinstance(predicate_sp, ConnectionEndpoint): return []
    if relation_id is None: return []
    subjects = split_coordinated_phrase(subject_sp.asu_value()); predicates = split_coordinated_phrase(predicate_sp.asu_value())
    if not subjects or not predicates: return []
    source = clean_source_text((record or {}).get("source", "")) or "unknown"; evidence = clean_clause_text((record or {}).get("text", "")); expansions = []
    for subject in subjects:
        for predicate in predicates:
            if len(expansions) >= MAX_CONNECTION_EXPANSIONS: return expansions
            if subject.lower() == predicate.lower(): continue
            cleaned = copy.copy(record); cleaned["subject"] = _clean_endpoint(subject_sp, subject); cleaned["predicate"] = _clean_endpoint(predicate_sp, predicate); cleaned["source"] = source; cleaned["text"] = clean_clause_text(evidence, subject=subject, predicate=predicate)
            cleaned["subject_specifics"] = clean_specifics((record or {}).get("subject_specifics")); cleaned["predicate_specifics"] = clean_specifics((record or {}).get("predicate_specifics")); cleaned["connection_specifics"] = clean_specifics((record or {}).get("connection_specifics")); expansions.append(cleaned)
    return expansions
