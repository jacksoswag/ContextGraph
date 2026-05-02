import re; from u_language_constants import DERIVATIONAL_SUFFIXES, INFLECTIONAL_SUFFIXES, STOPWORD_TOKENS; TARGET_TOKEN_RE = re.compile(r"[a-z0-9]+"); SHORT_TARGET_TOKEN_LENGTH = 2; MIN_STEM_LENGTH = 4
# Returns whether a stem is long and word-like enough for target matching.
def _valid_stem(stem): return len(stem) >= MIN_STEM_LENGTH and re.search(r"[aeiouy]", stem) and re.search(r"[bcdfghjklmnpqrstvwxyz]", stem)
# Returns whether stripping an inflection leaves a plausible target stem.
def _valid_inflection_stem(token, suffix, stem):
    if not _valid_stem(stem): return False
    if suffix == "ing" and not re.search(r"(.)\1ing$|[wlkmnprt]ing$", token): return False
    return True
# Returns the normalized stem key for one target token.
def target_token_key(token):
    token = str(token or "").strip().lower()
    if not token: return ""
    for suffix in DERIVATIONAL_SUFFIXES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            if _valid_stem(stem): return stem
    for suffix in INFLECTIONAL_SUFFIXES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            if _valid_inflection_stem(token, suffix, stem):
                if suffix == "ing" and len(stem) >= 2 and stem[-1] == stem[-2]: stem = stem[:-1]
                return stem
    if token.endswith("ies") and len(token) > 4: return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4: return token[:-1]
    return token
# Returns deduped normalized tokens from target text.
def target_tokens(text, min_length=3):
    tokens = TARGET_TOKEN_RE.findall(str(text or "").lower()); normalized = []; seen = set(); min_length = int(min_length)
    for token in tokens:
        token_length = len(token)
        if token in STOPWORD_TOKENS: continue
        if token_length < min_length:
            if token_length != SHORT_TARGET_TOKEN_LENGTH: continue
        key = target_token_key(token)
        if not key or key in STOPWORD_TOKENS or key in seen: continue
        if len(key) < min_length and len(key) != SHORT_TARGET_TOKEN_LENGTH: continue
        seen.add(key); normalized.append(key)
    return normalized
# Returns acronym variants from multi-word target text.
def target_acronym_tokens(text, min_terms=2, max_terms=4):
    terms = [target_token_key(token) for token in TARGET_TOKEN_RE.findall(str(text or "").lower()) if len(token) >= 3 and token not in STOPWORD_TOKENS]; terms = [term for term in terms if term and term not in STOPWORD_TOKENS]; acronyms = []; seen = set()
    for size in range(max(2, int(min_terms)), max(2, int(max_terms)) + 1):
        if size > len(terms): break
        for index in range(0, len(terms) - size + 1):
            acronym = "".join(term[0] for term in terms[index:index + size] if term)
            if len(acronym) < 2 or acronym in STOPWORD_TOKENS or acronym in seen: continue
            seen.add(acronym); acronyms.append(acronym)
    return acronyms
# Returns the strongest target tokens used to require distinctive overlap.
def distinctive_target_tokens(tokens):
    tokens = {str(token or "").strip().lower() for token in tokens or [] if str(token or "").strip()}
    if len(tokens) <= 3: return tokens
    ranked = sorted(tokens, key=lambda token: (len(token), token), reverse=True)
    return set(ranked[:3])
