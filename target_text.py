import re

from constants import STOPWORD_TOKENS

TARGET_TOKEN_RE = re.compile(r"[a-z0-9]+")
MIN_STEM_LENGTH = 4
DERIVATIONAL_SUFFIXES = (
    "ability",
    "ibility",
    "able",
    "ible",
    "ness",
)
INFLECTIONAL_SUFFIXES = (
    "ing",
    "ed",
)


def _valid_stem(stem):
    return (
        len(stem) >= MIN_STEM_LENGTH
        and re.search(r"[aeiouy]", stem)
        and re.search(r"[bcdfghjklmnpqrstvwxyz]", stem)
    )


def _valid_inflection_stem(token, suffix, stem):
    if not _valid_stem(stem):
        return False
    if suffix == "ing" and not re.search(r"(.)\1ing$|[wlkmnprt]ing$", token):
        return False
    return True


def target_token_key(token):
    token = str(token or "").strip().lower()
    if not token:
        return ""
    for suffix in DERIVATIONAL_SUFFIXES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            if _valid_stem(stem):
                return stem
    for suffix in INFLECTIONAL_SUFFIXES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            if _valid_inflection_stem(token, suffix, stem):
                if suffix == "ing" and len(stem) >= 2 and stem[-1] == stem[-2]:
                    stem = stem[:-1]
                return stem
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def target_tokens(text, min_length=3):
    tokens = TARGET_TOKEN_RE.findall(str(text or "").lower())
    normalized = []
    seen = set()
    for token in tokens:
        if len(token) < int(min_length) or token in STOPWORD_TOKENS:
            continue
        key = target_token_key(token)
        if not key or key in STOPWORD_TOKENS or len(key) < int(min_length) or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def distinctive_target_tokens(tokens):
    tokens = set(tokens or [])
    if len(tokens) <= 1:
        return tokens
    max_len = max(len(token) for token in tokens)
    return {token for token in tokens if len(token) == max_len}
