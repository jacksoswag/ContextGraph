import re

from constants import STOPWORD_TOKENS

TARGET_TOKEN_RE = re.compile(r"[a-z0-9]+")


def target_token_key(token):
    token = str(token or "").strip().lower()
    for suffix in (
        "ability",
        "ibility",
        "able",
        "ible",
        "ation",
        "tion",
        "ment",
        "ness",
        "ing",
        "ed",
    ):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def target_tokens(text, min_length=3):
    tokens = TARGET_TOKEN_RE.findall(str(text or "").lower())
    return [
        target_token_key(token)
        for token in tokens
        if len(token) >= int(min_length) and token not in STOPWORD_TOKENS
    ]


def distinctive_target_tokens(tokens):
    tokens = set(tokens or [])
    if len(tokens) <= 1:
        return tokens
    max_len = max(len(token) for token in tokens)
    return {token for token in tokens if len(token) == max_len}
