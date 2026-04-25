import re
from constants import (
    BRIDGE_QUERY_COUNT,
    BRIDGE_QUERY_LIMIT,
    MAX_EXPANDED_QUERY_TOKENS,
    QUERY_EXPANSION_MODEL,
    STOPWORD_TOKENS,
    TARGET_FOCUS_PHRASE_COUNT,
    TARGET_FOCUS_PHRASE_LIMIT,
    TARGET_QUERY_EVIDENCE_COUNT,
    TARGET_QUERY_LIMIT,
    TARGET_QUERY_SHORT_COUNT,
)

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover - optional local dependency
    ollama = None

TARGET_QUERY_PROMPTS = (
    (
        f"Generate {TARGET_QUERY_SHORT_COUNT} terse search keyword phrases for the target. "
        f"Each line must be 2-{MAX_EXPANDED_QUERY_TOKENS} words. No questions. No sentences. "
        "No intro, numbering, bullets, punctuation, or filler."
    ),
    (
        f"Generate {TARGET_QUERY_EVIDENCE_COUNT} terse evidence keyword phrases for the target. "
        f"Each line must be 2-{MAX_EXPANDED_QUERY_TOKENS} words. No questions. No sentences. "
        "No intro, numbering, bullets, punctuation, or filler."
    ),
)
BRIDGE_QUERY_PROMPT = (
    f"Generate {BRIDGE_QUERY_COUNT} terse bridge keyword phrases connecting the two targets. "
    f"Each line must be 2-{MAX_EXPANDED_QUERY_TOKENS} words. No questions. No sentences. "
    "No intro, numbering, bullets, punctuation, or filler."
)
TARGET_FOCUS_PROMPT = (
    f"You are a researcher. Given two research targets, generate exactly {TARGET_FOCUS_PHRASE_COUNT} short aspect phrases "
    "for the requested focus target only. Each phrase must be 1-2 words, must cover a distinct "
    "aspect of that target, and must not overlap with the other phrases. The phrases must stay tied "
    "to the focus target itself and must not drift into the other target. RULES: no intro, no numbering, "
    "no bullets, no punctuation-heavy phrases, just the phrases."
)
FOCUS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
QUESTION_START_RE = re.compile(
    r"^(what|how|why|when|where|who|which|is|are|does|do|did|can|could|would|should)\b\s*",
    re.IGNORECASE,
)
QUERY_GLUE_TOKENS = {
    "a", "an", "and", "are", "between", "can", "could", "did", "do", "does",
    "how", "in", "is", "of", "on", "or", "role", "shape", "the", "there",
    "to", "what", "when", "where", "which", "who", "why", "would",
}

def _normalize_query_text(text):
    return " ".join(str(text or "").strip().split())

def _clean_generated_line(line):
    clean = re.sub(r"^[0-9]+[\.\-\)\s]+", "", str(line or "").strip())
    clean = re.sub(r"^[\-\*\•\s]+", "", clean)
    clean = _normalize_query_text(clean)
    lower = clean.lower()
    if not clean or ":" in clean:
        return ""
    if "here are" in lower or "research queries" in lower:
        return ""
    clean = QUESTION_START_RE.sub("", clean).strip(" ?.!,:;")
    tokens = [
        token
        for token in QUERY_TOKEN_RE.findall(clean)
        if token.lower() not in QUERY_GLUE_TOKENS
    ]
    if len(tokens) < 2:
        return ""
    return " ".join(tokens[:MAX_EXPANDED_QUERY_TOKENS])

def _dedupe_queries(queries, limit=None):
    ordered = []
    seen = set()
    for query in queries:
        clean = _normalize_query_text(query)
        normalized = clean.lower()
        if not clean or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(clean)
        if limit is not None and len(ordered) >= limit:
            break
    return ordered

def _ollama_queries(prompts, limit):
    if ollama is None:
        return []

    expanded = []
    try:
        for prompt in prompts:
            response = ollama.generate(
                model=QUERY_EXPANSION_MODEL,
                prompt=prompt,
                options={"temperature": 0.0},
            )
            raw = response.get("response", "")
            for line in raw.split("\n"):
                clean = _clean_generated_line(line)
                if clean:
                    expanded.append(clean)
    except Exception as e:
        print(f"LLM Error: {e}")

    return _dedupe_queries(expanded, limit=limit)

def _lexical_focus_score(phrase, target):
    phrase_tokens = {
        token.lower()
        for token in FOCUS_TOKEN_RE.findall(_normalize_query_text(phrase))
        if token.lower() not in STOPWORD_TOKENS
    }
    target_tokens = {
        token.lower()
        for token in FOCUS_TOKEN_RE.findall(_normalize_query_text(target))
        if token.lower() not in STOPWORD_TOKENS
    }
    if not phrase_tokens or not target_tokens:
        return 0.0
    overlap = phrase_tokens & target_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(phrase_tokens)
    recall = len(overlap) / len(target_tokens)
    return max(precision, 0.85 * recall)


def _focus_anchor_score(phrase, target):
    phrase = _normalize_query_text(phrase)
    target = _normalize_query_text(target)
    if not phrase or not target:
        return 0.0
    return _lexical_focus_score(phrase, target)


def _filter_focus_phrases(phrases, focus_target, other_target, limit):
    kept = []
    seen = set()
    for phrase in list(phrases or []):
        clean = _normalize_query_text(phrase)
        lowered = clean.lower()
        if not clean or lowered in seen: continue
        focus_score = _focus_anchor_score(clean, focus_target)
        other_score = _focus_anchor_score(clean, other_target)
        if focus_score < 0.34: continue
        if other_target and (focus_score + 0.06) < other_score: continue
        seen.add(lowered)
        kept.append(clean)
        if len(kept) >= limit: break
    return kept

def _dedupe_focus_pair(left_phrases, right_phrases, limit):
    left = _dedupe_queries(left_phrases or [], limit=limit)
    left_seen = {phrase.lower() for phrase in left}

    right = []
    for phrase in list(right_phrases or []):
        clean = _normalize_query_text(phrase)
        lowered = clean.lower()
        if not clean or lowered in left_seen or lowered in {item.lower() for item in right}:
            continue
        right.append(clean)
        if len(right) >= limit:
            break

    return left[:limit], right[:limit]

def expand_target_queries(target, limit=TARGET_QUERY_LIMIT):
    topic = _normalize_query_text(target)
    print(f"Expanding target query set: '{topic}'")
    if not topic:
        return []

    prompts = [f"{prompt}\nTarget: {topic}\nQueries:" for prompt in TARGET_QUERY_PROMPTS]
    queries = _ollama_queries(prompts, limit=limit)
    if queries:
        return queries[:limit]

    print("LLM Error: ollama unavailable or empty output. No expanded target queries generated.")
    return []

def expand_bridge_queries(target_a, target_b, limit=BRIDGE_QUERY_LIMIT):
    target_a = _normalize_query_text(target_a)
    target_b = _normalize_query_text(target_b)
    print(f"Expanding bridge queries: '{target_a}' <-> '{target_b}'")
    if not target_a or not target_b:
        return []

    prompt = (
        f"{BRIDGE_QUERY_PROMPT}\n"
        f"Target A: {target_a}\n"
        f"Target B: {target_b}\n"
        "Queries:"
    )
    queries = _ollama_queries([prompt], limit=limit)
    if queries: return queries[:limit]

    print("LLM Error: ollama unavailable or empty output. No expanded bridge queries generated.")
    return []


def expand_target_focus_phrases(focus_target, other_target, limit=TARGET_FOCUS_PHRASE_LIMIT):
    focus_target = _normalize_query_text(focus_target)
    other_target = _normalize_query_text(other_target)
    print(f"Expanding focus phrases for '{focus_target}' against '{other_target}'")
    if not focus_target: return []

    prompt = (
        f"{TARGET_FOCUS_PROMPT}\n"
        f"Focus target: {focus_target}\n"
        f"Other target: {other_target}\n"
        "Phrases:"
    )
    phrases = _ollama_queries([prompt], limit=limit)
    cleaned = []
    seen = set()
    for phrase in phrases:
        normalized = _normalize_query_text(phrase)
        if not normalized: continue
        token_count = len(normalized.split())
        if token_count < 1 or token_count > 2: continue
        lowered = normalized.lower()
        if lowered in seen: continue
        seen.add(lowered)
        cleaned.append(normalized)
        if len(cleaned) >= limit: break
    cleaned = _filter_focus_phrases(cleaned, focus_target, other_target, limit)
    if cleaned: return cleaned

    print("LLM Error: ollama unavailable or empty output. No expanded focus phrases generated.")
    return []

def build_query_plan(target_a, target_b, per_target_limit=TARGET_QUERY_LIMIT, bridge_limit=BRIDGE_QUERY_LIMIT):
    target_a_queries = expand_target_queries(target_a, limit=per_target_limit)
    target_b_queries = expand_target_queries(target_b, limit=per_target_limit)
    bridge_queries = expand_bridge_queries(target_a, target_b, limit=bridge_limit)
    target_a_focus_phrases = expand_target_focus_phrases(target_a, target_b)
    target_b_focus_phrases = expand_target_focus_phrases(target_b, target_a)
    target_a_focus_phrases, target_b_focus_phrases = _dedupe_focus_pair(target_a_focus_phrases, target_b_focus_phrases, TARGET_FOCUS_PHRASE_LIMIT)
    all_queries = _dedupe_queries(target_a_queries + target_b_queries + bridge_queries)
    return {
        "target_a_queries": target_a_queries, "target_b_queries": target_b_queries, "bridge_queries": bridge_queries,
        "target_a_focus_phrases": target_a_focus_phrases, "target_b_focus_phrases": target_b_focus_phrases, "all_queries": all_queries,
    }
