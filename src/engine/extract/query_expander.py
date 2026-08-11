import re; from engine.common.constants import (BRIDGE_QUERY_LIMIT, TARGET_FOCUS_PHRASE_LIMIT, TARGET_QUERY_LIMIT); from engine.common.language import STOPWORD_TOKENS; TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")

# Cleans text for query expansion.
def _clean_text(text): return " ".join(str(text or "").strip().split())

# Returns non-stopword query tokens used to build search variants.
def _tokens(text): return [token.lower() for token in TOKEN_RE.findall(str(text or "")) if len(token) > 2 and token.lower() not in STOPWORD_TOKENS]

# Deduplicates query strings while preserving order and respecting an optional limit.
def _dedupe(values, limit=None):
    ordered = []; seen = set()
    for value in list(values or []):
        clean = _clean_text(value); key = clean.lower()
        if not clean or key in seen: continue
        seen.add(key); ordered.append(clean)
        if limit is not None and len(ordered) >= int(limit): break
    return ordered

# Returns target queries for target matching.
def _target_queries(target):
    target = _clean_text(target)
    if not target: return []
    queries = [target,f"{target} facts",f"{target} dates amounts",f"{target} statistics figures",f"{target} history", f"{target} causes effects",f"{target} recent evidence",f"{target} policy research"]
    return _dedupe(queries, TARGET_QUERY_LIMIT)

# Returns focus phrases for query expansion.
def _focus_phrases(target):
    target = _clean_text(target)
    if not target: return []
    terms = _tokens(target); phrases = [target]; phrases.extend(terms)
    if len(terms) >= 2:
        phrases.append(" ".join(terms[:2])); phrases.append(" ".join(terms[-2:]))
    return _dedupe(phrases, TARGET_FOCUS_PHRASE_LIMIT)

# Returns bridge queries used by query expansion.
def _bridge_queries(target_a, target_b):
    target_a, target_b = _clean_text(target_a), _clean_text(target_b)
    if not target_a or not target_b: return []
    return _dedupe([f"{target_a} {target_b}",f"{target_a} and {target_b}",f"{target_a} relationship to {target_b}", f"{target_a} impact on {target_b}",f"{target_a} evidence {target_b}",f"{target_a} causes {target_b}", f"{target_a} effects on {target_b}", f"{target_a} data statistics {target_b}"], BRIDGE_QUERY_LIMIT)

# Builds query plan for query expansion.
def build_query_plan(target_a, target_b):
    target_a, target_b = _clean_text(target_a), _clean_text(target_b)
    return {"target_a_queries": _target_queries(target_a),"target_b_queries": _target_queries(target_b), "bridge_queries": _bridge_queries(target_a, target_b), "target_a_focus_phrases": _focus_phrases(target_a), "target_b_focus_phrases": _focus_phrases(target_b)}
