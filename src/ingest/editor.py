from __future__ import annotations
import logging, os, re
from typing import Iterator
from ingest.extraction import extract_clauses

LOGGER = logging.getLogger(__name__)

# Question-detection: bare standalone question → skip ingestion (no factual content).
# Primary path is a 1B classifier; heuristic is the fallback when the model is unreachable.
def _is_bare_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped: return False
    # Only skip if the ENTIRE text is a single question (no additional sentences).
    sentences = [s.strip() for s in re.split(r"[.!]\s+", stripped) if s.strip()]
    if len(sentences) > 1: return False
    from llm import call_json
    try:
        ans = call_json(
            f'You are an expert NLP classifier. Is the following text a bare question asking for information, without stating any facts? Output a JSON object with a single boolean key "is_question".\n\nText: "{stripped}"',
            model="1B", role="interpreter")
        if isinstance(ans, dict) and "is_question" in ans: return bool(ans["is_question"])
    except Exception as exc:
        LOGGER.warning("LLM question check failed: %s", exc)
    return False


# Produce typed-triple dicts from interactive text. Pure: no store, no side
# effects — yields the same compositional edge dicts extract_clauses emits.
# Skips bare questions (no factual content) and, when DI_DROP_PENDING=1, the
# intransitive [event] pending edges (their signal lives in grounding only).
def ingest_text(text: str, *, drop_pending: bool | None = None) -> Iterator[dict]:
    if not text or _is_bare_question(text): return
    if drop_pending is None:
        drop_pending = os.getenv("DI_DROP_PENDING", "0") == "1"
    for clause in extract_clauses(text):
        if drop_pending and clause.get("_pending_completion"): continue
        yield clause
