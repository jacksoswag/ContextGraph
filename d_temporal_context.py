import re
from datetime import date

from constants import DEFAULT_UNKNOWN_YEAR


YEAR_RE = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
SOURCE_YEAR_RE = re.compile(r"(?:^|\|)year=((?:18|19|20|21)\d{2})(?:\||$)")

FUTURE_WORDS = {"will", "shall", "future"}
PRESENT_WORDS = {"am", "are", "is", "do", "does", "has", "have", "present", "vbp", "vbz"}
PAST_WORDS = {"did", "had", "past", "vbd", "vbn", "was", "were"}

STALE_PRESENT_YEARS = 2


def _clean_text(text):
    return " ".join(str(text or "").strip().split())


def source_year_from_source(source):
    text = _clean_text(source)
    match = SOURCE_YEAR_RE.search(text)
    if match:
        return int(match.group(1))
    return DEFAULT_UNKNOWN_YEAR


def _years_from_text(text):
    years = []
    seen = set()
    for match in YEAR_RE.finditer(str(text or "")):
        year = int(match.group(0))
        if year in seen:
            continue
        seen.add(year)
        years.append(year)
    return years


def event_year_from_clause(clause_text, source_year=DEFAULT_UNKNOWN_YEAR):
    clause = _clean_text(clause_text).lower()
    years = _years_from_text(clause)
    if years:
        return years[0], "explicit_year"

    if not source_year:
        return DEFAULT_UNKNOWN_YEAR, ""

    if re.search(r"\bnext\s+year\b", clause):
        return int(source_year) + 1, "relative_next_year"
    if re.search(r"\blast\s+year\b|\bprevious\s+year\b", clause):
        return int(source_year) - 1, "relative_last_year"
    if re.search(r"\bthis\s+year\b|\bcurrent\s+year\b", clause):
        return int(source_year), "relative_this_year"

    return DEFAULT_UNKNOWN_YEAR, ""


def original_tense_from_words(tense_words=None, clause_text=""):
    words = {
        _clean_text(word).lower()
        for word in list(tense_words or [])
        if _clean_text(word)
    }
    clause = _clean_text(clause_text).lower()

    if words & FUTURE_WORDS or re.search(r"\bwill\b|\bshall\b|\bgoing\s+to\b", clause):
        return "future"
    if words & PAST_WORDS:
        return "past"
    if words & PRESENT_WORDS:
        return "present"
    return "unknown"


def _tense_score(event_year, current_year):
    if not event_year:
        return 0.0
    delta = int(event_year) - int(current_year)
    if delta < 0:
        return -1.0
    if delta > 0:
        return 1.0
    return 0.0


def _relative_tense(event_year, current_year):
    if int(event_year) < int(current_year):
        return "past"
    if int(event_year) > int(current_year):
        return "future"
    return "present"


def _temporal_note(original_tense, current_tense, source_year, event_year, event_basis, current_year):
    if event_year:
        return (
            f"Original tense is {original_tense}; event year {event_year} "
            f"({event_basis}) resolves as {current_tense} relative to {current_year}."
        )

    if original_tense == "future":
        if source_year:
            return (
                f"Original tense is future from source year {source_year}, but no event date was found; "
                "treat this as a source-time projection, not a confirmed current future."
            )
        return "Original tense is future, but no source date or event date was found."

    if original_tense == "present":
        if source_year and current_year - source_year > STALE_PRESENT_YEARS:
            return (
                f"Original tense is present from source year {source_year}, but no event date was found; "
                "treat this as present-at-source-time and potentially stale today."
            )
        if source_year:
            return f"Original tense is present from recent source year {source_year}."
        return "Original tense is present, but no source date was found."

    if original_tense == "past":
        return "Original tense is past and remains past relative to today."

    return "No reliable temporal anchor was found."


def resolve_temporal_context(clause_text, source="", tense_words=None, today=None):
    current_date = today or date.today()
    current_year = int(current_date.year)
    source_year = source_year_from_source(source)
    event_year, event_basis = event_year_from_clause(clause_text, source_year=source_year)
    original_tense = original_tense_from_words(tense_words=tense_words, clause_text=clause_text)

    if event_year:
        current_tense = _relative_tense(event_year, current_year)
        confidence = "high"
        staleness = "resolved"
    elif original_tense == "future":
        current_tense = "unknown"
        confidence = "low"
        staleness = "projection"
    elif original_tense == "present" and source_year and current_year - source_year > STALE_PRESENT_YEARS:
        current_tense = "unknown"
        confidence = "low"
        staleness = "stale"
    elif original_tense == "present":
        current_tense = "present"
        confidence = "medium"
        staleness = "unknown"
    elif original_tense == "past":
        current_tense = "past"
        confidence = "medium"
        staleness = "stable"
    else:
        current_tense = "unknown"
        confidence = "low"
        staleness = "unknown"

    text = _temporal_note(
        original_tense,
        current_tense,
        source_year,
        event_year,
        event_basis,
        current_year,
    )
    return {
        "kind": "temporal_context",
        "text": text,
        "original_tense": original_tense,
        "current_tense": current_tense,
        "tense_score": round(_tense_score(event_year, current_year), 4),
        "temporal_confidence": confidence,
        "staleness": staleness,
        "source_year": int(source_year or DEFAULT_UNKNOWN_YEAR),
        "event_year": int(event_year or DEFAULT_UNKNOWN_YEAR),
        "event_basis": event_basis,
        "current_year": current_year,
    }
