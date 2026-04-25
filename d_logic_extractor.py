from d_clause_extractor import extract_clauses
from constants import MAX_CONNECTIONS_PER_QUERY
from d_temporal_context import resolve_temporal_context
from d_word_info_map import (
    contextual_component_vector,
    contextual_component_vectors,
    existing_concept_index,
    get_index,
    get_literal_index,
)
from o_connection import ConnectionEndpoint

def _clean_text(text):
    return " ".join(str(text or "").strip().split())

def _literal_index(text, clause_text=""):
    cleaned = _clean_text(text)
    if not cleaned:
        return -1
    return int(get_literal_index(cleaned))

def _literal_indices(words, clause_text=""):
    values = []
    seen = set()
    for word in words or []:
        index = _literal_index(word, clause_text=clause_text)
        if index < 0 or index in seen:
            continue
        seen.add(index)
        values.append(index)
    return values

def _first_literal(words, clause_text=""):
    for word in words or []:
        index = _literal_index(word, clause_text=clause_text)
        if index >= 0:
            return index
    return -1

def _concept_index(text, clause_text=""):
    cleaned = _clean_text(text)
    if not cleaned:
        return -1
    return int(get_index(cleaned, contextual_component_vector(cleaned, clause_text)))

def _concept_index_with_vector(text, vect):
    cleaned = _clean_text(text)
    if not cleaned:
        return -1
    return int(get_index(cleaned, vect))

def _strip_leading_quantifier(surface, quantifier_words):
    tokens = _clean_text(surface).split()
    quantifier_tokens = [_clean_text(word).lower() for word in quantifier_words or [] if _clean_text(word)]
    if quantifier_tokens and len(tokens) >= len(quantifier_tokens):
        leading = [token.lower() for token in tokens[: len(quantifier_tokens)]]
        if leading == quantifier_tokens:
            tokens = tokens[len(quantifier_tokens) :]
    return _clean_text(" ".join(tokens))

def _component_core_text(meta):
    meta = dict(meta or {})
    core = _strip_leading_quantifier(
        meta.get("core", ""),
        meta.get("quantifier_words", []),
    )
    surface = _strip_leading_quantifier(
        meta.get("surface", ""),
        meta.get("quantifier_words", []),
    )
    head = _clean_text(meta.get("head", ""))
    return core or surface or head

def _relation_index(record):
    relation_meta = dict(record.get("relation_meta") or {})
    clause_text = _clean_text(record.get("text", ""))
    relation_lemma = _clean_text(
        relation_meta.get("head")
        or record.get("two_place_predicate")
    )
    if not relation_lemma:
        return -1
    return _literal_index(relation_lemma, clause_text=clause_text)

def _simple_truth_literal(meta, clause_text=""):
    meta = dict(meta or {})
    return _first_literal(meta.get("truth_words", []), clause_text=clause_text)

def _concept_id_from_prepared(text, clause_text, concept_vectors=None, existing_concepts=None):
    cleaned = _clean_text(text)
    if not cleaned:
        return -1
    existing = (existing_concepts or {}).get(cleaned)
    if existing is not None and existing >= 0:
        return existing
    vector = (concept_vectors or {}).get((cleaned, clause_text))
    if vector is not None:
        return _concept_index_with_vector(cleaned, vector)
    return _concept_index(cleaned, clause_text=clause_text)


def _build_subject_sp(record, concept_vectors=None, existing_concepts=None):
    subject_meta = dict(record.get("subject_meta") or {})
    clause_text = _clean_text(record.get("text", ""))
    concept_text = _component_core_text(subject_meta)
    modifier_ids = _literal_indices(subject_meta.get("noun_modifiers", []), clause_text=clause_text)
    tense_id = _first_literal(subject_meta.get("tense_words", []), clause_text=clause_text)
    relation_meta = dict(record.get("relation_meta") or {})
    truth_id = _simple_truth_literal(relation_meta, clause_text=clause_text)
    if tense_id < 0:
        tense_id = _first_literal(relation_meta.get("tense_words", []), clause_text=clause_text)

    return ConnectionEndpoint(
        quantifier=_first_literal(subject_meta.get("quantifier_words", []), clause_text=clause_text),
        tense=tense_id,
        truth=truth_id,
        ASU_idx=_concept_id_from_prepared(
            concept_text,
            clause_text,
            concept_vectors=concept_vectors,
            existing_concepts=existing_concepts,
        ),
        modifier_idx=modifier_ids,
    )

def _build_predicate_sp(record, concept_vectors=None, existing_concepts=None):
    clause_text = _clean_text(record.get("text", ""))
    predicate_meta = dict(record.get("predicate_meta") or {})
    relation_meta = dict(record.get("relation_meta") or {})
    concept_text = _component_core_text(predicate_meta)
    modifier_ids = _literal_indices(predicate_meta.get("noun_modifiers", []), clause_text=clause_text)
    return ConnectionEndpoint(
        quantifier=_first_literal(predicate_meta.get("quantifier_words", []), clause_text=clause_text),
        tense=_first_literal(relation_meta.get("tense_words", []), clause_text=clause_text),
        truth=_simple_truth_literal(relation_meta, clause_text=clause_text),
        ASU_idx=_concept_id_from_prepared(
            concept_text,
            clause_text,
            concept_vectors=concept_vectors,
            existing_concepts=existing_concepts,
        ),
        modifier_idx=modifier_ids,
    )

def _connection_record(record, concept_vectors=None, existing_concepts=None):
    subject_sp = _build_subject_sp(
        record,
        concept_vectors=concept_vectors,
        existing_concepts=existing_concepts,
    )
    predicate_sp = _build_predicate_sp(
        record,
        concept_vectors=concept_vectors,
        existing_concepts=existing_concepts,
    )
    try:
        relation_index = int(record.get("_relation_index", -1))
    except (TypeError, ValueError):
        relation_index = -1
    if relation_index < 0:
        relation_index = _relation_index(record)
    clause_text = _clean_text(record.get("text", ""))
    temporal_context = _connection_temporal_context(record, clause_text)
    connection_specifics = [temporal_context] if temporal_context else []
    return {
        "subject": subject_sp,
        "connection": relation_index,
        "predicate": predicate_sp,
        "text": clause_text,
        "source": _clean_text(record.get("source", "")) or "unknown",
        "subject_specifics": [],
        "predicate_specifics": [],
        "connection_specifics": connection_specifics,
        "truth": 1,
    }


def _connection_temporal_context(record, clause_text):
    relation_meta = dict(record.get("relation_meta") or {})
    subject_meta = dict(record.get("subject_meta") or {})
    predicate_meta = dict(record.get("predicate_meta") or {})
    tense_words = (
        list(relation_meta.get("tense_words", []) or [])
        + list(subject_meta.get("tense_words", []) or [])
        + list(predicate_meta.get("tense_words", []) or [])
    )
    temporal = resolve_temporal_context(
        clause_text,
        source=_clean_text(record.get("source", "")),
        tense_words=tense_words,
    )
    if (
        temporal.get("original_tense") == "unknown"
        and not temporal.get("source_year")
        and not temporal.get("event_year")
    ):
        return None
    return temporal


def _connection_key(connection):
    subject_sp = connection["subject"]
    predicate_sp = connection["predicate"]
    return (
        subject_sp.sp_id,
        connection["connection"],
        predicate_sp.sp_id,
        subject_sp.quantifier,
        subject_sp.tense,
        subject_sp.truth,
        tuple(subject_sp.modifier_idx),
        predicate_sp.quantifier,
        predicate_sp.tense,
        predicate_sp.truth,
        tuple(predicate_sp.modifier_idx),
    )


def _candidate_records(records, connection_limit):
    if connection_limit is not None and connection_limit <= 0:
        return []

    candidates = []
    seen = set()
    for record in records:
        subject_text = _component_core_text(record.get("subject_meta") or {})
        predicate_text = _component_core_text(record.get("predicate_meta") or {})
        relation_id = _relation_index(record)
        if not subject_text or not predicate_text or relation_id < 0:
            continue

        signature = (
            subject_text.lower(),
            relation_id,
            predicate_text.lower(),
            _clean_text(record.get("text", "")).lower(),
        )
        if signature in seen:
            continue
        seen.add(signature)

        prepared = dict(record)
        prepared["_relation_index"] = relation_id
        candidates.append(prepared)
        if connection_limit is not None and len(candidates) >= connection_limit * 2:
            break
    return candidates


def find_connections(blocks, query="", connection_limit=MAX_CONNECTIONS_PER_QUERY):
    if connection_limit is not None and connection_limit <= 0:
        return []

    connections = []
    seen = set()
    records = _candidate_records(
        extract_clauses(blocks, query=query),
        connection_limit,
    )
    vector_keys = []
    seen_vector_keys = set()
    concept_texts = []
    seen_concept_texts = set()
    for record in records:
        clause_text = _clean_text(record.get("text", ""))
        for meta_key in ("subject_meta", "predicate_meta"):
            concept_text = _component_core_text(record.get(meta_key) or {})
            if not concept_text:
                continue
            if concept_text not in seen_concept_texts:
                seen_concept_texts.add(concept_text)
                concept_texts.append(concept_text)

    existing_concepts = {
        concept_text: existing_concept_index(concept_text)
        for concept_text in concept_texts
    }

    for record in records:
        clause_text = _clean_text(record.get("text", ""))
        for meta_key in ("subject_meta", "predicate_meta"):
            concept_text = _component_core_text(record.get(meta_key) or {})
            key = (concept_text, clause_text)
            if concept_text and existing_concepts.get(concept_text, -1) < 0 and key not in seen_vector_keys:
                seen_vector_keys.add(key)
                vector_keys.append(key)

    vectors = contextual_component_vectors(vector_keys)
    concept_vectors = {
        key: vector
        for key, vector in zip(vector_keys, vectors)
    }

    for record in records:
        connection = _connection_record(
            record,
            concept_vectors=concept_vectors,
            existing_concepts=existing_concepts,
        )
        subject_sp = connection["subject"]
        predicate_sp = connection["predicate"]
        if subject_sp.sp_id < 0 or predicate_sp.sp_id < 0 or connection["connection"] < 0:
            continue
        key = _connection_key(connection)
        if key in seen:
            continue
        seen.add(key)
        connections.append(connection)
        if connection_limit is not None and len(connections) >= connection_limit:
            break
    return connections
