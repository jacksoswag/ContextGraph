import json
import os
import re
from datetime import date
import fcntl

from constants import LOGICAL_CONNECTORS, REL_CONDITIONAL, REL_IDENTITY, REL_SUBSET
from d_clause_extractor import extract_clauses, nlp, resolve_token
from o_composed_sp import composed_sp


IDENTITY_SIGNS = {
    "is", "be", "become", "characterize", "constitute", "denote", "embody",
    "equal", "exemplify", "mean", "represent", "signify",
}
CONDITIONAL_SIGNS = {
    "affect", "begin", "cause", "compel", "create", "determine", "drive",
    "entail", "ensure", "follow", "force", "generate", "guarantee", "impact",
    "imply", "indicate", "induce", "influence", "initiate", "lead", "mandate",
    "necessitate", "predict", "produce", "prompt", "render", "require",
    "result", "spark", "start", "stipulate", "suggest", "trigger", "yield",
}
BICONDITIONAL_SIGNS = {
    "coextensive", "equal", "equate", "equivalent", "exchangeable", "iff",
    "interchangeable", "synonymous",
}
IFF_RE = re.compile(r"\b(?:iff|if and only if)\b", re.IGNORECASE)
CONJUNCTIVE_MARKERS = {"&", "and", "plus", "alongside", "together"}
DISJUNCTIVE_MARKERS = {"or", "either", "neither", "nor", "otherwise"}
SUBSET_NOUNS = {"class", "form", "group", "kind", "member", "part", "sort", "subset", "type"}
NEGATION_SIGNS = {
    "not", "never", "no", "none", "neither", "n't", "without", "ban", "cease",
    "deny", "disallow", "exclude", "fail", "forbid", "lack", "prohibit",
    "refuse", "reject", "stop", "veto",
}
DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "each", "every", "all", "no",
}
PRONOUNS = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "he",
    "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "we", "us", "our", "ours", "ourselves", "nothing", "they", "them",
    "their", "theirs", "themselves", "this", "that", "these", "those", "who",
    "whom", "whose", "which", "what", "someone", "anyone", "everyone", "no one",
    "something", "anything", "everything",
}
META_TERMS = {
    "study", "paper", "article", "report", "research", "thesis", "dissertation",
    "authors", "author", "researchers", "findings", "results", "data", "evidence",
    "analysis", "section", "chapter", "figure", "table", "example", "examples",
    "case", "cases", "instance", "bibliography", "references", "updated bibliography",
    "topics", "table of contents", "index", "introduction", "conclusion",
}
GENERIC_SUBJECTS = {
    "one", "ones", "thing", "things", "way", "ways", "kind", "kinds", "type",
    "types", "sort", "part", "parts", "area", "place", "patterns", "features",
    "elements", "processes",
}
UNIVERSAL_QUANTIFIERS = {"all", "every", "each"}
EXISTENTIAL_QUANTIFIERS = {"some"}
NEGATIVE_QUANTIFIERS = {"no", "none"}
PAST_TIME_MARKERS = {"ago", "earlier", "formerly", "previously", "yesterday"}
FUTURE_TIME_MARKERS = {"eventually", "later", "soon", "tomorrow", "upcoming"}
PRESENT_TIME_MARKERS = {"currently", "now", "presently", "today"}
PAST_TIME_MODIFIERS = {"last", "previous", "prior"}
FUTURE_TIME_MODIFIERS = {"coming", "following", "next"}
TEMPORAL_NOUNS = {
    "afternoon", "autumn", "day", "days", "decade", "decades", "evening", "fall",
    "future", "hour", "hours", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
    "month", "months", "morning", "night", "nights", "season", "seasons",
    "spring", "summer", "tonight", "week", "weeks", "winter", "year", "years",
}
NOISY_TEXT_RE = re.compile(r"(https?://|www\.|\\url\{|(?:=\s*){2,}|[{}|])", re.IGNORECASE)

REL_BICONDITIONAL = "biconditional"
REL_PREDICATE = "predicate"
RELATION_INDEX_PATH = "verb_index.json"
CURRENT_YEAR = date.today().year
COPULAR_PREDICATE_LABEL = "is"

QUANT_UNKNOWN = 0
QUANT_EXISTENTIAL = 1
QUANT_UNIVERSAL = 2
QUANT_NEGATIVE = 3

TENSE_UNKNOWN = 0
TENSE_PAST = 1
TENSE_PRESENT = 2
TENSE_FUTURE = 3

TENSE_CODE_TO_NAME = {
    TENSE_UNKNOWN: "unknown",
    TENSE_PAST: "past",
    TENSE_PRESENT: "present",
    TENSE_FUTURE: "future",
}


def _normalize_filter_term(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'()[]{}*_`~")


FILTERED_NODE_TERMS = {_normalize_filter_term(term) for term in (META_TERMS | GENERIC_SUBJECTS)}


def normalize_relation_name(label):
    return " ".join(str(label).lower().split()).strip()


def load_relation_index():
    labels = [normalize_relation_name(label) for label in LOGICAL_CONNECTORS]
    if not os.path.exists(RELATION_INDEX_PATH):
        return labels

    try:
        with open(RELATION_INDEX_PATH, "r") as handle:
            payload = json.load(handle)
    except Exception:
        return labels

    stored_logical = [normalize_relation_name(label) for label in payload.get("logical_connectors", [])]
    if stored_logical != labels:
        return labels

    seen = set(labels)
    for label in payload.get("verbs", []):
        normalized = normalize_relation_name(label)
        if normalized and normalized not in seen:
            labels.append(normalized)
            seen.add(normalized)
    return labels


RELATION_LABELS = load_relation_index()
RELATION_TO_ID = {label: idx for idx, label in enumerate(RELATION_LABELS)}


def _refresh_relation_cache(labels):
    global RELATION_LABELS, RELATION_TO_ID
    RELATION_LABELS = labels
    RELATION_TO_ID = {label: idx for idx, label in enumerate(RELATION_LABELS)}


def _relation_payload_from_labels(labels):
    logical_count = len(LOGICAL_CONNECTORS)
    relations = [
        {
            "id": relation_id,
            "label": label,
            "kind": "logical" if relation_id < logical_count else "verb",
        }
        for relation_id, label in enumerate(labels)
    ]
    return {
        "logical_connector_count": logical_count,
        "logical_connectors": list(LOGICAL_CONNECTORS),
        "verbs": labels[logical_count:],
        "relations": relations,
    }


def relation_id_for_label(label):
    normalized = normalize_relation_name(label)
    if not normalized:
        return None
    relation_id = RELATION_TO_ID.get(normalized)
    if relation_id is not None:
        return relation_id

    logical_labels = [normalize_relation_name(value) for value in LOGICAL_CONNECTORS]
    labels = list(logical_labels)

    with open(RELATION_INDEX_PATH, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        if raw:
            try:
                payload = json.loads(raw)
                stored_logical = [
                    normalize_relation_name(value)
                    for value in payload.get("logical_connectors", [])
                ]
                if stored_logical == logical_labels:
                    seen = set(logical_labels)
                    for value in payload.get("verbs", []):
                        normalized_value = normalize_relation_name(value)
                        if normalized_value and normalized_value not in seen:
                            labels.append(normalized_value)
                            seen.add(normalized_value)
            except Exception:
                labels = list(logical_labels)

        if normalized not in labels:
            labels.append(normalized)
            handle.seek(0)
            handle.truncate()
            json.dump(_relation_payload_from_labels(labels), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    _refresh_relation_cache(labels)
    return RELATION_TO_ID.get(normalized)


def relation_name_from_id(relation_id):
    if isinstance(relation_id, int) and 0 <= relation_id < len(RELATION_LABELS):
        return RELATION_LABELS[relation_id]
    return ""


def predicate_forms(token):
    forms = {token.lemma_.lower(), token.text.lower()}
    text = token.text.lower()
    if text.endswith("ed") and len(text) > 3:
        forms.add(text[:-2])
        forms.add(text[:-1])
    return forms


def relation_label(token, relation_token=None):
    if relation_token is not None:
        return relation_token.lemma_.lower()
    return token.lemma_.lower()


def predicate_relation_label(token, relation_token=None):
    label = relation_label(token, relation_token)
    if relation_token is None and label == "be":
        return COPULAR_PREDICATE_LABEL
    return label


def resolve_pronoun(token):
    return resolve_token(token)


def singularize_text(text):
    doc = nlp(text)
    singular_words = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        word = token.text.lower()
        if token.pos_ in ("NOUN", "PROPN", "PRON") or token.tag_ in ("NNS", "NNPS"):
            lemma = token.lemma_.lower().strip()
            if lemma and lemma != "-pron-":
                word = lemma
        singular_words.append(word)
    return " ".join(singular_words).strip()


def clean_text(text):
    if not text:
        return None

    text = re.sub(r"\s+", " ", str(text)).strip().lower()
    if NOISY_TEXT_RE.search(text):
        return None
    text = text.strip(" \t\n\r.,;:!?\"'()[]{}*_`~")
    if not text:
        return None

    words = text.split()
    while words and words[0] in DETERMINERS:
        words.pop(0)

    if not words:
        return None

    cleaned = " ".join(words)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = singularize_text(cleaned)
    if cleaned in PRONOUNS or cleaned in FILTERED_NODE_TERMS:
        return None
    if NOISY_TEXT_RE.search(cleaned):
        return None
    if len(cleaned.split()) > 12:
        return None
    return cleaned


def truth_value(token):
    current = token
    visited = set()

    while current.pos_ in ("VERB", "AUX") and current.i not in visited:
        visited.add(current.i)

        if current.dep_ == "neg" or current.lemma_.lower() in NEGATION_SIGNS:
            return False

        for child in current.children:
            if child.dep_ == "neg" or child.lemma_.lower() in NEGATION_SIGNS:
                return False

        if current.dep_ == "xcomp" and current.head.pos_ in ("VERB", "AUX"):
            current = current.head
            continue
        break

    return True
def predicate_modifiers(token):
    context = []
    visited = set()
    current = token

    while current.i not in visited:
        visited.add(current.i)
        context.append(current)
        context.extend(
            child for child in current.children
            if child.dep_ in ("aux", "auxpass", "neg", "advmod", "npadvmod")
        )
        if current.dep_ == "xcomp" and current.head.pos_ in ("VERB", "AUX"):
            current = current.head
            continue
        break

    if token.dep_ == "xcomp" and token.head.lemma_.lower() == "go":
        context.extend(child for child in token.head.children if child.dep_ in ("aux", "auxpass", "mark"))

    return [tok for tok in sorted({tok.i: tok for tok in context}.values(), key=lambda item: item.i)]


def temporal_direction(token):
    for word in token.sent:
        lower = word.text.lower()
        lemma = word.lemma_.lower()

        if lower in FUTURE_TIME_MARKERS or lemma in FUTURE_TIME_MARKERS:
            return TENSE_FUTURE
        if lower in PRESENT_TIME_MARKERS or lemma in PRESENT_TIME_MARKERS:
            return TENSE_PRESENT
        if lower in PAST_TIME_MARKERS or lemma in PAST_TIME_MARKERS:
            return TENSE_PAST

        if word.dep_ == "amod" and word.head.lemma_.lower() in TEMPORAL_NOUNS:
            if lemma in FUTURE_TIME_MODIFIERS:
                return TENSE_FUTURE
            if lemma in PAST_TIME_MODIFIERS:
                return TENSE_PAST

        if word.like_num and len(word.text) == 4 and word.text.isdigit():
            year = int(word.text)
            if year > CURRENT_YEAR:
                return TENSE_FUTURE
            if year == CURRENT_YEAR:
                return TENSE_PRESENT
            return TENSE_PAST

    return TENSE_UNKNOWN


def infer_tense(token):
    modifiers = predicate_modifiers(token)
    lemmas = {tok.lemma_.lower() for tok in modifiers}
    words = {tok.text.lower() for tok in modifiers}

    if lemmas & {"will", "shall"} or words & {"'ll"}:
        return TENSE_FUTURE

    if (
        token.dep_ == "xcomp"
        and token.head.lemma_.lower() == "go"
        and any(child.lower_ == "to" for child in token.head.children)
        and any(child.lemma_.lower() == "be" for child in token.head.children if child.dep_ in ("aux", "auxpass"))
    ):
        return TENSE_FUTURE

    if words & FUTURE_TIME_MARKERS:
        return TENSE_FUTURE

    tense = temporal_direction(token)
    if tense != TENSE_UNKNOWN:
        return tense

    if "Past" in token.morph.get("Tense"):
        return TENSE_PAST
    if "Pres" in token.morph.get("Tense"):
        return TENSE_PRESENT
    return TENSE_UNKNOWN


def tense_name(tense_code):
    return TENSE_CODE_TO_NAME.get(tense_code, "unknown")


def quantifier(token):
    for child in token.children:
        if child.dep_ != "det":
            continue
        value = child.lemma_.lower()
        if value in UNIVERSAL_QUANTIFIERS:
            return QUANT_UNIVERSAL
        if value in EXISTENTIAL_QUANTIFIERS:
            return QUANT_EXISTENTIAL
        if value in NEGATIVE_QUANTIFIERS:
            return QUANT_NEGATIVE
    return QUANT_UNKNOWN


def build_composed_sp(quantifier_value, tense_value, truth, asu_value):
    return composed_sp(
        quantifier=quantifier_value,
        tense=tense_value,
        truth=truth,
        ASU_idx=asu_value,
    )


def is_temporal_object(token, relation_token=None):
    relation_name = relation_token.lemma_.lower() if relation_token is not None else ""
    token_name = token.lemma_.lower()

    if token.ent_type_ in {"DATE", "TIME"}:
        return True
    if token.like_num and len(token.text) == 4 and token.text.isdigit():
        return True

    if relation_name in {"at", "before", "by", "during", "in", "on", "since", "until"}:
        if token_name in TEMPORAL_NOUNS:
            return True
        if any(
            child.dep_ == "amod" and child.lemma_.lower() in (PAST_TIME_MODIFIERS | FUTURE_TIME_MODIFIERS)
            for child in token.children
        ):
            return True

    if token.dep_ == "npadvmod" and token_name in TEMPORAL_NOUNS:
        return True
    return False


def is_participial_predicate(token):
    return (
        token.pos_ == "ADJ"
        and bool(predicate_forms(token) & (CONDITIONAL_SIGNS | IDENTITY_SIGNS))
        and token.dep_ in ("amod", "acomp", "oprd")
        and token.head.pos_ in ("NOUN", "PROPN")
    )


def is_nominal_modifier_predicate(token):
    return (
        token.dep_ in ("acl", "relcl", "amod")
        and token.head.pos_ in ("NOUN", "PROPN", "PRON")
        and (token.pos_ in ("VERB", "AUX") or is_participial_predicate(token))
    )


def is_subset_relation(token):
    if not predicate_forms(token) & IDENTITY_SIGNS:
        return False

    complements = [child for child in token.children if child.dep_ in ("attr", "acomp", "oprd")]
    for complement in complements:
        dets = {child.lemma_.lower() for child in complement.children if child.dep_ == "det"}
        if dets & {"a", "an"}:
            return True
        if complement.lemma_.lower() in SUBSET_NOUNS:
            return True
        if any(child.dep_ == "prep" and child.lemma_.lower() == "of" for child in complement.children):
            return True
    return False


def has_adjectival_complement(token):
    return any(child.dep_ in ("acomp", "oprd") and child.pos_ == "ADJ" for child in token.children)


def has_nominal_complement(token):
    return any(child.dep_ in ("attr", "oprd") and child.pos_ in ("NOUN", "PROPN", "PRON") for child in token.children)


def nominal_complements(token):
    return [child for child in token.children if child.dep_ in ("attr", "oprd") and child.pos_ in ("NOUN", "PROPN", "PRON")]


def subject_roots(token):
    roots = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
    if not roots and token.dep_ == "conj" and token.head.pos_ in ("VERB", "AUX"):
        roots = [child for child in token.head.children if child.dep_ in ("nsubj", "nsubjpass")]
    return roots


def is_named_identity_candidate(node):
    if node.pos_ == "PROPN" or bool(node.ent_type_):
        return True
    return any(child.dep_ in ("compound", "flat") and (child.pos_ == "PROPN" or bool(child.ent_type_)) for child in node.children)


def is_strong_identity_relation(token):
    subjects = subject_roots(token)
    complements = nominal_complements(token)
    if not subjects or not complements:
        return False
    return any(is_named_identity_candidate(subject) for subject in subjects) and any(
        is_named_identity_candidate(complement) for complement in complements
    )


def classify_relation(token):
    forms = predicate_forms(token)
    if is_subset_relation(token):
        return REL_SUBSET
    if forms & BICONDITIONAL_SIGNS:
        return REL_BICONDITIONAL
    if token == token.sent.root and IFF_RE.search(token.sent.text):
        return REL_BICONDITIONAL
    if forms & IDENTITY_SIGNS and has_adjectival_complement(token) and not has_nominal_complement(token):
        return REL_PREDICATE
    if forms & IDENTITY_SIGNS and has_nominal_complement(token):
        return REL_IDENTITY if is_strong_identity_relation(token) else REL_PREDICATE
    if forms & IDENTITY_SIGNS:
        return REL_IDENTITY
    if forms & CONDITIONAL_SIGNS:
        return REL_CONDITIONAL
    if token.pos_ in ("VERB", "AUX") or is_participial_predicate(token):
        return REL_PREDICATE
    return None


def expand_coordinated_roots(root):
    coordinated = [root]
    markers = {
        child.lemma_.lower()
        for child in root.children
        if child.dep_ == "cc"
    }
    if markers & (CONJUNCTIVE_MARKERS | DISJUNCTIVE_MARKERS):
        coordinated.extend(root.conjuncts)
    elif root.dep_ == "conj":
        coordinated.extend(root.conjuncts)
    return coordinated


def extract_arguments(token, rel_type):
    is_passive = any(child.dep_ == "nsubjpass" for child in token.children)

    active_subject_roots = [child for child in token.children if child.dep_ == "nsubj"]
    passive_subject_roots = [child for child in token.children if child.dep_ == "nsubjpass"]
    if not active_subject_roots and token.dep_ == "conj" and token.head.pos_ in ("VERB", "AUX"):
        active_subject_roots = [child for child in token.head.children if child.dep_ == "nsubj"]
    if not active_subject_roots and is_nominal_modifier_predicate(token):
        active_subject_roots = [token.head]
    if is_participial_predicate(token):
        active_subject_roots.extend(child for child in token.children if child.dep_ == "npadvmod")

    object_deps = {"obj", "dobj", "iobj", "attr"}
    if rel_type in (REL_CONDITIONAL, REL_PREDICATE, REL_BICONDITIONAL):
        object_deps.update({"xcomp", "acomp", "oprd"})

    direct_object_roots = [(child, None, False) for child in token.children if child.dep_ in object_deps]
    if is_participial_predicate(token):
        direct_object_roots = [(token.head, None, False)] + direct_object_roots

    prep_object_roots = []
    temporal_event_roots = []
    for prep in token.children:
        if prep.dep_ != "prep":
            continue
        for pobj in prep.children:
            if pobj.dep_ != "pobj":
                continue
            if is_temporal_object(pobj, prep):
                if is_nominal_modifier_predicate(token):
                    temporal_event_roots.append((pobj, None, False))
                continue
            prep_object_roots.append((pobj, prep, True))

    if temporal_event_roots:
        direct_object_roots.extend(temporal_event_roots)

    agent_subject_roots = [
        pobj
        for child in token.children
        if child.dep_ == "agent"
        for pobj in child.children
        if pobj.dep_ == "pobj"
    ]

    if is_passive:
        subject_roots = agent_subject_roots
        direct_object_roots = [(child, None, False) for child in passive_subject_roots] + direct_object_roots
    else:
        subject_roots = active_subject_roots

    subjects = []
    for root in subject_roots:
        subjects.extend(expand_coordinated_roots(root))

    objects = []
    for root, relation_token, via_prep in direct_object_roots + prep_object_roots:
        for item in expand_coordinated_roots(root):
            objects.append((item, relation_token, via_prep))

    return subjects, objects


def emit_connection(subject_text, predicate_text, connection_id, verb_label, truth, tense_value, evidence, source, left_quantifier):
    relation_name = verb_label or relation_name_from_id(connection_id)
    return {
        "subject": build_composed_sp(left_quantifier, tense_value, truth, subject_text),
        "connection": connection_id,
        "predicate": build_composed_sp(QUANT_UNKNOWN, tense_value, truth, predicate_text),
        "relation_label": relation_name,
        "source": source,
        "evidence": evidence,
    }


def extract_logical_connection(token, sent, source, rel_type):
    results = []
    tense_value = infer_tense(token)
    truth = truth_value(token)
    subjects, objects = extract_arguments(token, rel_type)

    if rel_type == REL_IDENTITY:
        logical_id = REL_IDENTITY
    elif rel_type == REL_SUBSET:
        logical_id = REL_SUBSET
    else:
        logical_id = REL_CONDITIONAL

    for subject_token in subjects:
        left_quantifier = quantifier(subject_token)
        for object_token, relation_token, via_prep in objects:
            subject_raw = resolve_pronoun(subject_token)
            if is_participial_predicate(token) and object_token == token.head:
                object_raw = object_token.text
            else:
                object_raw = resolve_pronoun(object_token)

            subject_text = clean_text(subject_raw)
            predicate_text = clean_text(object_raw)
            if not subject_text or not predicate_text or subject_text == predicate_text:
                continue

            if via_prep:
                verb_label = relation_label(token, relation_token)
                results.append(
                    emit_connection(
                        subject_text,
                        predicate_text,
                        relation_id_for_label(verb_label),
                        verb_label,
                        truth,
                        tense_value,
                        sent.text,
                        source,
                        left_quantifier,
                    )
                )
                continue

            results.append(
                emit_connection(
                    subject_text,
                    predicate_text,
                    logical_id,
                    None,
                    truth,
                    tense_value,
                    sent.text,
                    source,
                    left_quantifier,
                )
            )
            if rel_type == REL_BICONDITIONAL:
                results.append(
                    emit_connection(
                        predicate_text,
                        subject_text,
                        REL_CONDITIONAL,
                        None,
                        truth,
                        tense_value,
                        sent.text,
                        source,
                        QUANT_UNKNOWN,
                    )
                )

    return results


def extract_sp_connection(token, sent, source):
    results = []
    tense_value = infer_tense(token)
    truth = truth_value(token)
    subjects, objects = extract_arguments(token, REL_PREDICATE)

    for subject_token in subjects:
        left_quantifier = quantifier(subject_token)
        for object_token, relation_token, _via_prep in objects:
            subject_raw = resolve_pronoun(subject_token)
            if is_participial_predicate(token) and object_token == token.head:
                object_raw = object_token.text
            else:
                object_raw = resolve_pronoun(object_token)

            subject_text = clean_text(subject_raw)
            predicate_text = clean_text(object_raw)
            if not subject_text or not predicate_text or subject_text == predicate_text:
                continue

            verb_label = predicate_relation_label(token, relation_token)
            results.append(
                emit_connection(
                    subject_text,
                    predicate_text,
                    relation_id_for_label(verb_label),
                    verb_label,
                    truth,
                    tense_value,
                    sent.text,
                    source,
                    left_quantifier,
                )
            )

    return results


def statement_extractor(sent):
    heads = []
    seen = set()

    def add(token):
        if token.i in seen:
            return
        seen.add(token.i)
        heads.append(token)

    if sent.root.pos_ in ("VERB", "AUX") or is_participial_predicate(sent.root):
        add(sent.root)

    for token in sent:
        if token.i == sent.root.i:
            continue
        if token.pos_ not in ("VERB", "AUX") and not is_participial_predicate(token):
            continue
        if is_nominal_modifier_predicate(token):
            add(token)
            continue
        if token.dep_ in ("conj", "advcl", "ccomp", "xcomp", "relcl", "parataxis"):
            add(token)
            continue
        if any(child.dep_ in ("nsubj", "nsubjpass", "expl", "attr", "obj", "dobj", "iobj") for child in token.children):
            add(token)

    return heads


def analyze_statement(statement, sent, source):
    relation_type = classify_relation(statement)
    if relation_type is None:
        return []
    if relation_type in (REL_IDENTITY, REL_CONDITIONAL, REL_SUBSET, REL_BICONDITIONAL):
        return extract_logical_connection(statement, sent, source, relation_type)
    return extract_sp_connection(statement, sent, source)


def find_connections(blocks):
    connections = []
    seen = set()
    for clause in extract_clauses(blocks):
        source = clause["source"]
        doc = clause.get("doc")
        if doc is None:
            doc = nlp(clause["text"])
        for sentence in doc.sents:
            sentence_connections = []
            for statement in statement_extractor(sentence):
                sentence_connections.extend(analyze_statement(statement, sentence, source))

            for connection in sentence_connections:
                subject_sp = connection["subject"]
                predicate_sp = connection["predicate"]
                key = (
                    subject_sp.ASU_idx,
                    subject_sp.truth,
                    subject_sp.tense,
                    connection["connection"],
                    predicate_sp.ASU_idx,
                    predicate_sp.truth,
                    predicate_sp.tense,
                )
                if key in seen:
                    continue
                seen.add(key)
                connections.append(connection)
    return connections


def find_logic_forms(blocks):
    forms = []
    for conn in find_connections(blocks):
        subject_sp = conn["subject"]
        predicate_sp = conn["predicate"]
        relation_id = conn["connection"]
        relation_name = relation_name_from_id(relation_id)
        atom = {
            "kind": "atom",
            "subject": subject_sp.asu_value(),
            "predicate": predicate_sp.asu_value(),
            "truth": subject_sp.truth,
            "tense": tense_name(subject_sp.tense),
            "subject_truth": subject_sp.truth,
            "subject_tense": tense_name(subject_sp.tense),
            "subject_tense_code": subject_sp.tense,
            "predicate_truth": predicate_sp.truth,
            "predicate_tense": tense_name(predicate_sp.tense),
            "predicate_tense_code": predicate_sp.tense,
            "source": conn.get("source", "unknown"),
            "connection": relation_id,
        }
        if relation_id in (REL_IDENTITY, REL_CONDITIONAL, REL_SUBSET):
            atom["type"] = relation_id
        elif relation_name:
            atom["two_place_predicate"] = relation_name
        forms.append(atom)
    return forms
