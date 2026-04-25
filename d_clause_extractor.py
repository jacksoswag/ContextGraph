import re, spacy  # type: ignore
from functools import lru_cache
from constants import (
    EXTRACTION_BLOCK_LIMIT,
    EXTRACTION_CLAUSE_LIMIT,
    EXTRACTION_SENTENCE_LIMIT,
    MAX_CHARS_PER_BLOCK,
)

try: nlp = spacy.load("en_core_web_sm", exclude=["coreferee", "ner"])
except Exception:
    try:
        import en_core_web_sm  # type: ignore

        nlp = en_core_web_sm.load(exclude=["coreferee", "ner"])
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

PRONOUNS = {"he","him","his","she","her","hers","it","its","they","them","their","theirs","this","that","these","those",}
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
LATEX_URL_RE = re.compile(r"\\url\{[^}]*\}", re.IGNORECASE)
HEADING_RE = re.compile(r"(?:=\s*){2,}[^=]{0,120}(?:\s*=){2,}")
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
CLAUSE_HEAD_DEPS = {"ROOT", "advcl", "ccomp", "xcomp", "conj", "relcl", "acl"}
CLAUSE_HEAD_POS = {"VERB", "AUX", "ADJ", "NOUN", "PROPN"}
DIRECT_COMPLEMENT_DEPS = {"obj", "dobj", "iobj", "attr", "oprd", "acomp"}
PREPOSITION_OBJECT_DEPS = {"pobj", "pcomp"}
INHERITED_SUBJECT_DEPS = {"conj", "xcomp", "advcl", "acl", "relcl"}
CLAUSAL_LINK_MARKERS = {"because", "if", "once", "unless", "when", "whenever"}
QUANTIFIER_DEPS = {"det", "predet", "quantmod"}
NOUN_MODIFIER_DEPS = {
    "amod",
    "compound",
    "flat",
    "fixed",
    "goeswith",
    "nmod:poss",
    "poss",
}
TRUTH_DEPS = {"neg"}
TENSE_DEPS = {"aux", "auxpass"}
RELEVANCE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def sanitize_block_text(text):
    text = str(text or "")
    text = LATEX_URL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = HEADING_RE.sub(" ", text)
    text = re.sub(r"[{}|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS_PER_BLOCK]


def _relevance_tokens(text):
    return {
        token.lower()
        for token in RELEVANCE_TOKEN_RE.findall(str(text or ""))
        if token
    }


def _relevance_score(text, query_tokens, query_text=""):
    if not query_tokens:
        return 0.0
    text_tokens = _relevance_tokens(text)
    if not text_tokens:
        return 0.0
    overlap = text_tokens & query_tokens
    if not overlap:
        return 0.0
    recall = len(overlap) / max(1, len(query_tokens))
    precision = len(overlap) / max(1, len(text_tokens))
    phrase_bonus = 0.25 if query_text and query_text.lower() in str(text or "").lower() else 0.0
    return recall + precision + phrase_bonus


def _rank_blocks(blocks, query=""):
    query_text = _clean_clause_piece(query)
    query_tokens = _relevance_tokens(query_text)
    ranked = []
    for order, block in enumerate(blocks or []):
        text = sanitize_block_text(block.get("content", ""))
        if not text:
            continue
        ranked.append(
            (
                _relevance_score(text, query_tokens, query_text),
                -order,
                text,
                str(block.get("tag") or "no-url"),
            )
        )
    if query_tokens:
        ranked.sort(reverse=True)
    return ranked[:EXTRACTION_BLOCK_LIMIT], query_tokens, query_text


def _allowed_phrase_token(candidate, head):
    if candidate.i == head.i:
        return True
    if candidate.is_space or candidate.dep_ == "punct":
        return False

    allowed_left_modifiers = {
        "amod", "case", "compound", "det", "flat", "fixed", "goeswith",
        "nmod:poss", "nummod", "poss", "quantmod",
    }
    allowed_right_modifiers = {"flat", "fixed", "goeswith"}

    if candidate.i < head.i:
        return candidate.dep_ in allowed_left_modifiers
    return candidate.dep_ in allowed_right_modifiers


def _phrase_tokens_from_head(token):
    phrase_tokens = {token.i: token}
    for child in token.children:
        if not _allowed_phrase_token(child, token):
            continue
        phrase_tokens[child.i] = child

        # Pull in short proper-name continuations like "los angeles" or "one america plaza".
        for grandchild in child.children:
            if _allowed_phrase_token(grandchild, token):
                phrase_tokens[grandchild.i] = grandchild

    return [phrase_tokens[idx] for idx in sorted(phrase_tokens)]


def phrase_from_token(token):
    words = [tok.text for tok in _phrase_tokens_from_head(token)]
    return " ".join(words).strip()


def _sentence_index(token):
    for idx, sent in enumerate(token.doc.sents):
        if sent.start <= token.i < sent.end:
            return idx
    return 0


def _is_valid_antecedent(token):
    if token is None:
        return False
    phrase = phrase_from_token(token).strip().lower()
    if not phrase or phrase in PRONOUNS:
        return False
    return token.pos_ in ("NOUN", "PROPN")

def _number_matches(pronoun, candidate):
    pronoun_lower = pronoun.lower_
    candidate_number = set(candidate.morph.get("Number"))
    if pronoun_lower in {"they", "them", "their", "theirs", "these", "those"}:
        return "Plur" in candidate_number or not candidate_number
    if pronoun_lower in {"it", "its", "this", "that", "he", "him", "his", "she", "her", "hers"}:
        return "Plur" not in candidate_number
    return True


def _score_antecedent(pronoun, candidate):
    if candidate.i >= pronoun.i:
        return -999

    distance = _sentence_index(pronoun) - _sentence_index(candidate)
    if distance < 0 or distance > 2:
        return -999

    score = 0
    if distance == 0:
        score += 3
    elif distance == 1:
        score += 6
    else:
        score += 2

    if candidate.dep_ in ("nsubj", "nsubjpass"):
        score += 4
    if candidate.pos_ == "PROPN":
        score += 3
    elif candidate.pos_ == "NOUN":
        score += 2
    if not _number_matches(pronoun, candidate):
        score -= 4
    return score

def _best_recent_antecedent(token):
    candidates = []
    for candidate in token.doc:
        if candidate.i >= token.i:
            break
        score = _score_antecedent(token, candidate)
        if score > -999:
            candidates.append((score, candidate.i, candidate))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _best_i, best_candidate = candidates[0]
    if best_score < 5:
        return None
    return best_candidate


def resolve_token(token):
    original = phrase_from_token(token)
    if token.pos_ != "PRON" and token.lower_ not in PRONOUNS:
        return original

    heuristic_candidate = _best_recent_antecedent(token)
    heuristic_phrase = phrase_from_token(heuristic_candidate) if heuristic_candidate is not None else ""
    heuristic_score = _score_antecedent(token, heuristic_candidate) if heuristic_candidate is not None else -999

    if heuristic_candidate is not None and heuristic_score >= 5:
        return heuristic_phrase

    return original


def resolve_pronouns(tokens):
    pieces = []
    for token in tokens:
        replacement = token.text
        if token.dep_ != "punct" and token.pos_ == "PRON":
            candidate = resolve_token(token)
            if candidate:
                replacement = candidate
        pieces.append(replacement + token.whitespace_)
    return "".join(pieces).strip()


def _clean_clause_piece(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _clean_literal_words(tokens):
    words = []
    seen = set()
    for token in tokens:
        if token is None or token.is_space or token.is_punct:
            continue
        token_id = getattr(token, "i", id(token))
        if token_id in seen:
            continue
        seen.add(token_id)
        words.append(_clean_clause_piece(token.text.lower()))
    return [word for word in words if word]

def _empty_component_metadata(cleaned=""):
    return {
        "surface": cleaned,
        "core": "",
        "head": "",
        "quantifier_words": [],
        "truth_words": [],
        "tense_words": [],
        "noun_modifiers": [],
    }


@lru_cache(maxsize=32768)
def _component_metadata(text):
    cleaned = _clean_clause_piece(text)
    if not cleaned:
        return _empty_component_metadata(cleaned)
    try:
        doc = nlp(cleaned)
    except Exception:
        doc = None
    return _component_metadata_from_doc(cleaned, doc)


def _component_metadata_map(texts):
    metadata = {}
    pending = []
    seen = set()
    for text in list(texts or []):
        cleaned = _clean_clause_piece(text)
        if cleaned in seen:
            continue
        seen.add(cleaned)
        if not cleaned:
            metadata[cleaned] = _empty_component_metadata(cleaned)
        else:
            pending.append(cleaned)

    if pending:
        try:
            for cleaned, doc in zip(pending, nlp.pipe(pending, batch_size=64)):
                metadata[cleaned] = _component_metadata_from_doc(cleaned, doc)
        except Exception:
            for cleaned in pending:
                metadata[cleaned] = _component_metadata(cleaned)
    return metadata


def _component_metadata_from_doc(cleaned, doc):
    metadata = {
        "surface": cleaned,
        "core": "",
        "head": "",
        "quantifier_words": [],
        "truth_words": [],
        "tense_words": [],
        "noun_modifiers": [],
    }
    if not cleaned:
        return metadata

    if doc is None or not len(doc):
        metadata["core"] = cleaned
        metadata["head"] = cleaned.lower()
        return metadata

    root = getattr(doc[:], "root", None)
    if root is None:
        root = next((token for token in doc if not token.is_space and not token.is_punct), None)
    if root is None:
        metadata["core"] = cleaned
        metadata["head"] = cleaned.lower()
        return metadata

    metadata["core"] = cleaned
    metadata["head"] = _clean_clause_piece((root.lemma_ or root.text).lower())

    quantifier_tokens = []
    modifier_tokens = []
    truth_tokens = []
    tense_tokens = []

    if root.pos_ in {"VERB", "AUX"}:
        subject_root = next((child for child in root.children if child.dep_ in SUBJECT_DEPS), None)
        if subject_root is not None:
            if subject_root.pos_ == "PRON" or subject_root.lower_ in PRONOUNS:
                metadata["core"] = _clean_clause_piece(resolve_token(subject_root))
            else:
                metadata["core"] = _clean_clause_piece(phrase_from_token(subject_root))

    for token in doc:
        if token.is_space or token.is_punct or token.i == root.i:
            continue
        if token.dep_ in QUANTIFIER_DEPS:
            quantifier_tokens.append(token)
            continue
        if token.dep_ == "nummod":
            if root.pos_ in {"NOUN", "PROPN"}:
                quantifier_tokens.append(token)
            else:
                modifier_tokens.append(token)
            continue
        if token.dep_ in NOUN_MODIFIER_DEPS:
            modifier_tokens.append(token)
            continue
        if token.dep_ in TRUTH_DEPS:
            truth_tokens.append(token)
            continue
        if token.dep_ in TENSE_DEPS:
            tense_tokens.append(token)

    if not tense_tokens and root.pos_ in {"VERB", "AUX"}:
        morph = set(root.morph.get("Tense")) | set(root.morph.get("VerbForm")) | set(root.morph.get("Mood"))
        if morph:
            tense_tokens.append(root)

    metadata["quantifier_words"] = _clean_literal_words(quantifier_tokens)
    metadata["truth_words"] = _clean_literal_words(truth_tokens)
    metadata["tense_words"] = _clean_literal_words(tense_tokens)
    metadata["noun_modifiers"] = _clean_literal_words(modifier_tokens)
    return metadata


def _unique_tokens(tokens):
    ordered = []
    seen = set()
    for token in tokens:
        if token is None:
            continue
        token_id = getattr(token, "i", id(token))
        if token_id in seen:
            continue
        seen.add(token_id)
        ordered.append(token)
    return ordered


def _token_phrase(token):
    if token is None:
        return ""
    if token.pos_ == "PRON" or token.lower_ in PRONOUNS:
        return _clean_clause_piece(resolve_token(token))
    return _clean_clause_piece(phrase_from_token(token))


def _expand_coordinated_roots(root):
    roots = [root]
    for child in root.children:
        if child.dep_ == "conj":
            roots.extend(_expand_coordinated_roots(child))
    return _unique_tokens(roots)


def _predicate_surface(token):
    if token is None:
        return ""
    surface = str(token.text or token.lemma_ or "").strip().lower()
    return _clean_clause_piece(surface)


def _predicate_surface_phrase(token):
    if token is None:
        return ""

    bag = {token.i: token}
    for child in token.children:
        if child.dep_ in {"aux", "auxpass", "neg"}:
            bag[child.i] = child

    if token.dep_ == "xcomp" and token.head is not None and token.head.pos_ in ("VERB", "AUX"):
        bag[token.head.i] = token.head
        for child in token.head.children:
            if child.dep_ in {"aux", "auxpass", "neg"}:
                bag[child.i] = child

    words = [
        bag[idx].text.lower()
        for idx in sorted(bag)
        if not bag[idx].is_space and not bag[idx].is_punct
    ]
    return _clean_clause_piece(" ".join(words))


def _subtree_text(token):
    words = [
        child.text
        for child in sorted(token.subtree, key=lambda item: item.i)
        if not child.is_space and child.dep_ != "mark"
    ]
    return _clean_clause_piece(" ".join(words))


def _copula_relation(token):
    if token is None:
        return ""
    copula = next((child for child in token.children if child.dep_ == "cop"), None)
    if copula is not None:
        return _predicate_surface(copula)
    if token.dep_ in {"attr", "acomp", "oprd"} and token.head is not None and token.head != token:
        if token.head.pos_ == "AUX":
            return _predicate_surface(token.head)
    return ""


def _clause_heads(sent):
    heads = [
        token
        for token in sent
        if token.dep_ in CLAUSE_HEAD_DEPS
        and (
            token.pos_ in CLAUSE_HEAD_POS
            or any(child.dep_ == "cop" for child in token.children)
        )
    ]
    if not heads and getattr(sent, "root", None) is not None:
        heads = [sent.root]
    return _unique_tokens(heads)

def _subject_roots_for_head(token):
    roots = [child for child in token.children if child.dep_ in SUBJECT_DEPS]

    if not roots and token.dep_ in INHERITED_SUBJECT_DEPS and token.head is not None and token.head != token:
        roots.extend(child for child in token.head.children if child.dep_ in SUBJECT_DEPS)

    if not roots and token.dep_ in {"attr", "acomp", "oprd"} and token.head is not None and token.head != token:
        roots.extend(child for child in token.head.children if child.dep_ in SUBJECT_DEPS)

    # SpaCy occasionally mis-tags compact clauses like "demand increases" with a
    # nominal root and a left-side `compound` instead of a normal subject. Treat
    # that left nominal as the clause subject so we can still preserve the
    # proposition shape.
    if not roots and token.pos_ in {"NOUN", "PROPN"}:
        roots.extend(
            child
            for child in token.children
            if child.dep_ == "compound" and child.i < token.i and child.pos_ in {"NOUN", "PROPN"}
        )
    expanded = []
    for root in roots:
        expanded.extend(_expand_coordinated_roots(root))
    return _unique_tokens(expanded)


def _clausal_link_marker(token):
    for child in token.children:
        marker = _clean_clause_piece(child.text.lower())
        if child.dep_ in {"advmod", "mark"} and marker in CLAUSAL_LINK_MARKERS:
            return marker
    return ""


def _direct_complement_pairs(token):
    base_relation = _predicate_surface_phrase(token) or _predicate_surface(token)
    pairs = []
    for child in token.children:
        if child.dep_ not in DIRECT_COMPLEMENT_DEPS:
            continue
        for complement in _expand_coordinated_roots(child):
            predicate_text = _token_phrase(complement) or _subtree_text(complement)
            if predicate_text:
                pairs.append((base_relation, predicate_text))
    return pairs

def _prepositional_pairs(token):
    base_relation = _predicate_surface_phrase(token) or _predicate_surface(token)
    pairs = []
    for child in token.children:
        if child.dep_ != "prep":
            continue
        relation_text = _clean_clause_piece(f"{base_relation} {child.text.lower()}")
        objects = [grandchild for grandchild in child.children if grandchild.dep_ in PREPOSITION_OBJECT_DEPS]
        for object_root in objects:
            for complement in _expand_coordinated_roots(object_root):
                predicate_text = _token_phrase(complement) or _subtree_text(complement)
                if predicate_text:
                    pairs.append((relation_text, predicate_text))
    return pairs

def _clausal_complement_pairs(token):
    base_relation = _predicate_surface_phrase(token) or _predicate_surface(token)
    pairs = []
    for child in token.children:
        if child.dep_ not in {"ccomp", "xcomp"}:
            continue
        predicate_text = _subtree_text(child) or _predicate_surface(child)
        if predicate_text:
            pairs.append((base_relation, predicate_text))
    return pairs

def _unary_predicate_text(token):
    copula = _copula_relation(token)
    if copula:
        return _token_phrase(token) or _predicate_surface(token)
    if token.pos_ in {"VERB", "AUX"}:
        return _predicate_surface_phrase(token) or _predicate_surface(token)
    if token.pos_ in {"NOUN", "PROPN"} and any(
        child.dep_ == "compound" and child.i < token.i for child in token.children
    ):
        return _predicate_surface_phrase(token) or _predicate_surface(token)
    return _token_phrase(token) or _predicate_surface(token)

def _compose_clause_text(subject_text, two_place_predicate, one_place_predicate, fallback=""):
    parts = []
    if subject_text:
        parts.append(subject_text)
    if two_place_predicate:
        parts.append(two_place_predicate)
    if one_place_predicate and one_place_predicate != subject_text:
        parts.append(one_place_predicate)
    text = _clean_clause_piece(" ".join(parts))
    return text or _clean_clause_piece(fallback)


def _fallback_clause_payload(sent):
    root = getattr(sent, "root", None)
    sentence_text = _clean_clause_piece(resolve_pronouns(sent) or sent.text)
    if root is None:
        return {
            "text": sentence_text,
            "subject": "",
            "two_place_predicate": "",
            "one_place_predicate": sentence_text,
        }

    subjects = _subject_roots_for_head(root)
    subject_text = _token_phrase(subjects[0]) if subjects else ""
    two_place_predicate = _copula_relation(root)
    one_place_predicate = _unary_predicate_text(root) or sentence_text
    fallback_text = _compose_clause_text(
        subject_text,
        two_place_predicate,
        one_place_predicate,
        fallback=sentence_text,
    ) or sentence_text
    return {
        "text": fallback_text,
        "subject": subject_text,
        "two_place_predicate": two_place_predicate,
        "one_place_predicate": one_place_predicate,
    }


def _head_clause_payloads(head, sentence_text):
    subjects = [
        _token_phrase(root)
        for root in _subject_roots_for_head(head)
        if _token_phrase(root)
    ]
    if not subjects:
        return []

    pairs = []
    pairs.extend(_direct_complement_pairs(head))
    pairs.extend(_prepositional_pairs(head))
    pairs.extend(_clausal_complement_pairs(head))
    if not pairs:
        pairs.append((_copula_relation(head), _unary_predicate_text(head)))

    payloads = []
    seen = set()
    for subject_text in subjects:
        for two_place_predicate, one_place_predicate in pairs:
            two_place_predicate = _clean_clause_piece(two_place_predicate)
            one_place_predicate = _clean_clause_piece(one_place_predicate)
            if not subject_text or not one_place_predicate:
                continue
            clause_text = _compose_clause_text(
                subject_text,
                two_place_predicate,
                one_place_predicate,
                fallback=sentence_text,
            )
            if not clause_text:
                continue
            signature = (
                subject_text.lower(),
                two_place_predicate.lower(),
                one_place_predicate.lower(),
            )
            if signature in seen:
                continue
            seen.add(signature)
            payloads.append({
                "text": clause_text,
                "subject": subject_text,
                "two_place_predicate": two_place_predicate,
                "one_place_predicate": one_place_predicate,
            })
    return payloads


def _structured_clauses_from_sentence(sent):
    sentence_text = _clean_clause_piece(resolve_pronouns(sent) or sent.text)
    records = []
    seen = set()

    for head in _clause_heads(sent):
        head_payloads = _head_clause_payloads(head, sentence_text)

        for payload in head_payloads:
            signature = (
                payload["subject"].lower(),
                payload["two_place_predicate"].lower(),
                payload["one_place_predicate"].lower(),
            )
            if signature in seen:
                continue
            seen.add(signature)
            records.append(payload)

        for child in head.children:
            if child.dep_ != "advcl":
                continue
            marker = _clausal_link_marker(child)
            if not marker:
                continue
            child_payloads = _head_clause_payloads(child, sentence_text)
            for main_payload in head_payloads:
                for child_payload in child_payloads:
                    link_subject = _clean_clause_piece(main_payload["text"])
                    link_predicate = _clean_clause_piece(child_payload["text"])
                    if not link_subject or not link_predicate:
                        continue
                    signature = (
                        link_subject.lower(),
                        marker.lower(),
                        link_predicate.lower(),
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    records.append({
                        "text": _compose_clause_text(
                            link_subject,
                            marker,
                            link_predicate,
                            fallback=sentence_text,
                        ),
                        "subject": link_subject,
                        "two_place_predicate": marker,
                        "one_place_predicate": link_predicate,
                    })

    if records:
        return records
    return [_fallback_clause_payload(sent)]


def clauses_from_sentence(sent):
    sentence_text = _clean_clause_piece(resolve_pronouns(sent) or sent.text)
    if not sentence_text:
        return []
    return _structured_clauses_from_sentence(sent)


def extract_clauses(blocks, query=""):
    if not blocks:
        return []

    ranked_blocks, query_tokens, query_text = _rank_blocks(blocks, query=query)
    cleaned = [item[2] for item in ranked_blocks]
    sources = [item[3] for item in ranked_blocks]

    sentence_rows = []
    for doc, source in zip(nlp.pipe(cleaned, batch_size=16), sources):
        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            if not sentence_text:
                continue
            sentence_rows.append(
                (
                    _relevance_score(sentence_text, query_tokens, query_text),
                    -len(sentence_rows),
                    sentence,
                    source,
                )
            )

    if query_tokens:
        sentence_rows.sort(reverse=True)
    sentence_rows = sentence_rows[:EXTRACTION_SENTENCE_LIMIT]

    clause_records = []
    for _score, _order, sentence, source in sentence_rows:
        for clause_payload in clauses_from_sentence(sentence):
            clause_records.append({
                "text": clause_payload.get("text", ""),
                "source": source,
                "subject": clause_payload.get("subject", ""),
                "two_place_predicate": clause_payload.get("two_place_predicate", ""),
                "one_place_predicate": clause_payload.get("one_place_predicate", ""),
            })
            if len(clause_records) >= EXTRACTION_CLAUSE_LIMIT:
                break
        if len(clause_records) >= EXTRACTION_CLAUSE_LIMIT:
            break

    if not clause_records:
        return []

    component_texts = []
    for record in clause_records:
        component_texts.extend(
            (
                record["subject"],
                record["two_place_predicate"],
                record["one_place_predicate"],
            )
        )
    metadata_by_text = _component_metadata_map(component_texts)

    parsed_records = []
    for record in clause_records:
        subject_meta = metadata_by_text.get(_clean_clause_piece(record["subject"])) or _component_metadata(record["subject"])
        relation_meta = metadata_by_text.get(_clean_clause_piece(record["two_place_predicate"])) or _component_metadata(record["two_place_predicate"])
        predicate_meta = metadata_by_text.get(_clean_clause_piece(record["one_place_predicate"])) or _component_metadata(record["one_place_predicate"])
        parsed_records.append({
            "text": record["text"],
            "source": record["source"],
            "subject": record["subject"],
            "two_place_predicate": record["two_place_predicate"],
            "one_place_predicate": record["one_place_predicate"],
            "subject_meta": subject_meta,
            "relation_meta": relation_meta,
            "predicate_meta": predicate_meta,
        })

    return parsed_records
