import re, coreferee, spacy  # type: ignore

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    try:
        import en_core_web_sm  # type: ignore

        nlp = en_core_web_sm.load()
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

if "coreferee" not in nlp.pipe_names:
    try:
        nlp.add_pipe("coreferee")
    except Exception:
        pass

PRONOUNS = {
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "this", "that", "these", "those",
}
META_TERMS = {
    "study", "paper", "article", "report", "research", "findings", "results",
    "data", "evidence", "analysis", "section", "chapter", "figure", "table",
    "example", "case", "instance", "introduction", "conclusion",
}
MAX_CHARS_PER_BLOCK = 2000
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
LATEX_URL_RE = re.compile(r"\\url\{[^}]*\}", re.IGNORECASE)
HEADING_RE = re.compile(r"(?:=\s*){2,}[^=]{0,120}(?:\s*=){2,}")
NOISY_TOKEN_RE = re.compile(r"(https?://|www\.|\\url\{|(?:=\s*){2,})", re.IGNORECASE)


def sanitize_block_text(text):
    text = str(text or "")
    text = LATEX_URL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = HEADING_RE.sub(" ", text)
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\(\d{4}(?:-\d{4})?\)", "", text)
    text = re.sub(r"[{}|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS_PER_BLOCK]


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
    phrase = phrase_from_token(token).strip().lower()
    if not phrase or phrase in PRONOUNS or phrase in META_TERMS:
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
    if not _is_valid_antecedent(candidate):
        return -999
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

    coref_candidate = None
    if hasattr(token._, "coref_chains") and token._.coref_chains:
        resolved = token._.coref_chains.resolve(token)
        if resolved:
            valid = [candidate for candidate in resolved if _is_valid_antecedent(candidate)]
            if valid:
                coref_candidate = max(valid, key=lambda candidate: (_score_antecedent(token, candidate), candidate.i))

    if coref_candidate is not None:
        coref_score = _score_antecedent(token, coref_candidate)
        if coref_score >= max(5, heuristic_score):
            return phrase_from_token(coref_candidate)

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


def looks_noisy_text(text):
    text = str(text or "").strip()
    if not text:
        return True
    if NOISY_TOKEN_RE.search(text):
        return True
    if text.count("=") >= 2:
        return True
    alpha_chars = sum(char.isalpha() for char in text)
    if alpha_chars < max(3, len(text) // 4):
        return True
    return False


def clauses_from_sentence(sent):
    sentence_text = resolve_pronouns(sent)
    if not sentence_text:
        return []
    if looks_noisy_text(sentence_text):
        return []
    if any(meta in sentence_text.lower() for meta in META_TERMS) and len(sentence_text.split()) <= 10:
        return []
    return [sentence_text]


def extract_clauses(blocks):
    if not blocks:
        return []

    cleaned = []
    sources = []
    for block in blocks:
        text = sanitize_block_text(block.get("content", ""))
        if not text:
            continue
        cleaned.append(text)
        sources.append(str(block.get("tag") or "no-url"))

    clause_records = []
    for doc, source in zip(nlp.pipe(cleaned, batch_size=16), sources):
        for sentence in doc.sents:
            if not sentence.text.strip():
                continue
            for clause_text in clauses_from_sentence(sentence):
                clause_records.append({"text": clause_text, "source": source})

    if not clause_records:
        return []

    parsed_records = []
    clause_texts = [record["text"] for record in clause_records]
    for clause_doc, record in zip(nlp.pipe(clause_texts, batch_size=16), clause_records):
        parsed_records.append({
            "text": record["text"],
            "source": record["source"],
            "doc": clause_doc,
        })

    return parsed_records
