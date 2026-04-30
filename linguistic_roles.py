import re
from collections import OrderedDict
from functools import lru_cache

from constants import STOPWORD_TOKENS


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
DOC_CACHE_LIMIT = 8192
_DOC_CACHE = OrderedDict()

# Closed-class grammar markers. These are stable English function words, not
# domain/query terms.
AUXILIARY_TOKENS = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "was",
        "were",
        "will",
        "would",
    }
)
SUBORDINATE_CLAUSE_TOKENS = frozenset(
    {"because", "how", "that", "whether", "which", "who", "whom", "whose", "why"}
)
DISCOURSE_MARKER_TOKENS = frozenset(
    {
        "actually",
        "according",
        "additionally",
        "also",
        "anyway",
        "but",
        "compared",
        "consequently",
        "conversely",
        "even",
        "finally",
        "furthermore",
        "hence",
        "here",
        "however",
        "instead",
        "lastly",
        "meanwhile",
        "moreover",
        "nevertheless",
        "nonetheless",
        "notably",
        "only",
        "otherwise",
        "rather",
        "similarly",
        "since",
        "so",
        "still",
        "then",
        "there",
        "therefore",
        "though",
        "thus",
        "ultimately",
        "well",
        "what",
        "yet",
    }
)


def token_words(text):
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def is_auxiliary_token(token):
    return str(token or "").lower() in AUXILIARY_TOKENS


def is_subordinate_marker(token):
    return str(token or "").lower() in SUBORDINATE_CLAUSE_TOKENS


def is_discourse_marker(token):
    return str(token or "").lower() in DISCOURSE_MARKER_TOKENS


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy  # type: ignore

        return spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:
        return None


def _remember_doc(text, doc):
    _DOC_CACHE[text] = doc
    _DOC_CACHE.move_to_end(text)
    while len(_DOC_CACHE) > DOC_CACHE_LIMIT:
        _DOC_CACHE.popitem(last=False)


def _doc(text):
    text = " ".join(str(text or "").strip().split())
    if not text:
        return None
    cached = _DOC_CACHE.get(text)
    if cached is not None:
        _DOC_CACHE.move_to_end(text)
        return cached
    nlp = _nlp()
    if nlp is None:
        return None
    try:
        doc = nlp(text)
        _remember_doc(text, doc)
        return doc
    except Exception:
        return None


def preparse_texts(texts):
    nlp = _nlp()
    if nlp is None:
        return
    missing = []
    seen = set()
    for text in list(texts or []):
        clean = " ".join(str(text or "").strip().split())
        if not clean or clean in seen or clean in _DOC_CACHE:
            continue
        seen.add(clean)
        missing.append(clean)
    if not missing:
        return
    try:
        for clean, doc in zip(missing, nlp.pipe(missing)):
            _remember_doc(clean, doc)
    except Exception:
        for clean in missing:
            _doc(clean)


def _subtree_text(doc, token):
    indices = [item.i for item in token.subtree if not item.is_punct]
    if not indices:
        return token.text
    return doc[min(indices) : max(indices) + 1].text.strip(" ,;:")


def _first_subject(root):
    for child in root.children:
        if child.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
            return child
    return None


def _first_predicate(root):
    preferred = {
        "attr",
        "acomp",
        "dobj",
        "dative",
        "npadvmod",
        "obj",
        "oprd",
        "obl",
    }
    for child in root.children:
        if child.dep_ in preferred:
            return child
    for child in root.children:
        if child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ in {"pobj", "obj"}:
                    return grandchild
            return child
    return None


def _root_token(doc):
    roots = [token for token in doc if token.dep_ == "ROOT"]
    for token in roots:
        if token.pos_ in {"VERB", "AUX"}:
            return token
    return roots[0] if roots else None


def embedded_statement_text(text):
    """Return a subordinate statement when a wrapper clause embeds one."""
    doc = _doc(text)
    if doc is None:
        return ""
    root = _root_token(doc)
    if root is None:
        return ""
    for child in root.children:
        if child.dep_ not in {"ccomp", "advcl"}:
            continue
        if any(item.dep_ in {"nsubj", "nsubjpass", "csubj"} for item in child.subtree):
            return _subtree_text(doc, child)
    return ""


def dependency_relation(text):
    """Extract a grammar-root relation without relying on enumerated verbs."""
    doc = _doc(text)
    if doc is None:
        return None
    root = _root_token(doc)
    if root is None or root.pos_ not in {"VERB", "AUX"}:
        return None
    subject = _first_subject(root)
    predicate = _first_predicate(root)
    if subject is None or predicate is None:
        return None

    relation = root.lemma_.lower() if root.lemma_ and root.lemma_ != "-PRON-" else root.text.lower()
    return {
        "subject_raw": _subtree_text(doc, subject),
        "relation": relation,
        "predicate_raw": _subtree_text(doc, predicate),
    }


def relation_word_score(token):
    """A fallback morphology score when the parser cannot find a relation."""
    word = str(token or "").lower()
    if not word or word in STOPWORD_TOKENS or word in AUXILIARY_TOKENS:
        return 0.0
    if len(word) <= 2 or not re.search(r"[a-z]", word):
        return 0.0
    if word.endswith(("ed", "ing", "izes", "ises", "ates", "ifies")):
        return 0.95
    if word.endswith(("ize", "ise", "ate", "ify")) and len(word) > 5:
        return 0.8
    if word.endswith("s") and len(word) > 4 and not word.endswith(("ics", "ness", "ous", "ss")):
        return 0.65
    return 0.0


def looks_like_relation_word(token):
    return relation_word_score(token) > 0.0


def parsed_tense(text, relation=""):
    doc = _doc(text)
    if doc is not None:
        relation = str(relation or "").lower()
        for token in doc:
            if token.pos_ not in {"VERB", "AUX"}:
                continue
            if relation and token.lemma_.lower() != relation and token.text.lower() != relation:
                continue
            if token.tag_ in {"VBD", "VBN"}:
                return "past"
            if token.tag_ == "MD" or token.text.lower() in {"will", "shall"}:
                return "future"
            return "present"

    lower = f" {relation.lower()} {str(text or '').lower()} "
    if re.search(r"\b(will|shall|going to)\b", lower):
        return "future"
    if re.search(r"\b(?:was|were|had|did)\b", lower) or relation.lower().endswith("ed"):
        return "past"
    return "present"


def looks_like_clause(text):
    tokens = token_words(text)
    if len(tokens) < 3:
        return False
    doc = _doc(text)
    if doc is not None:
        has_subject = any(token.dep_ in {"nsubj", "nsubjpass", "csubj"} for token in doc)
        has_predicate = any(token.pos_ in {"VERB", "AUX"} for token in doc)
        if has_subject and has_predicate:
            return True
    token_set = set(tokens)
    if len(tokens) >= 4 and token_set & AUXILIARY_TOKENS:
        return True
    return bool((token_set & SUBORDINATE_CLAUSE_TOKENS) and (token_set & AUXILIARY_TOKENS))
