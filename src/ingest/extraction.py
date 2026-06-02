from __future__ import annotations
import logging, os, typing
from types import SimpleNamespace

LOGGER = logging.getLogger(__name__)
_nlp = None

BASIC_PRONOUNS = frozenset({"he", "she", "it", "they", "his", "her", "its", "their", "him", "them"})
_SUBJ_DEPS = frozenset({"nsubj", "nsubjpass"})
_OBJ_DEPS = frozenset({"dobj", "attr"})
_CLAUSE_DEPS = frozenset({"ROOT", "relcl", "advcl", "ccomp", "xcomp"})
_MAX_EDGE_DEPTH = 5
# Allowed POS for object/prep-object/compound positions. NUM admits dates and
# figures; subjects stay NOUN/PROPN.
_VALUE_POS = frozenset({"NOUN", "PROPN", "NUM"})
# NER labels that should keep their full entity span as a single atomic node
# instead of being decomposed (proper names, organizations, places, events,
# laws, dates, etc).
_ATOMIC_ENT_LABELS = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE",
})
# NER labels we treat as time anchors — captured into edge.time_phrase
# alongside spaCy AUX-driven tense detection. Resolved to ISO dates by the
# Stage-3 LLM pass in deep_ingest.
_TIME_ENT_LABELS = frozenset({"DATE", "TIME"})

# Imperative subject sentinel.
_IMPLICIT_SYSTEM_SUBJ = SimpleNamespace(
    lemma_="system", lower_="system", pos_="PROPN", dep_="nsubj",
    children=[], i=-1, ent_type_="", ent_iob_="O", doc=None,
)


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        for model_name in ("en_core_web_sm", "en_core_web_md"):
            try:
                _nlp = spacy.load(model_name)
                break
            except OSError:
                continue
        if _nlp is None:
            raise RuntimeError(
                "no spaCy English model found; run: python -m spacy download en_core_web_sm"
            )
    return _nlp


_coref_nlp = None


def _get_coref_nlp():
    # Cached pipeline for the coref pre-pass — includes coreferee.
    # Kept distinct from get_nlp() so the main extraction chunks don't pay the
    # coreferee cost a second time after pronouns are already resolved.
    # Prefers md: 300d word vectors let coreferee use Token.similarity() properly
    # instead of falling back to tagger heuristics (W007 warning).
    # md is ONLY used here; main extraction stays on sm to avoid relcl regressions.
    global _coref_nlp
    if _coref_nlp is None:
        import spacy
        for model_name in ("en_core_web_md", "en_core_web_sm"):
            try:
                _coref_nlp = spacy.load(model_name)
                try:
                    import coreferee  # noqa: F401
                    _coref_nlp.add_pipe("coreferee")
                except Exception:
                    pass  # coreferee absent; resolve_basic_pronouns returns text unchanged
                break
            except OSError:
                continue
    return _coref_nlp


# ── Coreference pre-pass ─────────────────────────────────────────────────────
def resolve_basic_pronouns(text: str) -> str:
    nlp = _get_coref_nlp()
    if nlp is None: return text
    doc = nlp(text)
    chains = getattr(doc._, "coref_chains", None)
    if not chains: return text
    replacements: dict[int, str] = {}
    for chain in chains:
        root_text: str | None = None
        try:
            root_tokens = [doc[i] for i in chain[chain.most_specific_mention_index]]
            if not all(t.lower_ in BASIC_PRONOUNS for t in root_tokens):
                root_text = " ".join(t.text for t in root_tokens)
        except Exception: pass
        if root_text is None:
            for mention in chain:
                tokens = [doc[i] for i in mention]
                if not all(t.lower_ in BASIC_PRONOUNS for t in tokens):
                    root_text = " ".join(t.text for t in tokens)
                    break
        if root_text is None: continue
        for mention in chain:
            tokens = [doc[i] for i in mention]
            if len(tokens) == 1 and tokens[0].lower_ in BASIC_PRONOUNS:
                replacements[tokens[0].i] = root_text
    if not replacements: return text
    return "".join(
        (replacements[tok.i] + tok.whitespace_) if tok.i in replacements
        else tok.text_with_ws
        for tok in doc
    )


# ── NER entity helpers ───────────────────────────────────────────────────────
# Map token → containing NER entity span (cached lazily per doc to avoid
# rebuilding for every node lookup in the same sentence).
def _entity_for(token) -> tuple[int, int, str, str] | None:
    if getattr(token, "doc", None) is None or not token.ent_type_:
        return None
    # spaCy doesn't directly expose token→ent. doc.ents is short enough that
    # linear scan is fine; we cache results on the doc itself.
    cache = getattr(token.doc._, "_entity_map_cache", None)
    if cache is None:
        cache = {}
        for ent in token.doc.ents:
            for i in range(ent.start, ent.end):
                cache[i] = (ent.start, ent.end, ent.text, ent.label_)
        try: token.doc._._entity_map_cache = cache
        except Exception: pass  # Spans/docs sometimes can't set attrs
    return cache.get(token.i)


# ── Noun phrase / node construction ──────────────────────────────────────────
# Prefer the NER entity span for proper-name multi-word nodes (United States,
# North Atlantic Treaty Organization, Article 5). Otherwise compose from
# compound/nmod/nummod/appos siblings in source order.
def _np_text(token) -> str:
    ent = _entity_for(token)
    if ent is not None and ent[3] in _ATOMIC_ENT_LABELS:
        text = ent[2].lower()
        for prefix in ("the ", "a ", "an "):
            if text.startswith(prefix):
                text = text[len(prefix):]; break
        return text
    related = [
        c for c in token.children
        if c.dep_ in {"compound", "nmod", "nummod", "appos"}
        and c.pos_ in _VALUE_POS
    ]
    ordered = sorted(related + [token], key=lambda t: t.i)
    return " ".join(t.lemma_.lower() for t in ordered)


# Atomic node — no nested modifiers. Modifiers become hyperedges on the
# clause edge (see _build_edge), not baked into the node dict.
def _build_node(token) -> dict:
    return {"type": "node", "text": _np_text(token), "pos": token.pos_}


# ── Tense / time helpers ─────────────────────────────────────────────────────
def _detect_tense(verb_tok) -> str | None:
    aux_lemmas: list[str] = []
    aux_tags: list[str] = []
    for c in verb_tok.children:
        if c.dep_ in {"aux", "auxpass"}:
            aux_lemmas.append(c.lemma_.lower())
            aux_tags.append(c.tag_)
    has_modal_future = any(l in {"will", "shall"} for l in aux_lemmas) or "MD" in aux_tags
    has_have = any(l == "have" for l in aux_lemmas)
    has_past_aux = any(t in {"VBD"} for t in aux_tags)
    if has_modal_future:
        return "future"
    if has_have and has_past_aux:
        return "past_perfect"
    if has_have:
        return "present_perfect"
    if has_past_aux: return "past"
    # Fall back to the verb tag itself when no aux.
    if verb_tok.tag_ == "VBD": return "past"
    if verb_tok.tag_ in {"VBP", "VBZ", "VB"}: return "present"
    return None


# Find DATE/TIME entity strings anchored to this verb. Walks prep-pobj and
# direct dep children for a DATE/TIME entity span. Returns the entity text
# or None — no resolution, just preservation.
def _detect_time_phrase(verb_tok) -> str | None:
    def _scan_token(tok) -> str | None:
        ent = _entity_for(tok)
        if ent is not None and ent[3] in _TIME_ENT_LABELS:
            return ent[2].lower()
        return None
    # Direct children of verb.
    for c in verb_tok.children:
        s = _scan_token(c)
        if s is not None: return s
        # prep → pobj walk
        if c.dep_ == "prep":
            for gc in c.children:
                if gc.dep_ == "pobj":
                    s = _scan_token(gc)
                    if s is not None: return s
        # npadvmod (e.g., "last year")
        if c.dep_ in {"npadvmod", "advmod", "tmod"}:
            s = _scan_token(c)
            if s is not None: return s
    return None


# ── Passive-to-active normalization ─────────────────────────────────────────
# Detects passive constructions and returns (logical_subj, verb_rel, logical_obj)
# tokens so the caller emits active-voice edges (agent acts on patient).
#
# Passive pattern: nsubjpass is the patient; agent is the logical subject.
# "Rome was founded by Romulus" → Romulus -found-> Rome
# "The bill was signed" (agentless) → keep patient as subject (no inversion),
# relation gains "receive_" prefix to signal receptive framing.
def _passive_rewrite(verb_tok, subj_tok) -> tuple | None:
    # Only rewrite nsubjpass subjects.
    if getattr(subj_tok, "dep_", "") != "nsubjpass":
        return None
    # Look for "by <agent>" prepositional child on the verb.
    agent_tok = None
    for c in verb_tok.children:
        if c.dep_ == "agent":
            for gc in c.children:
                if gc.dep_ == "pobj" and gc.pos_ in _VALUE_POS:
                    agent_tok = gc
                    break
            if agent_tok: break
    if agent_tok is None:
        # Agentless passive — return prefix signal; subj stays as-is.
        return ("agentless", subj_tok)
    # Full passive inversion: agent becomes subject, patient becomes object.
    return ("invert", agent_tok, subj_tok)


# ── Edge construction (compositional / recursive) ────────────────────────────
# Returns a list of edge dicts. Each carries:
#   rel, source, target, _source_text, tense?, time_phrase?, modifiers?
#       where modifiers = list of {rel, target} hyperedges attached to THIS
#       edge (committed by editor with source_id = this edge's id).
def _build_edge(verb_tok, subj_tok, sent_text: str, depth: int = 0) -> list[dict]:
    if depth >= _MAX_EDGE_DEPTH:
        return []

    # Passive-to-active normalization: detect and flip before building nodes.
    rewrite = _passive_rewrite(verb_tok, subj_tok)
    if rewrite is not None and rewrite[0] == "invert":
        # Full inversion: agent acts on patient. Recurse with swapped roles.
        _, agent_tok, patient_tok = rewrite
        # Build the active-voice edge: agent -verb-> patient.
        subj_node = _build_node(agent_tok)
        obj_node = _build_node(patient_tok)
        rel = verb_tok.lemma_.lower()
        for c in verb_tok.children:
            if c.dep_ == "neg": rel = f"not_{rel}"; break
        tense = _detect_tense(verb_tok)
        time_phrase = _detect_time_phrase(verb_tok)
        clause_text = " ".join(t.text for t in verb_tok.subtree).strip()
        primary: dict = {
            "type": "edge", "rel": rel,
            "source": subj_node, "target": obj_node,
            "_source_text": sent_text, "_clause_text": clause_text,
        }
        if tense is not None:       primary["tense"] = tense
        if time_phrase is not None: primary["time_phrase"] = time_phrase
        return [primary]

    subj_node = _build_node(subj_tok)
    rel = verb_tok.lemma_.lower()

    # Agentless passive: prefix rel to signal receptive framing.
    if rewrite is not None and rewrite[0] == "agentless":
        rel = f"receive_{rel}"

    for c in verb_tok.children:
        if c.dep_ == "neg": rel = f"not_{rel}"; break

    tense = _detect_tense(verb_tok)
    time_phrase = _detect_time_phrase(verb_tok)
    clause_text = " ".join(t.text for t in verb_tok.subtree).strip()

    # Direct object / attribute candidate.
    obj_tok = next(
        (c for c in verb_tok.children
         if c.dep_ in _OBJ_DEPS and c.pos_ in _VALUE_POS),
        None,
    )

    # Complement clause → hyperedge target (the inner edge IS the target).
    comp_target: dict | None = None
    for c in verb_tok.children:
        if c.dep_ in {"ccomp", "xcomp"} and c.pos_ in {"VERB", "AUX"}:
            nested_subj = next(
                (gc for gc in c.children
                 if gc.dep_ in _SUBJ_DEPS and gc.pos_ in _VALUE_POS),
                None,
            )
            if nested_subj is None and c.dep_ == "xcomp":
                nested_subj = obj_tok if obj_tok is not None else subj_tok
            if nested_subj:
                nested = _build_edge(c, nested_subj, sent_text, depth + 1)
                if nested:
                    comp_target = nested[0]
            break

    # Relative clause: head noun is the implicit gap object.
    if obj_tok is None and comp_target is None and verb_tok.dep_ == "relcl":
        head = verb_tok.head
        if head.pos_ in _VALUE_POS:
            obj_tok = head

    edges: list[dict] = []

    # Build amod modifier list — these become hyperedge-on-edge modifiers
    # attached to the primary edge after commit. Captured for both subj and
    # the (optional) obj noun token.
    def _amod_mods(tok) -> list[dict]:
        if tok is None: return []
        return [{"rel": "qualifies",
                 "target": {"type": "node", "text": c.lemma_.lower(), "pos": "ADJ"}}
                for c in tok.children if c.dep_ == "amod"]
    modifier_list = _amod_mods(subj_tok) + _amod_mods(obj_tok)

    # Possessive ownership: emit `owner -have-> noun` for every poss child of
    # subj or obj. spaCy marks these with dep_="poss"; the possessor is always
    # attached directly to the owned noun. No tense/time — structural relation.
    def _poss_edges(tok) -> list[dict]:
        if tok is None: return []
        result = []
        for c in tok.children:
            if c.dep_ == "poss" and c.pos_ in {"NOUN", "PROPN"}:
                result.append({
                    "type": "edge", "rel": "have",
                    "source": _build_node(c), "target": _build_node(tok),
                    "_source_text": sent_text, "_clause_text": clause_text,
                })
        return result
    edges.extend(_poss_edges(subj_tok))
    if obj_tok is not None:
        edges.extend(_poss_edges(obj_tok))

    # Primary edge: subj — rel — (target).
    target: dict | None = None
    if comp_target is not None:
        target = comp_target
    elif obj_tok is not None:
        target = _build_node(obj_tok)

    if target is not None:
        primary: dict = {
            "type": "edge", "rel": rel,
            "source": subj_node, "target": target,
            "_source_text": sent_text, "_clause_text": clause_text,
        }
        if tense is not None:       primary["tense"] = tense
        if time_phrase is not None: primary["time_phrase"] = time_phrase
        if modifier_list:           primary["modifiers"] = modifier_list
        edges.append(primary)
        # Conjunctions on the direct object: parallel edges sharing rel/subj.
        if obj_tok is not None and comp_target is None:
            for cj in obj_tok.children:
                if cj.dep_ == "conj" and cj.pos_ in _VALUE_POS:
                    conj_edge = {**primary, "target": _build_node(cj)}
                    edges.append(conj_edge)

    # Prepositional objects on the VERB: each prep becomes its own edge with
    # the preposition as the relation (subj —prep—> pobj). This avoids the
    # old "fused" found_in style relations that polluted the relation
    # vocabulary; the preposition is the relation, period.
    for child in verb_tok.children:
        if child.dep_ != "prep": continue
        for gc in child.children:
            if gc.dep_ != "pobj" or gc.pos_ not in _VALUE_POS: continue
            prep_edge: dict = {
                "type": "edge", "rel": child.lemma_.lower(),
                "source": subj_node, "target": _build_node(gc),
                "_source_text": sent_text, "_clause_text": clause_text,
            }
            if tense is not None:       prep_edge["tense"] = tense
            if time_phrase is not None: prep_edge["time_phrase"] = time_phrase
            edges.append(prep_edge)
            # Coordinated pobj split.
            for cj in gc.children:
                if cj.dep_ == "conj" and cj.pos_ in _VALUE_POS:
                    edges.append({**prep_edge, "target": _build_node(cj)})

    # Noun-level prep phrases (alliance OF states, attack ON one). These
    # attach at NODE level — they describe the concept, not the clause event.
    # No tense/time on these (structural relations between concepts).
    for noun_tok in (t for t in (subj_tok, obj_tok) if t is not None):
        for c in noun_tok.children:
            if c.dep_ != "prep": continue
            for gc in c.children:
                if gc.dep_ != "pobj" or gc.pos_ not in _VALUE_POS: continue
                edges.append({
                    "type": "edge", "rel": c.lemma_.lower(),
                    "source": _build_node(noun_tok), "target": _build_node(gc),
                    "_source_text": sent_text, "_clause_text": clause_text,
                })
                for cj in gc.children:
                    if cj.dep_ == "conj" and cj.pos_ in _VALUE_POS:
                        edges.append({
                            "type": "edge", "rel": c.lemma_.lower(),
                            "source": _build_node(noun_tok),
                            "target": _build_node(cj),
                            "_source_text": sent_text, "_clause_text": clause_text,
                        })

    # Pending fallback — fires whenever no PRIMARY edge was built (no direct
    # object, no comp target). Even when prep edges exist, the verb itself
    # carries signal worth preserving. The prep edges describe HOW/WHERE/WHEN
    # the event happened; the pending edge captures THAT it happened.
    has_primary = target is not None
    if not has_primary:
        pending: dict = {
            "type": "edge", "rel": rel,
            "source": subj_node,
            "target": {"type": "node", "text": "[event]", "pos": "X"},
            "_source_text": sent_text, "_clause_text": clause_text,
            "_pending_completion": True,
        }
        if tense is not None:       pending["tense"] = tense
        if time_phrase is not None: pending["time_phrase"] = time_phrase
        if modifier_list:           pending["modifiers"] = modifier_list
        edges.append(pending)

    return edges


# ── LLM zero-extraction fallback ─────────────────────────────────────────────
# Fires when spaCy extracts nothing from a sentence that looks like real prose.
# Typical cause: parser quality (en_core_web_sm misclassifies verb POS/dep).
# Gated by DI_LLM_EXTRACTION (default "1"); disable for pure-spaCy pipelines.

def _looks_like_sentence(sent) -> bool:
    text = sent.text if hasattr(sent, "text") else str(sent)
    words = text.split()
    if len(words) < 3: return False
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / len(text) < 0.5: return False
    if text[:1] in ("[", "<", "#"): return False
    # Need at least 2 nouns for any SVO structure to exist.
    # This filters NP-only sentences ("The big red dog.") that have no verb
    # and no plausible object, while keeping coordination misparses like
    # "Bees and wasps both sting predators." (3 nouns).
    noun_count = sum(1 for t in sent if t.pos_ in {"NOUN", "PROPN"}) if hasattr(sent, "__iter__") else 0
    return noun_count >= 2


# POS tags that disqualify a relation word — 1B models frequently emit
# determiners ("both"), prepositions ("in"), or conjunctions ("and") as
# relations when the parser missed the true verb.
_NON_VERB_POS = frozenset({"DET", "ADP", "CCONJ", "SCONJ", "ADV", "PRON",
                            "PART", "PUNCT", "SPACE", "NUM", "INTJ"})


def _is_valid_relation(rel: str) -> bool:
    # Checks the root word (before any underscore) via the already-loaded model.
    # Single-token lookup — negligible cost.
    root_word = rel.split("_")[0]
    tok = get_nlp()(root_word)[0]
    return tok.pos_ not in _NON_VERB_POS


def _llm_extract_clauses(sent_text: str) -> list[dict]:
    from llm import call_json
    try:
        data = call_json(
            f"Extract subject-verb-object triples from this sentence.\n"
            f'Output ONLY JSON: {{"triples": [{{"subject": "noun phrase", "relation": "verb lemma", "object": "noun phrase"}}]}}\n'
            f"Rules: lowercase throughout; relation MUST be an action verb lemma "
            f"(e.g. sting, eat, cause, support, establish) — never a determiner "
            f"(both, all), preposition (in, on), or adverb; "
            f"omit triples you are not confident about; return [] if no clear SVO.\n\n"
            f'Sentence: "{sent_text}"',
            model="1B", role="llm_extraction",
        )
    except Exception as exc:
        LOGGER.warning("LLM clause extraction failed: %s", exc)
        return []
    triples = data.get("triples", []) if isinstance(data, dict) else []
    results: list[dict] = []
    for t in triples:
        subj = str(t.get("subject", "")).strip().lower()[:80]
        rel  = str(t.get("relation", "")).strip().lower().replace(" ", "_")[:40]
        obj  = str(t.get("object",  "")).strip().lower()[:80]
        if subj and rel and obj and _is_valid_relation(rel):
            results.append({
                "type": "edge", "rel": rel,
                "source": {"type": "node", "text": subj, "pos": "NOUN"},
                "target": {"type": "node", "text": obj,  "pos": "NOUN"},
                "_source_text": sent_text, "_clause_text": sent_text,
            })
    return results


def _sent_to_clauses(sent, sent_text: str) -> list[dict]:
    results: list[dict] = []
    root_tok = next((t for t in sent if t.dep_ == "ROOT"), None)
    # A sentence is imperative only when it has NO nsubj/nsubjpass at all.
    # Previously this checked pos_ in _VALUE_POS, which mis-fired on titles like
    # "secretary general" where spaCy may tag "general" as ADJ.
    sent_is_imperative = root_tok is not None and not any(
        c.dep_ in _SUBJ_DEPS for c in root_tok.children
    )
    for token in sent:
        if token.pos_ not in {"VERB", "AUX"}: continue
        if token.dep_ not in _CLAUSE_DEPS: continue
        subj = next(
            (c for c in token.children
             if c.dep_ in _SUBJ_DEPS and c.pos_ in _VALUE_POS),
            None,
        )
        # Fallback 1: any nsubj/nsubjpass regardless of POS (ADJ-tagged titles).
        if subj is None:
            subj = next((c for c in token.children if c.dep_ in _SUBJ_DEPS), None)
        # Fallback 2: pre-verbal nmod NOUN — spaCy sometimes attaches the
        # subject noun phrase as nmod on the verb instead of nsubj (e.g.
        # "The secretary general chairs X" → secretary nmod, general amod).
        # Reconstruct the full title-phrase from co-located siblings, then
        # wrap it in the same SimpleNamespace sentinel the imperative path uses
        # so _build_node receives a token-like object with the combined text.
        if subj is None:
            pre_noun = next(
                (c for c in token.children
                 if c.dep_ == "nmod" and c.pos_ in _VALUE_POS and c.i < token.i),
                None,
            )
            if pre_noun is not None:
                span = sorted(
                    [pre_noun] + [
                        t for t in token.children
                        if t.dep_ in {"amod", "compound"}
                        and t.pos_ in _VALUE_POS | {"ADJ"}
                        and pre_noun.i <= t.i < token.i
                    ],
                    key=lambda t: t.i,
                )
                phrase = " ".join(t.lemma_.lower() for t in span)
                subj = SimpleNamespace(
                    lemma_=phrase, lower_=phrase, pos_="NOUN", dep_="nsubj",
                    children=[], i=-1, ent_type_="", ent_iob_="O", doc=None,
                )
        if subj is None:
            if sent_is_imperative: subj = _IMPLICIT_SYSTEM_SUBJ
            else: continue
        all_subjs = [subj] + [
            c for c in subj.children if c.dep_ == "conj" and c.pos_ in _VALUE_POS
        ]
        for subj_tok in all_subjs:
            results.extend(_build_edge(token, subj_tok, sent_text))
    return results


# ── Public extraction API ────────────────────────────────────────────────────
def extract_clauses(text: str, *, llm_fallback: bool = True) -> typing.Iterator[dict]:
    # Yield compositional SVO hyperedge dicts from text via spaCy.
    # Passive constructions are normalized to active voice (agent acts on patient).
    # llm_fallback: when True (default), sentences that produce zero spaCy clauses
    # but look like real prose are passed to the 1B LLM for SVO extraction.
    # Pass False when the caller already has its own LLM pass (e.g. deep_ingest).
    try: text = resolve_basic_pronouns(text)
    except Exception as exc:
        LOGGER.warning("coref pre-pass failed: %s", exc)
    nlp = get_nlp()
    _llm_enabled = llm_fallback and os.getenv("DI_LLM_EXTRACTION", "1") == "1"
    chunks = [c.strip()[:5000] for c in text.split("\n") if c.strip()]
    for doc in nlp.pipe(chunks, batch_size=50):
        for span in doc.sents:
            sent_text = span.text.strip()
            if not sent_text: continue
            facts = _sent_to_clauses(span, sent_text)
            if not facts and _llm_enabled and _looks_like_sentence(span):
                facts = _llm_extract_clauses(sent_text)
            yield from facts
