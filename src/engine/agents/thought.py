import re; from engine.common.constants import (ARGUMENT_SOURCE_DIVERSITY_MIN_MULTIPLIER, ARGUMENT_SOURCE_DIVERSITY_TARGET, MAX_CONTEXT_SAMPLES, STARTING_SCORE,); from engine.extract.noise_cleanup import is_usable_clause_text; from engine.extract.word_info_map import existing_concept_index, literal_from_index, text_similarity; from engine.agents.connection import ConnectionEndpoint
from engine.agents.info import Info_Agent; from engine.extract.target_text import distinctive_target_tokens, target_acronym_tokens, target_tokens; from engine.common.shm import display_source; PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+\b"); NUMBER_RE = re.compile(r"\d"); SPECIFIC_DETAIL_RE = re.compile(r"\d[\d,./:-]*%?|%"); _MATCH_SCORE_CACHE = {}; _LEXICAL_MATCH_CACHE = {}
_SPECIFIC_CONTEXT_CACHE = {}
# Clears thought caches caches or state.
def clear_thought_caches():
    _MATCH_SCORE_CACHE.clear(); _LEXICAL_MATCH_CACHE.clear(); _SPECIFIC_CONTEXT_CACHE.clear()
# Coerces optional int for thought evaluation.
def _optional_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
# Clamps clamp01 for thought evaluation.
def _clamp01(value):
    return max(0.0, min(1.0, float(value)))
# Cleans text for thought evaluation.
def _clean_text(text):
    return " ".join(str(text or "").strip().split())
# Tracks one candidate reasoning path through the graph and its supporting evidence.
class Thought:
    # Initializes a thought path with target labels, success queries, and empty evidence stores.
    def __init__(self, current_asu: Info_Agent, seed_query: str = "", start_target: str = "", target_b: str = "", success_queries=None, goal_agents=None,):
        self.current_asu = current_asu; self.seed_query = _clean_text(seed_query); self.start_target = _clean_text(start_target); self.target_b = _clean_text(target_b); self.success_queries = [_clean_text(query) for query in list(success_queries or []) if _clean_text(query)]; self.goal_agents = list(goal_agents or []); self.reset(current_asu)
    # Resets path state around a new starting agent while preserving route configuration.
    def reset(self, start_asu: Info_Agent | None = None):
        if start_asu is not None:
            self.current_asu = start_asu
        self.last_asu_idx = -1; self.alive = True; self.successful = False; self.termination_reason = ""; self.score = STARTING_SCORE; self.success_payload = None; self.collected_notes = []; self.collected_adjacent_notes = []; self.collected_sources = []; self._agent_cited_score_cache = {}; self._connector_cited_score_cache = {}
        self.history = [{"node": self.current_asu.ASU, "seed_query": self.seed_query}]
    # Converts a connector into the next thought-hop record.
    def _candidate_hop(self, conn):
        source_agent = conn.source_agent; target = conn.target
        if source_agent is None or target is None:
            return None
        if source_agent.index == self.current_asu.index:
            return (conn, target, False)
        if target.index == self.current_asu.index:
            return (conn, source_agent, True)
        return None
    # Returns whether source is citable for thought evaluation.
    def _source_is_citable(self, source):
        source = _clean_text(source); display = display_source(source)
        return bool(display and display != "unknown" and ("|" in source or "://" in display or "/" in display or "." in display))
    # Builds context records for agent for thought evaluation.
    def _context_records_for_agent(self, agent):
        if agent is None:
            return []
        concept_id = existing_concept_index(agent.ASU)
        if concept_id < 0:
            return []
        return ConnectionEndpoint.contexts_from_idx(concept_id)
    # Returns whether a context record is specific and sourced enough to cite.
    def _should_cite(self, record, utility=0.0):
        source = _clean_text((record or {}).get("source", ""))
        if not self._source_is_citable(source):
            return False
        text = _clean_text((record or {}).get("text", ""))
        if NUMBER_RE.search(text):
            return True
        if PROPER_NOUN_RE.search(text):
            return True
        return utility >= 0.9
    # Converts a context record into a citable detail payload.
    def _detail_from_record(self, record, utility=0.0):
        text = _clean_text((record or {}).get("text", "")); source = _clean_text((record or {}).get("source", "unknown"))
        if not text:
            return ""
        if not is_usable_clause_text(text):
            return ""
        if self._should_cite(record, utility=utility):
            if source not in self.collected_sources:
                self.collected_sources.append(source)
            return f"{text} [{display_source(source)}]"
        return text
    # Stores remember note for thought evaluation.
    def _remember_note(self, text, adjacent=False):
        clean = _clean_text(text)
        if not clean or clean in self.collected_notes:
            if not adjacent:
                return
        if adjacent:
            if clean and clean not in self.collected_adjacent_notes:
                self.collected_adjacent_notes.append(clean)
                if len(self.collected_adjacent_notes) > 20:
                    del self.collected_adjacent_notes[:-20]
            return
        self.collected_notes.append(clean)
        if len(self.collected_notes) > 40:
            del self.collected_notes[:-40]
    # Collects citable context around the current thought endpoint.
    def gather_stop_details(self):
        records = self._context_records_for_agent(self.current_asu)
        for record in records[:MAX_CONTEXT_SAMPLES]:
            detail = self._detail_from_record(record)
            if detail:
                self._remember_note(detail)
    # Builds a stable identity tuple for a connector.
    def _connector_identity(self, conn):
        source_agent = getattr(conn, "source_agent", None); target = getattr(conn, "target", None)
        return (getattr(source_agent, "index", -1), getattr(target, "index", -1), getattr(conn, "relation_index", -1), _clean_text(getattr(conn, "evidence_text", "")),)
    # Returns predicate text for thought evaluation.
    def _predicate_text(self, conn):
        target = getattr(conn, "target", None)
        if target is None:
            return ""
        return _clean_text(getattr(target, "ASU", ""))
    # Returns specific value key for thought evaluation.
    def _specific_value_key(self, value):
        if isinstance(value, dict):
            return repr(sorted(value.items()))
        return _clean_text(value).lower()
    # Builds structured specific values for thought evaluation.
    def _structured_specific_values(self, conn):
        values = []; seen = set()
        for attr in ("predicate_specifics", "connection_specifics", "subject_specifics"):
            for value in list(getattr(conn, attr, []) or []):
                key = self._specific_value_key(value)
                if not key or key in seen:
                    continue
                seen.add(key); values.append(value)
        return values
    # Builds predicate specific values for thought evaluation.
    def _predicate_specific_values(self, conn):
        values = []; seen = set()
        for value in list(getattr(conn, "predicate_specifics", []) or []):
            key = self._specific_value_key(value)
            if not key or key in seen:
                continue
            seen.add(key); values.append(value)
        for value in SPECIFIC_DETAIL_RE.findall(self._predicate_text(conn)):
            key = self._specific_value_key(value)
            if not key or key in seen:
                continue
            seen.add(key); values.append(value)
        return values
    # Returns whether predicate has specific data for thought evaluation.
    def _predicate_has_specific_data(self, conn):
        return bool(self._predicate_specific_values(conn))
    # Returns whether connection has specific data for thought evaluation.
    def _connection_has_specific_data(self, conn, clause=""):
        if self._predicate_has_specific_data(conn):
            return True
        if self._structured_specific_values(conn):
            return True
        return bool(SPECIFIC_DETAIL_RE.search(_clean_text(clause)))
    # Returns specificity key for thought evaluation.
    def _specificity_key(self, conn, clause):
        predicate_values = self._predicate_specific_values(conn); structured_values = self._structured_specific_values(conn); clause_values = {self._specific_value_key(value) for value in SPECIFIC_DETAIL_RE.findall(clause) if self._specific_value_key(value)}
        return (len(predicate_values), len(structured_values), len(clause_values), len(_clean_text(clause)),)
    # Builds agent specific context details for thought evaluation.
    def _agent_specific_context_details(self):
        connectors = list(getattr(self.current_asu, "connectors", []) or []); cache_key = (getattr(self.current_asu, "index", -1), len(connectors)); cached = _SPECIFIC_CONTEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        details = []
        for conn in connectors:
            if self._candidate_hop(conn) is None:
                continue
            source = _clean_text(getattr(conn, "source", ""))
            if not self._source_is_citable(source):
                continue
            clause = self._connection_clause_text(conn)
            if not clause:
                continue
            if not is_usable_clause_text(clause):
                continue
            if not self._connection_has_specific_data(conn, clause):
                continue
            details.append({"identity": self._connector_identity(conn), "text": clause, "source": source, "specificity": self._specificity_key(conn, clause), "subject_specifics": list(getattr(conn, "subject_specifics", []) or []), "predicate_specifics": list(getattr(conn, "predicate_specifics", []) or []), "connection_specifics": list(getattr(conn, "connection_specifics", []) or []),})
        details.sort(key=lambda item: (item["specificity"], item["text"]), reverse=True); _SPECIFIC_CONTEXT_CACHE[cache_key] = details
        return details
    # Returns connection clause text for thought evaluation.
    def _connection_clause_text(self, conn):
        evidence_text = _clean_text(getattr(conn, "evidence_text", ""))
        if evidence_text:
            return evidence_text.rstrip(".") + "."
        subject_agent = getattr(conn, "source_agent", None); subject = _clean_text(getattr(subject_agent, "ASU", "")); predicate = self._predicate_text(conn); relation = literal_from_index(_optional_int(getattr(conn, "relation_index", -1))) or ""; relation = _clean_text(relation)
        if not subject or not relation or not predicate:
            return ""
        return f"{subject} {relation} {predicate}".strip().rstrip(".") + "."
    # Builds specific context details for thought evaluation.
    def _specific_context_details(self, chosen_conn, limit=3):
        chosen_identity = self._connector_identity(chosen_conn); details = []
        for detail in self._agent_specific_context_details():
            if detail.get("identity") == chosen_identity:
                continue
            clean_detail = dict(detail); clean_detail.pop("identity", None); details.append(clean_detail)
            if len(details) >= limit:
                break
        return details[:limit]
    # Returns target tokens for target matching.
    def _target_tokens(self, text):
        return target_tokens(_clean_text(text), min_length=1)
    # Returns target token sets for target matching.
    def _target_token_sets(self, candidate, target):
        candidate_base = set(self._target_tokens(candidate)); target_base = set(self._target_tokens(target)); candidate_tokens = candidate_base | (set(target_acronym_tokens(candidate)) & target_base); target_tokens = target_base | (set(target_acronym_tokens(target)) & candidate_base)
        return candidate_tokens, target_tokens
    # Returns lexical match for thought evaluation.
    def _lexical_match(self, candidate, target):
        candidate = _clean_text(candidate); target = _clean_text(target); key = (candidate.lower(), target.lower()); cached = _LEXICAL_MATCH_CACHE.get(key)
        if cached is not None:
            return cached
        candidate_tokens, target_tokens = self._target_token_sets(candidate, target)
        if not candidate_tokens or not target_tokens:
            _LEXICAL_MATCH_CACHE[key] = 0.0
            return 0.0
        overlap = candidate_tokens & target_tokens; distinctive_tokens = distinctive_target_tokens(target_tokens)
        if distinctive_tokens and not (overlap & distinctive_tokens):
            _LEXICAL_MATCH_CACHE[key] = 0.0
            return 0.0
        if not distinctive_tokens and not overlap:
            _LEXICAL_MATCH_CACHE[key] = 0.0
            return 0.0
        precision = len(overlap) / len(candidate_tokens); recall = len(overlap) / len(target_tokens); distinctive_coverage = (len(overlap & distinctive_tokens) / len(distinctive_tokens) if distinctive_tokens else 0.0); score = max(precision, 0.85 * recall, distinctive_coverage); _LEXICAL_MATCH_CACHE[key] = score
        return score
    # Returns grounding lexical match for thought evaluation.
    def _grounding_lexical_match(self, candidate, target):
        candidate_tokens, target_tokens = self._target_token_sets(candidate, target); distinctive_tokens = distinctive_target_tokens(target_tokens); overlap = candidate_tokens & target_tokens
        if distinctive_tokens and not (overlap & distinctive_tokens):
            return 0.0
        if not distinctive_tokens and not overlap:
            return 0.0
        return self._lexical_match(candidate, target)
    # Returns candidate match score for thought evaluation.
    def _candidate_match_score(self, candidate, target, semantic=True):
        candidate = _clean_text(candidate); target = _clean_text(target)
        if not candidate or not target:
            return 0.0
        key = (candidate.lower(), target.lower(), bool(semantic)); cached = _MATCH_SCORE_CACHE.get(key)
        if cached is not None:
            return cached
        lexical = self._lexical_match(candidate, target); score = lexical
        if semantic:
            score = max(score, _clamp01(text_similarity(candidate, target)))
        _MATCH_SCORE_CACHE[key] = score
        return score
    # Returns best lexical target match for thought evaluation.
    def _best_lexical_target_match(self, candidates, target, related_queries=None):
        target = _clean_text(target)
        if not target:
            return 0.0
        related_queries = [_clean_text(query) for query in list(related_queries or []) if _clean_text(query)]; candidate_texts = [_clean_text(candidate) for candidate in list(candidates or []) if _clean_text(candidate)]
        if not candidate_texts:
            return 0.0
        best = 0.0
        for candidate in candidate_texts:
            best = max(best, self._grounding_lexical_match(candidate, target))
            for query in related_queries:
                best = max(best, 0.92 * self._grounding_lexical_match(candidate, query))
        return best
    # Returns best target match for thought evaluation.
    def _best_target_match(self, candidates, target, related_queries=None, semantic=True):
        target = _clean_text(target)
        if not target:
            return 0.0
        related_queries = [_clean_text(query) for query in list(related_queries or []) if _clean_text(query)]; candidate_texts = [_clean_text(candidate) for candidate in list(candidates or []) if _clean_text(candidate)]
        if not candidate_texts:
            return 0.0
        best = 0.0
        for candidate in candidate_texts:
            best = max(best, self._candidate_match_score(candidate, target, semantic=semantic))
            for query in related_queries:
                best = max(best, 0.92 * self._candidate_match_score(candidate, query, semantic=semantic))
        return best
    # Returns step clause for thought evaluation.
    def _step_clause(self, step):
        if not isinstance(step, dict):
            return ""
        evidence_text = _clean_text(step.get("evidence_text", ""))
        if evidence_text:
            return evidence_text.rstrip(".") + "."
        subject = _clean_text(step.get("subject", "")); predicate = _clean_text(step.get("predicate", "")); relation_id = step.get("relation_id", -1); relation = literal_from_index(_optional_int(relation_id, default=-1)) or ""; relation = _clean_text(relation)
        if not subject or not relation or not predicate:
            return ""
        return f"{subject} {relation} {predicate}".strip().rstrip(".") + "."
    # Returns step specific records used by thought evaluation.
    def _step_specific_records(self, step):
        if not isinstance(step, dict):
            return []
        records = []
        for detail in list(step.get("specific_details", []) or []):
            if not isinstance(detail, dict):
                continue
            text = _clean_text(detail.get("text", ""))
            if not text:
                continue
            if not is_usable_clause_text(text):
                continue
            records.append({"text": text.rstrip(".") + ".", "source": _clean_text(detail.get("source", "")), "specificity": list(detail.get("specificity", ()) or ()), "role": "specific_context",})
        return records
    # Returns path texts for thought evaluation.
    def _path_texts(self):
        texts = []; seen = set()
        # Returns append text for thought evaluation.
        def append_text(value):
            clean = _clean_text(value)
            if not clean or clean in seen:
                return
            seen.add(clean); texts.append(clean)
        append_text(self.history[0].get("node", "")); append_text(self.current_asu.ASU)
        for step in self.history[1:]:
            append_text(step.get("subject", "")); append_text(step.get("subject_modifier", "")); append_text(f"{step.get('subject', '')} {step.get('subject_modifier', '')}"); append_text(step.get("predicate", "")); append_text(step.get("predicate_modifier", "")); append_text(f"{step.get('predicate', '')} {step.get('predicate_modifier', '')}")
            append_text(step.get("evidence_text", "")); append_text(self._step_clause(step))
            for record in self._step_specific_records(step):
                append_text(record.get("text", ""))
        return texts
    # Returns path target matches for thought evaluation.
    def _path_target_matches(self):
        path_texts = self._path_texts(); target_a_match = self._best_target_match(path_texts, self.start_target); target_b_match = self._best_target_match(path_texts, self.target_b, self.success_queries)
        return target_a_match, target_b_match
    # Returns path target lexical matches for thought evaluation.
    def _path_target_lexical_matches(self):
        path_texts = self._path_texts(); target_a_match = self._best_lexical_target_match(path_texts, self.start_target); target_b_match = self._best_lexical_target_match(path_texts, self.target_b, self.success_queries)
        return target_a_match, target_b_match
    # Returns a human label for support strength and target coverage.
    def _confidence_label_for_score(self, support_score, target_a_match, target_b_match):
        blended = ((float(support_score or 0.0) / 100.0) + target_a_match + target_b_match) / 3.0
        if blended >= 0.72:
            return "strong"
        if blended >= 0.54:
            return "moderate"
        return "tentative"
    # Returns source key for thought evaluation.
    def _source_key(self, source):
        display = display_source(source).strip()
        if not display or display == "unknown":
            return ""
        return display.rstrip("/").lower()
    # Returns source diversity stats for thought evaluation.
    def _source_diversity_stats(self, records):
        source_counts = {}
        for record in list(records or []):
            source = _clean_text((record or {}).get("source", ""))
            if not self._source_is_citable(source):
                continue
            key = self._source_key(source)
            if not key:
                continue
            source_counts[key] = source_counts.get(key, 0) + 1
        total = sum(source_counts.values()); distinct = len(source_counts)
        if total <= 0 or distinct <= 0:
            return {"distinct_urls": 0, "cited_records": 0, "dominant_url_share": 0.0, "source_diversity_multiplier": ARGUMENT_SOURCE_DIVERSITY_MIN_MULTIPLIER,}
        dominant_share = max(source_counts.values()) / total; target = max(1, int(ARGUMENT_SOURCE_DIVERSITY_TARGET)); floor = _clamp01(ARGUMENT_SOURCE_DIVERSITY_MIN_MULTIPLIER); distinct_multiplier = min(1.0, distinct / target); concentration_multiplier = max(floor, 1.0 - max(0.0, dominant_share - (1.0 / target)))
        multiplier = max(floor, min(distinct_multiplier, concentration_multiplier))
        return {"distinct_urls": distinct, "cited_records": total, "dominant_url_share": round(dominant_share, 3), "source_diversity_multiplier": round(multiplier, 3),}
    # Builds the final payload describing how one path links the two targets.
    def _build_relationship_statement(self):
        start = self.start_target or self.seed_query or self.history[0].get("node", ""); end = self.target_b or self.current_asu.ASU; path_clauses = []; clause_records = []; seen_path_clauses = set(); seen_records = set()
        # Returns record key for thought evaluation.
        def record_key(record):
            text = _clean_text((record or {}).get("text", "")).lower(); source = display_source((record or {}).get("source", "")).rstrip("/").lower(); role = _clean_text((record or {}).get("role", "path")).lower() or "path"
            return (role, text, source)
        # Builds add record for thought evaluation.
        def add_record(record):
            key = record_key(record)
            if not key[1] or key in seen_records:
                return
            seen_records.add(key); clause_records.append(record)
        for step in self.history[1:]:
            clause = self._step_clause(step)
            if clause:
                clause_key = _clean_text(clause).lower()
                if clause_key not in seen_path_clauses:
                    seen_path_clauses.add(clause_key); path_clauses.append(clause)
                add_record({"text": clause, "source": _clean_text(step.get("source", "")), "role": "path",})
            for record in self._step_specific_records(step):
                add_record(record)
        recent_records = clause_records[-(6 + (6 * 3)):]; supporting_clauses = [record["text"] for record in recent_records if record.get("role") == "specific_context"]; target_a_match, target_b_match = self._path_target_matches(); target_a_lexical, target_b_lexical = self._path_target_lexical_matches()
        summary = " ".join(path_clauses[-3:]) if path_clauses else f"{start} -> {end}"; source_diversity = self._source_diversity_stats(recent_records); adjusted_score = round(float(self.score) * float(source_diversity["source_diversity_multiplier"]), 2,)
        return {"statement": summary, "support_score": adjusted_score, "raw_support_score": round(self.score, 2), "confidence": self._confidence_label_for_score(adjusted_score, target_a_match, target_b_match), "match_target_a": round(target_a_match, 3), "match_target_b": round(target_b_match, 3), "lexical_target_a": round(target_a_lexical, 3), "lexical_target_b": round(target_b_lexical, 3), "source_diversity": source_diversity, "path_clauses": path_clauses[-6:], "supporting_clauses": supporting_clauses, "clause_records": recent_records, "endpoint": self.current_asu.ASU, "sources": list(self.collected_sources), "notes": list(self.collected_notes), "target_a": self.start_target or start, "target_b": self.target_b or end, "direction": getattr(self, "route_id", ""), "route": (f"{getattr(self, 'route_start_label', '')}->{getattr(self, 'route_goal_label', '')}" if getattr(self, "route_start_label", "") or getattr(self, "route_goal_label", "") else ""),}
