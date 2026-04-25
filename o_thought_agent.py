import random
import re

import numpy as np  # type: ignore

from constants import (
    ARGUMENT_SOURCE_DIVERSITY_MIN_MULTIPLIER,
    ARGUMENT_SOURCE_DIVERSITY_TARGET,
    LOW_SCORE_THRESHOLD,
    MAX_ADJACENT_SAMPLES,
    MAX_CONTEXT_SAMPLES,
    MAX_THOUGHT_HOPS,
    MIN_SUCCESS_STEPS,
    PATH_TARGET_MATCH_THRESHOLD,
    SPECIFIC_DETAIL_REWARD_RATE,
    SPECIFIC_DETAIL_REWARD_SATURATION,
    SPATIAL_PROGRESS_REWARD_RATE,
    STARTING_SCORE,
    STOPWORD_TOKENS,
    SUCCESS_SIMILARITY_THRESHOLD,
    THOUGHT_CITED_FACT_WEIGHT,
    THOUGHT_CITED_FACT_WEIGHT_CAP,
    THOUGHT_DESTINATION_CITED_FACT_WEIGHT,
    THOUGHT_SEMANTIC_CHECK_INTERVAL,
    THOUGHT_MIN_STEP_SCORE_MULTIPLIER,
    THOUGHT_TENSE_OPPOSITION_PENALTY,
    THOUGHT_TENSE_PREFERENCE_WEIGHT,
    THOUGHT_UTILITY_PENALTY_RATE,
    UTILITY_FLOOR,
)
from d_word_info_map import existing_concept_index, literal_from_index, text_similarity
from o_connection import ConnectionEndpoint
from o_info_agent import ASU_Agent
from utils import display_source

TARGET_TOKEN_RE = re.compile(r"[a-z0-9]+")

PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+\b")
NUMBER_RE = re.compile(r"\d")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
SPECIFIC_DETAIL_RE = re.compile(r"\d[\d,./:-]*%?|%")

_MATCH_SCORE_CACHE = {}
_LEXICAL_MATCH_CACHE = {}
_CONNECTOR_FACT_SCORE_CACHE = {}
_AGENT_FACT_SCORE_CACHE = {}
_SPECIFIC_CONTEXT_CACHE = {}
_CANDIDATE_HOP_CACHE = {}


def clear_thought_caches():
    _MATCH_SCORE_CACHE.clear()
    _LEXICAL_MATCH_CACHE.clear()
    _CONNECTOR_FACT_SCORE_CACHE.clear()
    _AGENT_FACT_SCORE_CACHE.clear()
    _SPECIFIC_CONTEXT_CACHE.clear()
    _CANDIDATE_HOP_CACHE.clear()


def _optional_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _clean_text(text):
    return " ".join(str(text or "").strip().split())


class Thought:
    def __init__(
        self,
        current_asu: ASU_Agent,
        seed_query: str = "",
        start_target: str = "",
        target_b: str = "",
        success_queries=None,
        goal_agents=None,
        tense_preference="none",
    ):
        self.current_asu = current_asu
        self.seed_query = _clean_text(seed_query)
        self.start_target = _clean_text(start_target)
        self.target_b = _clean_text(target_b)
        self.success_queries = [
            _clean_text(query)
            for query in list(success_queries or [])
            if _clean_text(query)
        ]
        self.goal_agents = list(goal_agents or [])
        self.tense_preference = self._normalize_tense_preference(tense_preference)
        self.reset(current_asu)

    def reset(self, start_asu: ASU_Agent | None = None):
        if start_asu is not None:
            self.current_asu = start_asu
        self.last_asu_idx = -1
        self.alive = True
        self.successful = False
        self.termination_reason = ""
        self.score = STARTING_SCORE
        self.success_payload = None
        self.collected_notes = []
        self.collected_adjacent_notes = []
        self.collected_sources = []
        self._agent_cited_score_cache = {}
        self._connector_cited_score_cache = {}
        self.history = [{"node": self.current_asu.ASU, "seed_query": self.seed_query}]

    def _hop_parts(self, hop):
        if isinstance(hop, tuple) and len(hop) == 3:
            return hop
        conn = hop
        if conn is None:
            return None, None, False
        source_agent = conn.source_agent
        target = conn.target
        if source_agent is not None and source_agent.index == self.current_asu.index:
            return conn, target, False
        if target is not None and target.index == self.current_asu.index:
            return conn, source_agent, True
        return conn, target, False

    @staticmethod
    def _normalize_tense_preference(value):
        clean = _clean_text(value).lower()
        return clean if clean in {"past", "future"} else "none"

    def _preferred_tense_score(self):
        if self.tense_preference == "past":
            return -1.0
        if self.tense_preference == "future":
            return 1.0
        return 0.0

    def _candidate_hop(self, conn):
        source_agent = conn.source_agent
        target = conn.target
        if source_agent is None or target is None:
            return None
        if source_agent.index == self.current_asu.index:
            return (conn, target, False)
        if target.index == self.current_asu.index:
            return (conn, source_agent, True)
        return None

    def _candidate_hops_for_current(self):
        connections = list(getattr(self.current_asu, "connectors", []) or [])
        if not connections:
            return [], []

        cache_key = (getattr(self.current_asu, "index", -1), len(connections))
        cached = _CANDIDATE_HOP_CACHE.get(cache_key)
        if cached is not None:
            return cached

        current_pos = self.current_asu.pos
        hops = []
        distances = []
        for conn in connections:
            hop = self._candidate_hop(conn)
            if hop is None:
                continue
            _conn, next_agent, _reversed_hop = hop
            if next_agent is None or next_agent.index == self.current_asu.index:
                continue
            hops.append(hop)
            distances.append(self.distance_value(next_agent, current_pos))

        cached = (hops, distances)
        _CANDIDATE_HOP_CACHE[cache_key] = cached
        return cached

    def distance_value(self, next_agent, current_pos):
        if next_agent is None:
            return float("inf")
        return float(np.linalg.norm(current_pos - next_agent.pos))

    def relative_distance_weights(self, distances):
        weights = [0.0 for _ in distances]
        ranked = sorted(
            (float(distance), idx)
            for idx, distance in enumerate(distances)
            if np.isfinite(float(distance))
        )
        if not ranked:
            return [1.0 for _ in distances]

        candidate_count = len(ranked)
        for rank, (_distance, idx) in enumerate(ranked):
            weights[idx] = float(candidate_count - rank)
        return weights

    def _source_is_citable(self, source):
        source = _clean_text(source)
        display = display_source(source)
        return bool(
            display
            and display != "unknown"
            and ("|" in source or "://" in display or "/" in display or "." in display)
        )

    def _specific_text_score(self, text):
        text = _clean_text(text)
        if not text:
            return 0.0
        number_score = len(SPECIFIC_DETAIL_RE.findall(text)) * 2.0
        acronym_score = len(ACRONYM_RE.findall(text)) * 1.5
        proper_score = len(PROPER_NOUN_RE.findall(text)) * 0.75
        return number_score + acronym_score + proper_score

    def _cited_record_score(self, record, utility=0.0):
        if not self._source_is_citable((record or {}).get("source", "")):
            return 0.0
        text = _clean_text((record or {}).get("text", ""))
        score = self._specific_text_score(text)
        if self._should_cite(record, utility=utility):
            score += 1.0
        return score

    def _connector_cited_fact_score(self, conn):
        key = self._connector_identity(conn)
        cached = _CONNECTOR_FACT_SCORE_CACHE.get(key)
        if cached is not None:
            return cached
        cached = self._connector_cited_score_cache.get(key)
        if cached is not None:
            return cached

        if not self._source_is_citable(getattr(conn, "source", "")):
            _CONNECTOR_FACT_SCORE_CACHE[key] = 0.0
            self._connector_cited_score_cache[key] = 0.0
            return 0.0

        clause = self._connection_clause_text(conn)
        score = self._specific_text_score(clause)

        specificity = self._specificity_key(conn, clause)
        for value in specificity[:3]:
            try:
                score += max(0, int(value))
            except (TypeError, ValueError):
                continue

        predicate_values = self._predicate_specific_values(conn)
        score += len(predicate_values) * 2.0
        _CONNECTOR_FACT_SCORE_CACHE[key] = score
        self._connector_cited_score_cache[key] = score
        return score

    def _agent_cited_fact_score(self, agent):
        key = getattr(agent, "index", None)
        cached = _AGENT_FACT_SCORE_CACHE.get(key)
        if cached is not None:
            return cached
        cached = self._agent_cited_score_cache.get(key)
        if cached is not None:
            return cached

        records = self._context_records_for_agent(agent)
        if not records:
            _AGENT_FACT_SCORE_CACHE[key] = 0.0
            self._agent_cited_score_cache[key] = 0.0
            return 0.0
        scores = sorted(
            (
                self._cited_record_score(record)
                for record in records[:MAX_CONTEXT_SAMPLES * 3]
            ),
            reverse=True,
        )
        score = sum(score for score in scores[:MAX_CONTEXT_SAMPLES])
        _AGENT_FACT_SCORE_CACHE[key] = score
        self._agent_cited_score_cache[key] = score
        return score

    def _cited_fact_weight_multiplier(self, conn, next_agent):
        score = (
            self._connector_cited_fact_score(conn) * THOUGHT_CITED_FACT_WEIGHT
            + self._agent_cited_fact_score(next_agent) * THOUGHT_DESTINATION_CITED_FACT_WEIGHT
        )
        capped = min(float(THOUGHT_CITED_FACT_WEIGHT_CAP), max(0.0, score))
        return 1.0 + capped

    def _nearest_goal_distance(self, position):
        if position is None:
            return None
        distances = []
        for agent in self.goal_agents:
            goal_pos = getattr(agent, "pos", None)
            if goal_pos is None:
                continue
            try:
                distances.append(float(np.linalg.norm(position - goal_pos)))
            except Exception:
                continue
        if not distances:
            return None
        return min(distances)

    def _apply_spatial_progress_reward(self, previous_pos, next_pos):
        previous_distance = self._nearest_goal_distance(previous_pos)
        next_distance = self._nearest_goal_distance(next_pos)
        if previous_distance is None or next_distance is None:
            return 0.0
        if previous_distance <= 0.0 or next_distance >= previous_distance:
            return 0.0

        progress = (previous_distance - next_distance) / previous_distance
        recovery_room = max(0.0, STARTING_SCORE - float(self.score))
        reward = recovery_room * _clamp01(progress) * SPATIAL_PROGRESS_REWARD_RATE
        self.score = min(STARTING_SCORE, self.score + reward)
        return reward

    def _specific_detail_points(self, details):
        points = 0
        for detail in list(details or []):
            specificity = list(detail.get("specificity", ()) or ())
            value_count = 0
            for value in specificity[:3]:
                try:
                    value_count += max(0, int(value))
                except (TypeError, ValueError):
                    continue
            points += max(1, value_count)
        return points

    def _apply_specific_detail_reward(self, details):
        points = self._specific_detail_points(details)
        if points <= 0:
            return 0.0

        recovery_room = max(0.0, STARTING_SCORE - float(self.score))
        if recovery_room <= 0.0:
            return 0.0

        saturation = max(1.0, float(SPECIFIC_DETAIL_REWARD_SATURATION))
        specificity_share = points / (points + saturation)
        reward = recovery_room * _clamp01(specificity_share) * SPECIFIC_DETAIL_REWARD_RATE
        self.score = min(STARTING_SCORE, self.score + reward)
        return reward

    def _apply_connection_score_penalty(self, conn):
        utility = _clamp01(float(getattr(conn, "utility", 0.0) or 0.0))
        utility = max(_clamp01(UTILITY_FLOOR), utility)
        penalty_rate = _clamp01(THOUGHT_UTILITY_PENALTY_RATE)
        multiplier = 1.0 - ((1.0 - utility) * penalty_rate)
        multiplier = max(_clamp01(THOUGHT_MIN_STEP_SCORE_MULTIPLIER), min(1.0, multiplier))
        self.score *= multiplier
        return multiplier

    def valid_candidates(self):
        if not self.alive:
            return [], [], None
        hops, cached_distances = self._candidate_hops_for_current()
        if not hops:
            return [], [], self.current_asu.pos

        current_pos = self.current_asu.pos
        valid_candidates = []
        distances = []
        cited_multipliers = []
        tense_multipliers = []

        for hop, distance in zip(hops, cached_distances):
            conn, next_agent, _reversed_hop = hop
            if next_agent.index == self.last_asu_idx:
                continue

            valid_candidates.append(hop)
            distances.append(distance)
            cited_multipliers.append(self._cited_fact_weight_multiplier(conn, next_agent))
            tense_multipliers.append(self._tense_preference_multiplier(conn))

        distance_weights = self.relative_distance_weights(distances)
        weights = [
            max(0.01, distance_weight) * max(1.0, cited_multiplier) * max(0.25, tense_multiplier)
            for distance_weight, cited_multiplier, tense_multiplier in zip(
                distance_weights,
                cited_multipliers,
                tense_multipliers,
            )
        ]
        return valid_candidates, weights, current_pos

    def _weighted_unique_sample(self, candidates, weights, k):
        remaining_candidates = list(candidates)
        remaining_weights = list(weights)
        sampled = []
        sample_count = min(k, len(remaining_candidates))
        for _ in range(sample_count):
            if not remaining_candidates:
                break
            choice = random.choices(remaining_candidates, weights=remaining_weights, k=1)[0]
            idx = remaining_candidates.index(choice)
            sampled.append(choice)
            remaining_candidates.pop(idx)
            remaining_weights.pop(idx)
        return sampled

    def _context_records_for_agent(self, agent):
        if agent is None:
            return []
        concept_id = existing_concept_index(agent.ASU)
        if concept_id < 0:
            return []
        return ConnectionEndpoint.contexts_from_idx(concept_id)

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

    def _detail_from_record(self, record, utility=0.0):
        text = _clean_text((record or {}).get("text", ""))
        source = _clean_text((record or {}).get("source", "unknown"))
        if not text:
            return ""
        if self._should_cite(record, utility=utility):
            if source not in self.collected_sources:
                self.collected_sources.append(source)
            return f"{text} [{display_source(source)}]"
        return text

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

    def gather_stop_details(self, extra_connections=None):
        records = self._context_records_for_agent(self.current_asu)
        for record in records[:MAX_CONTEXT_SAMPLES]:
            detail = self._detail_from_record(record)
            if detail:
                self._remember_note(detail)

        for hop in list(extra_connections or [])[:MAX_ADJACENT_SAMPLES]:
            conn, next_agent, _reversed_hop = self._hop_parts(hop)
            if conn is None or next_agent is None:
                continue
            detail = f"Adjacent topic: {next_agent.ASU}"
            target_records = self._context_records_for_agent(next_agent)
            if target_records:
                detail = f"{detail} | {self._detail_from_record(target_records[0], utility=getattr(conn, 'utility', 0.0))}"
            self._remember_note(detail, adjacent=True)

    def _connector_identity(self, conn):
        source_agent = getattr(conn, "source_agent", None)
        target = getattr(conn, "target", None)
        return (
            getattr(source_agent, "index", -1),
            getattr(target, "index", -1),
            getattr(conn, "relation_index", -1),
            _clean_text(getattr(conn, "evidence_text", "")),
        )

    def _predicate_text(self, conn):
        target = getattr(conn, "target", None)
        if target is None:
            return ""
        predicate_text = _clean_text(getattr(target, "ASU", ""))
        predicate_endpoint = getattr(conn, "predicate", None)
        modifiers = " ".join(predicate_endpoint.modifier_value()).strip() if predicate_endpoint else ""
        if modifiers:
            predicate_text = f"{modifiers} {predicate_text}".strip()
        return predicate_text

    def _specific_value_key(self, value):
        if isinstance(value, dict):
            return repr(sorted(value.items()))
        return _clean_text(value).lower()

    def _structured_specific_values(self, conn):
        values = []
        seen = set()
        for attr in ("predicate_specifics", "connection_specifics", "subject_specifics"):
            for value in list(getattr(conn, attr, []) or []):
                key = self._specific_value_key(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                values.append(value)
        return values

    def _temporal_context_from_specifics(self, specifics):
        for value in list(specifics or []):
            if isinstance(value, dict) and value.get("kind") == "temporal_context":
                return dict(value)
        return {}

    def _connection_temporal_context(self, conn):
        return self._temporal_context_from_specifics(getattr(conn, "connection_specifics", []))

    def _connection_tense_score(self, conn):
        temporal = self._connection_temporal_context(conn)
        try:
            return max(-1.0, min(1.0, float(temporal.get("tense_score", 0.0))))
        except (TypeError, ValueError):
            pass

        current_tense = _clean_text(temporal.get("current_tense", "")).lower()
        if current_tense == "past":
            return -1.0
        if current_tense == "future":
            return 1.0
        return 0.0

    def _tense_preference_multiplier(self, conn):
        preferred_score = self._preferred_tense_score()
        if preferred_score == 0.0:
            return 1.0

        alignment = preferred_score * self._connection_tense_score(conn)
        if alignment > 0.0:
            return 1.0 + (alignment * float(THOUGHT_TENSE_PREFERENCE_WEIGHT))
        if alignment < 0.0:
            return max(0.25, 1.0 + (alignment * float(THOUGHT_TENSE_OPPOSITION_PENALTY)))
        return 1.0

    def _predicate_specific_values(self, conn):
        values = []
        seen = set()
        for value in list(getattr(conn, "predicate_specifics", []) or []):
            key = self._specific_value_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            values.append(value)

        for value in SPECIFIC_DETAIL_RE.findall(self._predicate_text(conn)):
            key = self._specific_value_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            values.append(value)
        return values

    def _predicate_has_specific_data(self, conn):
        return bool(self._predicate_specific_values(conn))

    def _specificity_key(self, conn, clause):
        predicate_values = self._predicate_specific_values(conn)
        structured_values = self._structured_specific_values(conn)
        clause_values = {
            self._specific_value_key(value)
            for value in SPECIFIC_DETAIL_RE.findall(clause)
            if self._specific_value_key(value)
        }
        return (
            len(predicate_values),
            len(structured_values),
            len(clause_values),
            len(_clean_text(clause)),
        )

    def _agent_specific_context_details(self):
        connectors = list(getattr(self.current_asu, "connectors", []) or [])
        cache_key = (getattr(self.current_asu, "index", -1), len(connectors))
        cached = _SPECIFIC_CONTEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        details = []
        for conn in connectors:
            if self._candidate_hop(conn) is None:
                continue
            source = _clean_text(getattr(conn, "source", ""))
            if not self._source_is_citable(source):
                continue
            if not self._predicate_has_specific_data(conn):
                continue
            clause = self._connection_clause_text(conn)
            if not clause:
                continue
            details.append(
                {
                    "identity": self._connector_identity(conn),
                    "text": clause,
                    "source": source,
                    "specificity": self._specificity_key(conn, clause),
                    "predicate_specifics": list(getattr(conn, "predicate_specifics", []) or []),
                    "temporal": self._connection_temporal_context(conn),
                }
            )

        details.sort(key=lambda item: (item["specificity"], item["text"]), reverse=True)
        _SPECIFIC_CONTEXT_CACHE[cache_key] = details
        return details

    def _connection_clause_text(self, conn):
        evidence_text = _clean_text(getattr(conn, "evidence_text", ""))
        if evidence_text:
            return evidence_text.rstrip(".") + "."
        subject_agent = getattr(conn, "source_agent", None)
        subject = _clean_text(getattr(subject_agent, "ASU", ""))
        predicate = self._predicate_text(conn)
        relation = literal_from_index(_optional_int(getattr(conn, "relation_index", -1))) or ""
        relation = _clean_text(relation)
        if not subject or not relation or not predicate:
            return ""
        return f"{subject} {relation} {predicate}".strip().rstrip(".") + "."

    def _specific_context_details(self, chosen_conn, limit=3):
        chosen_identity = self._connector_identity(chosen_conn)
        details = []
        for detail in self._agent_specific_context_details():
            if detail.get("identity") == chosen_identity:
                continue
            clean_detail = dict(detail)
            clean_detail.pop("identity", None)
            details.append(clean_detail)
            if len(details) >= limit:
                break
        return details[:limit]

    def choose_next_hop(self):
        candidates, weights, current_pos = self.valid_candidates()
        if not candidates:
            return None, [], current_pos
        preview = self._weighted_unique_sample(candidates, weights, MAX_ADJACENT_SAMPLES)
        choice = random.choices(candidates, weights=weights, k=1)[0]
        return choice, preview, current_pos

    def _target_tokens(self, text):
        return [
            token
            for token in TARGET_TOKEN_RE.findall(_clean_text(text).lower())
            if token and token not in STOPWORD_TOKENS
        ]

    def _lexical_match(self, candidate, target):
        candidate = _clean_text(candidate)
        target = _clean_text(target)
        key = (candidate.lower(), target.lower())
        cached = _LEXICAL_MATCH_CACHE.get(key)
        if cached is not None:
            return cached

        candidate_tokens = set(self._target_tokens(candidate))
        target_tokens = set(self._target_tokens(target))
        if not candidate_tokens or not target_tokens:
            _LEXICAL_MATCH_CACHE[key] = 0.0
            return 0.0
        overlap = candidate_tokens & target_tokens
        if not overlap:
            _LEXICAL_MATCH_CACHE[key] = 0.0
            return 0.0
        precision = len(overlap) / len(candidate_tokens)
        recall = len(overlap) / len(target_tokens)
        score = max(precision, 0.85 * recall)
        _LEXICAL_MATCH_CACHE[key] = score
        return score

    def _grounding_lexical_match(self, candidate, target):
        candidate_tokens = set(self._target_tokens(candidate))
        target_tokens = set(self._target_tokens(target))
        required_candidate_tokens = min(2, len(target_tokens))
        if len(candidate_tokens) < required_candidate_tokens:
            return 0.0
        return self._lexical_match(candidate, target)

    def _candidate_match_score(self, candidate, target, semantic=True):
        candidate = _clean_text(candidate)
        target = _clean_text(target)
        if not candidate or not target:
            return 0.0
        key = (candidate.lower(), target.lower(), bool(semantic))
        cached = _MATCH_SCORE_CACHE.get(key)
        if cached is not None:
            return cached

        lexical = self._lexical_match(candidate, target)
        score = lexical
        if semantic:
            score = max(score, _clamp01(text_similarity(candidate, target)))
        _MATCH_SCORE_CACHE[key] = score
        return score

    def _best_lexical_target_match(self, candidates, target, related_queries=None):
        target = _clean_text(target)
        if not target:
            return 0.0

        related_queries = [
            _clean_text(query)
            for query in list(related_queries or [])
            if _clean_text(query)
        ]
        candidate_texts = [_clean_text(candidate) for candidate in list(candidates or []) if _clean_text(candidate)]
        if not candidate_texts:
            return 0.0

        best = 0.0
        for candidate in candidate_texts:
            best = max(best, self._grounding_lexical_match(candidate, target))
            for query in related_queries:
                best = max(best, 0.92 * self._grounding_lexical_match(candidate, query))
        return best

    def _best_target_match(self, candidates, target, related_queries=None, semantic=True):
        target = _clean_text(target)
        if not target:
            return 0.0

        related_queries = [
            _clean_text(query)
            for query in list(related_queries or [])
            if _clean_text(query)
        ]
        candidate_texts = [_clean_text(candidate) for candidate in list(candidates or []) if _clean_text(candidate)]
        if not candidate_texts:
            return 0.0

        best = 0.0
        for candidate in candidate_texts:
            best = max(best, self._candidate_match_score(candidate, target, semantic=semantic))
            for query in related_queries:
                best = max(best, 0.92 * self._candidate_match_score(candidate, query, semantic=semantic))
        return best

    def _step_clause(self, step):
        if not isinstance(step, dict):
            return ""
        evidence_text = _clean_text(step.get("evidence_text", ""))
        if evidence_text:
            return evidence_text.rstrip(".") + "."
        subject = _clean_text(step.get("subject", ""))
        predicate = _clean_text(step.get("predicate", ""))
        relation_id = step.get("relation_id", -1)
        relation = literal_from_index(_optional_int(relation_id, default=-1)) or ""
        relation = _clean_text(relation)
        if not subject or not relation or not predicate:
            return ""
        return f"{subject} {relation} {predicate}".strip().rstrip(".") + "."

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
            records.append(
                {
                    "text": text.rstrip(".") + ".",
                    "source": _clean_text(detail.get("source", "")),
                    "specificity": list(detail.get("specificity", ()) or ()),
                    "temporal": dict(detail.get("temporal") or {}),
                    "role": "specific_context",
                }
            )
        return records

    def _step_temporal_context(self, step):
        if not isinstance(step, dict):
            return {}
        temporal = step.get("temporal")
        if isinstance(temporal, dict):
            return dict(temporal)
        return self._temporal_context_from_specifics(step.get("connection_specifics", []))

    def _path_texts(self):
        texts = []
        seen = set()

        def append_text(value):
            clean = _clean_text(value)
            if not clean or clean in seen:
                return
            seen.add(clean)
            texts.append(clean)

        append_text(self.history[0].get("node", ""))
        append_text(self.current_asu.ASU)
        for step in self.history[1:]:
            append_text(step.get("subject", ""))
            append_text(step.get("predicate", ""))
            append_text(step.get("evidence_text", ""))
            append_text(self._step_clause(step))
            for record in self._step_specific_records(step):
                append_text(record.get("text", ""))
        return texts

    def _current_target_b_match(self):
        current_texts = [self.current_asu.ASU]
        for step in self.history[-MAX_CONTEXT_SAMPLES:]:
            if not isinstance(step, dict):
                continue
            current_texts.append(self._step_clause(step))
            current_texts.append(step.get("predicate", ""))
            for record in self._step_specific_records(step):
                current_texts.append(record.get("text", ""))
        return self._best_target_match(current_texts, self.target_b, self.success_queries)

    def _current_target_b_lexical_match(self):
        current_texts = [self.current_asu.ASU]
        for step in self.history[-MAX_CONTEXT_SAMPLES:]:
            if not isinstance(step, dict):
                continue
            current_texts.append(self._step_clause(step))
            current_texts.append(step.get("predicate", ""))
            for record in self._step_specific_records(step):
                current_texts.append(record.get("text", ""))
        return self._best_lexical_target_match(current_texts, self.target_b, self.success_queries)

    def _path_target_matches(self):
        path_texts = self._path_texts()
        target_a_match = self._best_target_match(path_texts, self.start_target)
        target_b_match = self._best_target_match(path_texts, self.target_b, self.success_queries)
        return target_a_match, target_b_match

    def _path_target_lexical_matches(self):
        path_texts = self._path_texts()
        target_a_match = self._best_lexical_target_match(path_texts, self.start_target)
        target_b_match = self._best_lexical_target_match(path_texts, self.target_b, self.success_queries)
        return target_a_match, target_b_match

    def _should_run_semantic_bridge_check(self):
        hop_count = len(self.history) - 1
        if hop_count < MIN_SUCCESS_STEPS:
            return False
        interval = max(1, int(THOUGHT_SEMANTIC_CHECK_INTERVAL))
        return hop_count >= MAX_THOUGHT_HOPS or hop_count % interval == 0

    def _has_grounded_bridge(self):
        if (len(self.history) - 1) < MIN_SUCCESS_STEPS:
            return False

        target_a_lexical, target_b_lexical = self._path_target_lexical_matches()
        current_b_lexical = self._current_target_b_lexical_match()
        if (
            target_a_lexical < PATH_TARGET_MATCH_THRESHOLD
            or target_b_lexical < PATH_TARGET_MATCH_THRESHOLD
            or max(current_b_lexical, target_b_lexical) < PATH_TARGET_MATCH_THRESHOLD
        ):
            return False

        if not self._should_run_semantic_bridge_check():
            return False

        target_a_match, target_b_match = self._path_target_matches()
        current_b_match = self._current_target_b_match()
        return (
            target_a_match >= PATH_TARGET_MATCH_THRESHOLD
            and target_b_match >= PATH_TARGET_MATCH_THRESHOLD
            and target_a_lexical >= PATH_TARGET_MATCH_THRESHOLD
            and target_b_lexical >= PATH_TARGET_MATCH_THRESHOLD
            and max(current_b_match, target_b_match) >= SUCCESS_SIMILARITY_THRESHOLD
        )

    def _confidence_label(self, target_a_match, target_b_match):
        return self._confidence_label_for_score(self.score, target_a_match, target_b_match)

    def _confidence_label_for_score(self, support_score, target_a_match, target_b_match):
        blended = ((float(support_score or 0.0) / 100.0) + target_a_match + target_b_match) / 3.0
        if blended >= 0.72:
            return "strong"
        if blended >= 0.54:
            return "moderate"
        return "tentative"

    def _source_key(self, source):
        display = display_source(source).strip()
        if not display or display == "unknown":
            return ""
        return display.rstrip("/").lower()

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

        total = sum(source_counts.values())
        distinct = len(source_counts)
        if total <= 0 or distinct <= 0:
            return {
                "distinct_urls": 0,
                "cited_records": 0,
                "dominant_url_share": 0.0,
                "source_diversity_multiplier": ARGUMENT_SOURCE_DIVERSITY_MIN_MULTIPLIER,
            }

        dominant_share = max(source_counts.values()) / total
        target = max(1, int(ARGUMENT_SOURCE_DIVERSITY_TARGET))
        floor = _clamp01(ARGUMENT_SOURCE_DIVERSITY_MIN_MULTIPLIER)
        distinct_multiplier = min(1.0, distinct / target)
        concentration_multiplier = max(floor, 1.0 - max(0.0, dominant_share - (1.0 / target)))
        multiplier = max(floor, min(distinct_multiplier, concentration_multiplier))
        return {
            "distinct_urls": distinct,
            "cited_records": total,
            "dominant_url_share": round(dominant_share, 3),
            "source_diversity_multiplier": round(multiplier, 3),
        }

    def _build_relationship_statement(self):
        start = self.start_target or self.seed_query or self.history[0].get("node", "")
        end = self.target_b or self.current_asu.ASU
        path_clauses = []
        clause_records = []
        for step in self.history[1:]:
            clause = self._step_clause(step)
            if clause:
                path_clauses.append(clause)
                clause_records.append(
                    {
                        "text": clause,
                        "source": _clean_text(step.get("source", "")),
                        "role": "path",
                        "temporal": self._step_temporal_context(step),
                    }
                )
            clause_records.extend(self._step_specific_records(step))

        recent_records = clause_records[-(6 + (6 * 3)):]
        supporting_clauses = [
            record["text"]
            for record in recent_records
            if record.get("role") == "specific_context"
        ]
        target_a_match, target_b_match = self._path_target_matches()
        target_a_lexical, target_b_lexical = self._path_target_lexical_matches()
        summary = " ".join(path_clauses[-3:]) if path_clauses else f"{start} -> {end}"
        source_diversity = self._source_diversity_stats(recent_records)
        adjusted_score = round(
            float(self.score) * float(source_diversity["source_diversity_multiplier"]),
            2,
        )
        return {
            "statement": summary,
            "support_score": adjusted_score,
            "raw_support_score": round(self.score, 2),
            "confidence": self._confidence_label_for_score(adjusted_score, target_a_match, target_b_match),
            "match_target_a": round(target_a_match, 3),
            "match_target_b": round(target_b_match, 3),
            "lexical_target_a": round(target_a_lexical, 3),
            "lexical_target_b": round(target_b_lexical, 3),
            "source_diversity": source_diversity,
            "path_clauses": path_clauses[-6:],
            "supporting_clauses": supporting_clauses,
            "clause_records": recent_records,
            "endpoint": self.current_asu.ASU,
            "sources": list(self.collected_sources),
            "notes": list(self.collected_notes),
            "target_a": self.start_target or start,
            "target_b": self.target_b or end,
        }

    def move(self):
        if not self.alive:
            return False
        if (len(self.history) - 1) >= MAX_THOUGHT_HOPS:
            self.alive = False
            self.termination_reason = "max_hops"
            return False

        hop, preview, current_pos = self.choose_next_hop()
        self.gather_stop_details(extra_connections=preview)
        conn, next_agent, reversed_hop = self._hop_parts(hop)
        if conn is None or next_agent is None:
            self.alive = False
            self.termination_reason = "endpoint"
            return False

        temporal_context = self._connection_temporal_context(conn)
        specific_details = self._specific_context_details(conn)
        for detail in specific_details:
            source = detail.get("source", "")
            if self._source_is_citable(source) and source not in self.collected_sources:
                self.collected_sources.append(source)
            self._remember_note(f"{detail.get('text', '')} [{display_source(source)}]")

        self.last_asu_idx = self.current_asu.index
        self.current_asu = next_agent
        previous_pos = current_pos
        next_pos = self.current_asu.pos
        score_multiplier = self._apply_connection_score_penalty(conn)
        spatial_reward = self._apply_spatial_progress_reward(previous_pos, next_pos)
        specific_reward = self._apply_specific_detail_reward(specific_details)
        if self._source_is_citable(conn.source) and conn.source not in self.collected_sources:
            self.collected_sources.append(conn.source)

        subject_text = conn.source_agent.ASU if conn.source_agent is not None else ""
        predicate_text = conn.target.ASU if conn.target is not None else ""
        if conn.subject.modifier_idx:
            subject_mods = " ".join(conn.subject.modifier_value()).strip()
            if subject_mods:
                subject_text = f"{subject_mods} {subject_text}".strip()
        if conn.predicate.modifier_idx:
            predicate_mods = " ".join(conn.predicate.modifier_value()).strip()
            if predicate_mods:
                predicate_text = f"{predicate_mods} {predicate_text}".strip()

        if conn.evidence_text:
            self._remember_note(self._detail_from_record({"text": conn.evidence_text, "source": conn.source}, utility=getattr(conn, "utility", 0.0)))

        self.history.append(
            {
                "subject": subject_text,
                "subject_truth": conn.subject.truth,
                "subject_tense": conn.subject.tense,
                "subject_quant": conn.subject.quantifier,
                "subject_modifier": " ".join(conn.subject.modifier_value()).strip(),
                "subject_specifics": list(conn.subject_specifics),
                "relation_id": conn.relation_index,
                "truth": conn.truth,
                "predicate": predicate_text,
                "predicate_truth": conn.predicate.truth,
                "predicate_tense": conn.predicate.tense,
                "predicate_quant": conn.predicate.quantifier,
                "predicate_modifier": " ".join(conn.predicate.modifier_value()).strip(),
                "predicate_specifics": list(conn.predicate_specifics),
                "connection_specifics": list(conn.connection_specifics),
                "temporal": temporal_context,
                "previous_agent_ids": list(conn.previous_agent_ids),
                "previous_agent": "",
                "source": conn.source,
                "evidence_text": conn.evidence_text,
                "specific_details": specific_details,
                "traversal_reversed": reversed_hop,
                "utility": round(float(getattr(conn, "utility", 0.0) or 0.0), 4),
                "score_multiplier": round(score_multiplier, 4),
                "spatial_reward": round(spatial_reward, 4),
                "specific_reward": round(specific_reward, 4),
                "score": round(self.score, 2),
                "adjacent_topics": [
                    sample_next.ASU
                    for _sample_conn, sample_next, _sample_reversed in (
                        self._hop_parts(sample)
                        for sample in preview
                    )
                    if sample_next is not None
                ],
                "details": list(self.collected_notes[-3:]),
            }
        )

        self.gather_stop_details()

        if self._has_grounded_bridge():
            self.successful = True
            self.success_payload = self._build_relationship_statement()
            self.alive = False
            self.termination_reason = "endpoint"
            return True

        if self.score < LOW_SCORE_THRESHOLD:
            self.alive = False
            self.termination_reason = "dead"
        return True
