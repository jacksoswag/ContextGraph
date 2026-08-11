import struct; from engine.extract.word_info_map import concept_from_index, connector_utility, get_or_create_index, get_literal_index, literal_from_index, str_to_vector; from engine.common.constants import CONNECTION_UTILITY_OFFSET, ENDPOINT_CONTEXT_LIMIT
# Stores endpoint concept, quantifier, tense, truth, and modifier ids for a connection side.
class ConnectionEndpoint:
    __slots__ = ("quantifier", "tense", "truth", "modifier_idx", "subject_predicate_id"); _context_registry = {}
    # Initializes this object from caller-provided state.
    def __init__(self, quantifier: int, tense: int, truth: int, ASU_idx, modifier_idx=None):
        self.quantifier = self._normalize_optional_id(quantifier, default=-1); self.tense = self._normalize_optional_id(tense, default=-1); self.truth = self._normalize_optional_id(truth, default=-1); self.subject_predicate_id = self._normalize_asu_id(ASU_idx); self.modifier_idx = self._normalize_modifier_ids(modifier_idx)
    # Normalizes id for connection endpoints.
    @staticmethod
    def _normalize_id(value, default=0):
        try: return int(value)
        except (TypeError, ValueError): return default
    # Normalizes optional id for connection endpoints.
    @classmethod
    def _normalize_optional_id(cls, value, default=-1):
        normalized = cls._normalize_id(value, default=default)
        return normalized if normalized >= -1 else default
    # Normalizes asu id for connection endpoints.
    @classmethod
    def _normalize_asu_id(cls, value):
        if isinstance(value, int): return cls._normalize_optional_id(value, default=-1)
        text = " ".join(str(value or "").strip().split())
        if not text: return -1
        try: return int(get_or_create_index(text, str_to_vector(text)))
        except Exception: return -1
    # Normalizes modifier ids for connection endpoints.
    @classmethod
    def _normalize_modifier_ids(cls, modifier_idx):
        if modifier_idx in (None, "", []): return ()
        if isinstance(modifier_idx, int): modifier_idx = [modifier_idx]
        values = []; seen = set()
        for value in list(modifier_idx or []):
            normalized = cls._normalize_optional_id(value, default=-1)
            if normalized < 0 or normalized in seen: continue
            seen.add(normalized); values.append(normalized)
        return tuple(values)
    # Returns or sets the endpoint concept id used by packed connection records.
    @property
    def ASU_idx(self):
        return self.subject_predicate_id
    # Returns or sets the endpoint concept id used by packed connection records.
    @ASU_idx.setter
    def ASU_idx(self, value):
        self.subject_predicate_id = self._normalize_asu_id(value)
    # Returns endpoint concept text for a stored concept id.
    @classmethod
    def asu_from_idx(cls, idx):
        return concept_from_index(cls._normalize_optional_id(idx, default=-1))
    # Returns this endpoint concept as display text.
    def asu_value(self):
        return self.asu_from_idx(self.subject_predicate_id)
    # Stores modifier text as a literal id for endpoint metadata.
    @classmethod
    def register_modifier(cls, modifier):
        if isinstance(modifier, int): return cls._normalize_optional_id(modifier, default=-1)
        text = " ".join(str(modifier or "").strip().split())
        return get_literal_index(text) if text else -1
    # Returns modifier literal text for this endpoint.
    def modifier_value(self):
        return [literal_from_index(idx) for idx in self.modifier_idx if idx >= 0]
    # Stores citable source context for an endpoint concept id.
    @classmethod
    def register_context(cls, asu_idx, text="", source="unknown"):
        normalized_idx = cls._normalize_optional_id(asu_idx, default=-1); clean_text = " ".join(str(text or "").strip().split()); clean_source = " ".join(str(source or "").strip().split()) or "unknown"
        if normalized_idx < 0 or not clean_text: return
        records = cls._context_registry.setdefault(normalized_idx, []); record = {"text": clean_text, "source": clean_source}
        if record in records: return
        records.append(record)
        if len(records) > ENDPOINT_CONTEXT_LIMIT: del records[:-ENDPOINT_CONTEXT_LIMIT]
    # Returns recent citable context records for an endpoint concept id.
    @classmethod
    def contexts_from_idx(cls, idx):
        normalized_idx = cls._normalize_optional_id(idx, default=-1)
        if normalized_idx < 0: return []
        return list(cls._context_registry.get(normalized_idx, ()))
    # Clears contexts caches or state.
    @classmethod
    def clear_contexts(cls): cls._context_registry.clear()
    # Returns pickle reconstruction data for multiprocessing transfer.
    def __reduce__(self):
        return (self.__class__, (self.quantifier, self.tense, self.truth, self.subject_predicate_id, list(self.modifier_idx)))
    # Returns a readable string form for logging and debugging.
    def __str__(self):
        return f"<ConnectionEndpoint: Q[{self.quantifier}] T[{self.tense}] TR[{self.truth}] MOD[{list(self.modifier_idx)}] ID[{self.subject_predicate_id}]>"
# Wraps a packed connection record with Python-side agents, metadata, and utility scoring.
class Connector:
    __slots__ = ("subject", "relation_index", "predicate", "utility", "source", "evidence_text", "subject_specifics", "predicate_specifics", "connection_specifics", "previous_agent_ids", "_agent_map", "_subject_idx", "_predicate_idx") # tells python to only allocate space for these variables
    # Initializes this object from caller-provided state.
    def __init__(self, connection_buff, offset, agent_map, subject_sp=None, predicate_sp=None, source="unknown", evidence_text="", subject_specifics=None, predicate_specifics=None, connection_specifics=None, previous_agent_ids=None):
        subject_id = int.from_bytes(connection_buff[offset : offset + 4], "little"); self.relation_index = int.from_bytes(connection_buff[offset + 4 : offset + 8], "little"); predicate_id = int.from_bytes(connection_buff[offset + 8 : offset + 12], "little")
        self.subject = subject_sp if isinstance(subject_sp, ConnectionEndpoint) else ConnectionEndpoint(-1, -1, -1, subject_id); self.predicate = predicate_sp if isinstance(predicate_sp, ConnectionEndpoint) else ConnectionEndpoint(-1, -1, -1, predicate_id); self.source = str(source).strip() or "unknown"
        self.evidence_text = " ".join(str(evidence_text or "").strip().split())
        self.subject_specifics = self._merged_specifics(subject_specifics) # stores modifiers and specific data (dates/numbers/percents)
        self.predicate_specifics = self._merged_specifics(predicate_specifics); self.connection_specifics = self._merged_specifics(connection_specifics); self.previous_agent_ids = tuple(sorted({int(value) for value in list(previous_agent_ids or []) if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit())}))
        self._subject_idx = subject_id; self._predicate_idx = predicate_id; self._agent_map = agent_map; self.utility = 0.0
        try: self.utility = float(struct.unpack_from("<f", connection_buff, offset + CONNECTION_UTILITY_OFFSET)[0])
        except struct.error: self.utility = 0.0
        if not (0.0 <= self.utility <= 1.0): self.utility = self.generate_utility()
    # Builds merged specifics for connection endpoints.
    @classmethod
    def _merged_specifics(cls, specifics=None): # merges specific data (dates/numbers/percents)
        merged = []; seen = set()
        for item in list(specifics or []):
            key = cls._specific_key(item)
            if not key or key in seen: continue
            seen.add(key); merged.append(item)
        return merged
    # Returns specific key for connection endpoints.
    @staticmethod
    def _specific_key(item): # turns specific data into a unique key
        if isinstance(item, dict): return repr(sorted(item.items()))
        return " ".join(str(item or "").strip().lower().split())
    # Computes utility for a potential connection from endpoint ids and specifics.
    @classmethod
    def utility_for(cls, subject_sp, relation_index, predicate_sp, subject_specifics=None, predicate_specifics=None, connection_specifics=None, evidence_text=""): #
        subject_index = getattr(subject_sp, "ASU_idx", getattr(subject_sp, "index", -1)); predicate_index = getattr(predicate_sp, "ASU_idx", getattr(predicate_sp, "index", -1)); subject_modifiers = subject_sp.modifier_value() if isinstance(subject_sp, ConnectionEndpoint) else []
        predicate_modifiers = predicate_sp.modifier_value() if isinstance(predicate_sp, ConnectionEndpoint) else []
        return float(connector_utility(subject_index, relation_index, predicate_index, subject_specifics=subject_specifics, predicate_specifics=predicate_specifics, connection_specifics=connection_specifics, subject_modifiers=subject_modifiers, predicate_modifiers=predicate_modifiers, evidence_text=evidence_text,))
    # Computes this connector utility from endpoint metadata and evidence.
    def generate_utility(self): # generates utility for the connection
        return self.utility_for(self.subject, self.relation_index, self.predicate, subject_specifics=self.subject_specifics, predicate_specifics=self.predicate_specifics, connection_specifics=self.connection_specifics, evidence_text=self.evidence_text)
    # Returns the subject-side agent for this connector.
    @property
    def source_agent(self): # returns subject agent
        return self._agent_map.get(self._subject_idx)
    # Returns the predicate-side agent for this connector.
    @property
    def target(self): # returns predicate agent
        return self._agent_map.get(self._predicate_idx)
    # Returns the best available truth flag from predicate or subject endpoint metadata.
    @property
    def truth(self):
        if getattr(self.predicate, "truth", -1) >= 0: return self.predicate.truth
        if getattr(self.subject, "truth", -1) >= 0: return self.subject.truth
        return -1
