CONNECTION_TRUTH_SHIFT = 0
SUBJECT_QUANT_SHIFT = 1
SUBJECT_TENSE_SHIFT = 4
SUBJECT_TRUTH_SHIFT = 6
PREDICATE_QUANT_SHIFT = 7
PREDICATE_TENSE_SHIFT = 10
PREDICATE_TRUTH_SHIFT = 12


def encode_connection_flags(subject_sp, predicate_sp):
    subject_quant = max(0, min(7, int(getattr(subject_sp, "quantifier", 0))))
    subject_tense = max(0, min(3, int(getattr(subject_sp, "tense", 0))))
    subject_truth = 1 if int(getattr(subject_sp, "truth", 1)) else 0
    predicate_quant = max(0, min(7, int(getattr(predicate_sp, "quantifier", 0))))
    predicate_tense = max(0, min(3, int(getattr(predicate_sp, "tense", 0))))
    predicate_truth = 1 if int(getattr(predicate_sp, "truth", subject_truth)) else 0
    connection_truth = 1 if (subject_truth and predicate_truth) else 0

    return (
        (connection_truth << CONNECTION_TRUTH_SHIFT)
        | (subject_quant << SUBJECT_QUANT_SHIFT)
        | (subject_tense << SUBJECT_TENSE_SHIFT)
        | (subject_truth << SUBJECT_TRUTH_SHIFT)
        | (predicate_quant << PREDICATE_QUANT_SHIFT)
        | (predicate_tense << PREDICATE_TENSE_SHIFT)
        | (predicate_truth << PREDICATE_TRUTH_SHIFT)
    )


class Connector:
    def __init__(self, connection_buff, offset, agent_map, relation_labels, source="unknown"):
        self._data = connection_buff[offset : offset + 16]
        self._agent_map = agent_map
        self._relation_labels = relation_labels
        self.source = source

    @property
    def flags(self):
        return int.from_bytes(self._data[8:12], "little")

    @property
    def source_agent(self): # Agent that connects to target (renamed to avoid conflict with 'source' string)
        idx = int.from_bytes(self._data[0:4], "little")
        return self._agent_map.get(idx)

    @property
    def target(self): # Agent that source is connected to
        idx = int.from_bytes(self._data[12:16], "little")
        return self._agent_map.get(idx)

    @property
    def conn_type(self): # Relation id for either a seeded logical connector or a verb
        return int.from_bytes(self._data[4:8], "little")

    @property
    def relation_label(self):
        relation_id = self.conn_type
        if 0 <= relation_id < len(self._relation_labels):
            return self._relation_labels[relation_id]
        return f"relation_{relation_id}"

    @property
    def truth(self):
        return (self.flags >> CONNECTION_TRUTH_SHIFT) & 0b1

    @property
    def connection_truth(self):
        return self.truth

    @property
    def subject_quant(self):
        return (self.flags >> SUBJECT_QUANT_SHIFT) & 0b111

    @property
    def subject_tense(self):
        return (self.flags >> SUBJECT_TENSE_SHIFT) & 0b11

    @property
    def subject_truth(self):
        return (self.flags >> SUBJECT_TRUTH_SHIFT) & 0b1

    @property
    def predicate_quant(self):
        return (self.flags >> PREDICATE_QUANT_SHIFT) & 0b111

    @property
    def predicate_tense(self):
        return (self.flags >> PREDICATE_TENSE_SHIFT) & 0b11

    @property
    def predicate_truth(self):
        return (self.flags >> PREDICATE_TRUTH_SHIFT) & 0b1
