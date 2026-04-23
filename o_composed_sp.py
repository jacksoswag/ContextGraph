class composed_sp:  # stores quantifier, tense, truth, and ASU index
    stored_asus = []
    asu_to_idx = {}

    @classmethod
    def _normalize_asu(cls, asu):
        return " ".join(str(asu or "").strip().lower().split())

    @classmethod
    def register_asu(cls, asu):
        normalized = cls._normalize_asu(asu)
        if not normalized:
            return -1
        idx = cls.asu_to_idx.get(normalized)
        if idx is not None:
            return idx
        idx = len(cls.stored_asus)
        cls.stored_asus.append(normalized)
        cls.asu_to_idx[normalized] = idx
        return idx

    @classmethod
    def asu_from_idx(cls, idx):
        if isinstance(idx, int) and 0 <= idx < len(cls.stored_asus):
            return cls.stored_asus[idx]
        return ""

    def __init__(self, quantifier: int, tense: int, truth: int, ASU_idx):
        self.quantifier = quantifier
        self.tense = tense
        self.truth = truth
        self._asu_value = self.asu_from_idx(ASU_idx) if isinstance(ASU_idx, int) else self._normalize_asu(ASU_idx)
        self.ASU_idx = self.register_or_link_asu(self._asu_value)

    def register_or_link_asu(self, asu):
        idx = self.register_asu(asu)
        self.ASU_idx = idx
        if idx >= 0:
            self._asu_value = self.asu_from_idx(idx)
        return idx

    def asu_value(self):
        return self.asu_from_idx(self.ASU_idx)

    def __reduce__(self):
        return (self.__class__, (self.quantifier, self.tense, self.truth, self.asu_value()))

    def __str__(self):
        return f"<Composed S/P: Q[{self.quantifier}] T[{self.tense}] TR[{self.truth}] ASU[{self.ASU_idx}]>"
