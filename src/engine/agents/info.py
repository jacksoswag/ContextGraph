import numpy as np  # type: ignore
from typing import List, Optional; from engine.agents.connection import Connector; from engine.common.constants import AGENT_SEMANTIC_SEED_SCALE; from engine.extract.word_info_map import concept_vector_from_text, str_to_vector
# Represents one semantic graph node with a stable id, vector, position, and connectors.
class Info_Agent:
    _projection_matrices = {} # maps info vector into 3D space (normalized to 0-1)
    _semantic_seed_scale = AGENT_SEMANTIC_SEED_SCALE # radius of initial 3D semantic space
    # Initializes this object from caller-provided state.
    def __init__(self, index: int, ASU: str = "", pos: Optional[np.ndarray] = None, asu_info_ref=None):
        self.index = index # unique ID
        self._asu_info_ref = self._normalize_asu_info_ref(ASU, asu_info_ref) # list: [ASU text, info_vector]
        self.pos = pos if pos is not None else np.zeros(3, dtype=np.float32) # the physical location of the agent
        self.connectors: List["Connector"] = [] # connections touching this agent
    # Normalizes coerce vector for agent identity.
    @staticmethod
    def _coerce_vector(vector): # turns inefficient python data in a clean numpy array
        if vector is None: return np.zeros(0, dtype=np.float32) # handles 0-dimensional vectors (edge case)
        array = np.asarray(vector, dtype=np.float32).reshape(-1) # converts to numpy array
        return array
    # Builds projection matrix for agent identity.
    @classmethod
    def _projection_matrix(cls, dimension): # converts n-dimensional vectors to 3D vectors (normalized to 0-1)
        dimension = int(max(0, dimension))
        if dimension == 0: return np.zeros((0, 3), dtype=np.float32) # handles 0-dimensional vectors (edge case)
        matrix = cls._projection_matrices.get(dimension)
        if matrix is not None: return matrix # returns matrix if it exists
        rng = np.random.default_rng(1000 + dimension) # generates random number
        matrix = rng.normal(0.0, 1.0, size=(dimension, 3)).astype(np.float32) # creates matrix
        matrix /= max(1.0, np.sqrt(float(dimension))) # normalizes matrix
        cls._projection_matrices[dimension] = matrix # stores matrix
        return matrix
    # Projects a semantic vector into a deterministic 3D seed position.
    @classmethod
    def semantic_seed_position_for_vector(cls, vector, scale=None): # the physical location of the agent
        array = cls._coerce_vector(vector)
        if array.size == 0: return np.zeros(3, dtype=np.float32) # handles 0-dimensional vectors (edge case)
        matrix = cls._projection_matrix(array.size) # converts n-dimensional vectors to 3D vectors (normalized to 0-1)
        if matrix.size == 0: return np.zeros(3, dtype=np.float32) # handles 0-dimensional vectors (edge case)
        projected = np.matmul(array, matrix).astype(np.float32) # multiplies matrix by vector
        norm = float(np.linalg.norm(projected)) # normalizes vector
        if norm > 1e-12: projected = projected / norm
        seed_scale = float(scale if scale is not None else cls._semantic_seed_scale)
        return (projected * seed_scale).astype(np.float32)
    # Builds a semantic vector for ASU text, falling back to zeros on embedding failure.
    @classmethod
    def _vector_for_asu(cls, asu): # turns asu into an info vector
        text = str(asu or "").strip()
        if not text: return np.zeros(0, dtype=np.float32) # handles 0-dimensional vectors (edge case)
        stored = concept_vector_from_text(text) # gets stored vector
        if stored: return cls._coerce_vector(stored) # returns stored vector if it exists
        return cls._coerce_vector(str_to_vector(text)) # returns vector from text
    # Normalizes asu info ref for agent identity.
    @classmethod
    def _normalize_asu_info_ref(cls, ASU, asu_info_ref): # packs ASU and info_vector into one object
        if isinstance(asu_info_ref, list) and len(asu_info_ref) >= 2:
            asu_info_ref[0] = str(asu_info_ref[0] or ASU or "").strip(); asu_info_ref[1] = cls._coerce_vector(asu_info_ref[1])
            return asu_info_ref
        text = str(ASU or "").strip()
        return [text, cls._vector_for_asu(text)]
    # Returns asu info ref for agent identity.
    @property
    def asu_info_ref(self): return self._asu_info_ref # getter for asu_info_ref
    # Returns or sets the agent concept text while keeping ids and vectors synchronized.
    @property
    def ASU(self): return self._asu_info_ref[0] # getter for ASU
    # Returns or sets the agent concept text while keeping ids and vectors synchronized.
    @ASU.setter
    def ASU(self, value): # setter for agent
        text = str(value or "").strip(); self._asu_info_ref[0] = text; self._asu_info_ref[1] = self._vector_for_asu(text)
    # Returns or sets the vector used for semantic merge and seeded position logic.
    @property
    def info_vector(self): return self._asu_info_ref[1] # getter for info_vector
    # Returns or sets the vector used for semantic merge and seeded position logic.
    @info_vector.setter
    def info_vector(self, value): # setter for agent's info vector
        self._asu_info_ref[1] = self._coerce_vector(value)
    # Returns the deterministic 3D seed position derived from the agent vector.
    def semantic_seed_position(self, scale=None): # the starting location of the agent
        return self.semantic_seed_position_for_vector(self.info_vector, scale=scale)
