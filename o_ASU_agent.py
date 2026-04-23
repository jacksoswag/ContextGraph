import numpy as np # type: ignore
from typing import List, Optional
from o_connection import Connector

class ASU_Agent: # An agent that stores an ASU (Atomic Semantic Unit) and its logical connection (LC) with other agents.    
    def __init__(self, index: int, ASU: str = "", pos: Optional[np.ndarray] = None):
        self.index = index
        self.ASU = ASU # word
        self.pos = pos if pos is not None else np.zeros(3, dtype=np.float32) # position in simulation
        self.connectors: List['Connector'] = [] # A list of Connections from this agent